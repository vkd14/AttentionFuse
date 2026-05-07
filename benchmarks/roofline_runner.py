"""Generate roofline.csv from measured AttnFuse kernel timings.

Measures latency for four representative attention variants at four sequence
lengths, computes TFLOPS and estimated HBM bandwidth, and writes a CSV
suitable for roofline_plot.py.

FLOP accounting
---------------
Dense / Causal / Causal+ALiBi: 4 × B × H × N × N × D
Sliding-window (W=256):         4 × B × H × N × min(2W, N) × D

The sliding-window formula uses the effective number of key-value pairs
each query actually attends to (at most 2W), which is the correct figure
for arithmetic intensity; using the dense formula would overcount FLOPs
and give tc_pct > 100%.

HBM bandwidth estimate (no Nsight Compute)
------------------------------------------
We estimate HBM traffic analytically:
  reads  = Q(B×H×N×D) + K(B×H×N×D_eff) + V(B×H×N×D_eff) + cos/sin if RoPE
  writes = O(B×H×N×D)
where D_eff for SW = D × (2W / N) ≤ D.
This is a lower bound; real traffic includes SMEM spills and tiling overhead.

Usage::

    python -m benchmarks.roofline_runner [--output results/roofline.csv]
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from pathlib import Path

import torch
import attnfuse as af

BATCH      = 4
NUM_HEADS  = 12
HEAD_DIM   = 64
DTYPE      = torch.float16
ELEM_BYTES = 2          # fp16
WARMUP     = 20
ITERS      = 100
WINDOW     = 256        # sliding-window size

PEAK_TFLOPS_F16 = 142.0   # RTX 3090 fp16 tensor-core peak
PEAK_HBM_GBS    = 936.0   # RTX 3090 GDDR6X bandwidth


@af.attention
def _dense(Q, K, V):
    return af.softmax(af.scaled_dot_product(Q, K)) @ V


@af.attention
def _causal(Q, K, V):
    return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V


@af.attention
def _sw(Q, K, V):
    return af.softmax(af.sliding_window(af.scaled_dot_product(Q, K), WINDOW)) @ V


@af.attention
def _causal_alibi(Q, K, V):
    s = af.scaled_dot_product(Q, K)
    s = af.alibi(s, num_heads=NUM_HEADS)
    s = af.causal(s)
    return af.softmax(s) @ V


def _bench_ms(fn, *args, **kw) -> float:
    for _ in range(WARMUP):
        fn(*args, **kw)
    torch.cuda.synchronize()
    es = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ee = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for s, e in zip(es, ee):
        s.record(); fn(*args, **kw); e.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(es, ee))


def _flops(variant: str, B: int, H: int, N: int, D: int) -> float:
    if variant == "sw_w256":
        effective_n = min(2 * WINDOW, N)
        return 4.0 * B * H * N * effective_n * D
    return 4.0 * B * H * N * N * D


def _hbm_bytes(variant: str, B: int, H: int, N: int, D: int) -> float:
    eb = ELEM_BYTES
    # Q and O: always full
    q_bytes = B * H * N * D * eb
    o_bytes = B * H * N * D * eb
    # K and V: proportional to effective keys seen
    if variant == "sw_w256":
        kv_scale = min(2 * WINDOW, N) / N
    else:
        kv_scale = 1.0
    kv_bytes = 2 * B * H * N * D * eb * kv_scale
    return q_bytes + kv_bytes + o_bytes


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/roofline.csv")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available"); return 1

    variants = [
        ("dense",       _dense,        {}),
        ("causal",      _causal,       {}),
        ("sw_w256",     _sw,           {}),
        ("causal_alibi",_causal_alibi, {}),
    ]
    seqlens = [512, 1024, 2048, 4096]
    B, H, D = BATCH, NUM_HEADS, HEAD_DIM

    # Force JIT compilation at N=512 before timing loop
    g = torch.Generator(device="cuda").manual_seed(0)
    Qw = torch.randn(B, H, 512, D, generator=g, device="cuda", dtype=DTYPE)
    Kw, Vw = torch.randn_like(Qw), torch.randn_like(Qw)
    for _, fn, kw in variants:
        fn(Qw, Kw, Vw, **kw)
    torch.cuda.synchronize()
    del Qw, Kw, Vw

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fout = open(args.output, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow([
        "variant", "seqlen", "latency_ms", "tflops",
        "tc_pct",   # % of fp16 peak TFLOPS (using effective FLOPs)
        "hbm_gbs",  # estimated HBM bandwidth (lower bound, no NCU)
        "hbm_pct",  # % of peak HBM bandwidth
    ])
    print(f"# Roofline runner  device={torch.cuda.get_device_name(0)}")
    print(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# NOTE: hbm_gbs/tc_pct are analytical estimates (no Nsight Compute available)")

    for N in seqlens:
        g2 = torch.Generator(device="cuda").manual_seed(42)
        Q = torch.randn(B, H, N, D, generator=g2, device="cuda", dtype=DTYPE)
        K, V = torch.randn_like(Q), torch.randn_like(Q)

        for vname, fn, kw in variants:
            lat_ms  = _bench_ms(fn, Q, K, V, **kw)
            flops   = _flops(vname, B, H, N, D)
            hbm_b   = _hbm_bytes(vname, B, H, N, D)
            tflops  = flops / (lat_ms * 1e-3) / 1e12
            hbm_gbs = hbm_b / (lat_ms * 1e-3) / 1e9
            tc_pct  = tflops / PEAK_TFLOPS_F16 * 100
            hbm_pct = hbm_gbs / PEAK_HBM_GBS * 100
            row = [vname, N,
                   f"{lat_ms:.4f}", f"{tflops:.2f}",
                   f"{tc_pct:.1f}", f"{hbm_gbs:.1f}", f"{hbm_pct:.1f}"]
            writer.writerow(row); fout.flush()
            print("  ".join(str(x) for x in row))

        del Q, K, V
        torch.cuda.empty_cache()

    fout.close()
    print(f"\n[ok] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
