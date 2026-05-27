"""Backward-pass performance benchmark.

Measures AttnFuse vs PyTorch SDPA on forward+backward and isolates the
backward-alone latency. Reports per-seqlen latency and the AttnFuse/SDPA
ratio for both directions.

Configurations: causal attention, fp16, GPT-2 geometry (B=4, H=12, D=64).
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import attnfuse as af

BATCH      = 4
NUM_HEADS  = 12
HEAD_DIM   = 64
DTYPE      = torch.float16
WARMUP     = 8
ITERS      = 40
SEQLENS    = [512, 1024, 2048, 4096]


@af.attention
def causal_attn(Q, K, V):
    return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V


def _make(N: int, requires_grad: bool):
    g = torch.Generator(device="cuda").manual_seed(0)
    Q = torch.randn(BATCH, NUM_HEADS, N, HEAD_DIM,
                    generator=g, device="cuda", dtype=DTYPE,
                    requires_grad=requires_grad)
    K = torch.randn_like(Q, requires_grad=requires_grad)
    V = torch.randn_like(Q, requires_grad=requires_grad)
    return Q, K, V


def _bench_ms(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    es = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ee = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for s, e in zip(es, ee):
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(es, ee))


def _af_fwd(Q, K, V):       return causal_attn(Q, K, V)
def _sdpa_fwd(Q, K, V):     return F.scaled_dot_product_attention(Q, K, V, is_causal=True)


def _make_bwd_call(fwd_fn, Q, K, V):
    def call():
        Q.grad = K.grad = V.grad = None
        O = fwd_fn(Q, K, V)
        O.sum().backward()
    return call


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/backward_bench.csv")
    args = p.parse_args()
    if not torch.cuda.is_available():
        print("CUDA not available"); return 1
    print(f"# Backward bench  device={torch.cuda.get_device_name(0)}")
    print(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fout = open(args.output, "w", newline="")
    w = csv.writer(fout)
    w.writerow(["seqlen", "stage", "backend", "latency_ms", "ratio_vs_sdpa"])

    print(f"{'N':>5s}  {'stage':>12s}  {'AttnFuse':>10s}  {'SDPA':>10s}  {'ratio':>6s}")
    for N in SEQLENS:
        # Forward-only timing (no autograd graph captured)
        with torch.no_grad():
            Q_inf, K_inf, V_inf = _make(N, requires_grad=False)
            af_fwd_ms   = _bench_ms(lambda: _af_fwd(Q_inf, K_inf, V_inf))
            sdpa_fwd_ms = _bench_ms(lambda: _sdpa_fwd(Q_inf, K_inf, V_inf))

        # Forward+backward
        Q, K, V = _make(N, requires_grad=True)
        af_full_ms   = _bench_ms(_make_bwd_call(_af_fwd, Q, K, V))
        Q2, K2, V2 = _make(N, requires_grad=True)
        sdpa_full_ms = _bench_ms(_make_bwd_call(_sdpa_fwd, Q2, K2, V2))

        af_bwd_ms   = af_full_ms - af_fwd_ms
        sdpa_bwd_ms = sdpa_full_ms - sdpa_fwd_ms

        for stage, af_ms, sdpa_ms in [("fwd",   af_fwd_ms,  sdpa_fwd_ms),
                                       ("bwd",   af_bwd_ms,  sdpa_bwd_ms),
                                       ("fwd+bwd", af_full_ms, sdpa_full_ms)]:
            ratio = af_ms / sdpa_ms if sdpa_ms > 0 else float("nan")
            w.writerow([N, stage, "attnfuse", f"{af_ms:.3f}", f"{ratio:.2f}"])
            w.writerow([N, stage, "sdpa",     f"{sdpa_ms:.3f}", "1.00"])
            fout.flush()
            print(f"{N:>5d}  {stage:>12s}  {af_ms:>10.3f}  {sdpa_ms:>10.3f}  {ratio:>5.2f}x")

        del Q, K, V, Q2, K2, V2, Q_inf, K_inf, V_inf
        torch.cuda.empty_cache()

    fout.close()
    print(f"\n[ok] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
