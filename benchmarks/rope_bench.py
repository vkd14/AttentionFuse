"""RoPE: fused kernel vs. pre-processing host-side comparison.

Measures end-to-end latency for causal attention with RoPE positional embeddings
using two approaches:
  1. Pre-processing: apply_rope() in Python host code, then run causal kernel.
  2. Fused: af.rope() combinator rotates Q/K tiles inside the Triton kernel.

The fused path saves two host-side elementwise ops and one HBM round-trip for Q.

Usage::

    python -m benchmarks.rope_bench [--seqlen 2048] [--output results/rope_bench.csv]
"""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import torch

import attnfuse as af
from attnfuse.rope_utils import build_rope_cache, apply_rope


BATCH     = 4
NUM_HEADS = 12
HEAD_DIM  = 64
DTYPE     = torch.float16
WARMUP    = 20
ITERS     = 100


@af.attention
def _causal_preproc(Q, K, V):
    return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V


@af.attention
def _causal_fused(Q, K, V):
    s = af.rope(Q, K)
    s = af.causal(s)
    return af.softmax(s) @ V


def _bench(fn, *args, warmup: int, iters: int):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    events_s = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    events_e = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for s, e in zip(events_s, events_e):
        s.record(); fn(*args); e.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(events_s, events_e))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/rope_bench.csv")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available"); return 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fout = open(args.output, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["seqlen", "method", "latency_ms", "tflops"])

    print(f"# RoPE bench  device={torch.cuda.get_device_name(0)}")
    print(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}")

    for N in (512, 1024, 2048, 4096):
        g = torch.Generator(device="cuda").manual_seed(42)
        Q = torch.randn(BATCH, NUM_HEADS, N, HEAD_DIM, generator=g, device="cuda", dtype=DTYPE)
        K = torch.randn_like(Q)
        V = torch.randn_like(Q)
        cos, sin = build_rope_cache(N, HEAD_DIM, device="cuda", dtype=DTYPE)

        # Force JIT compilation before timing
        Q_rot = apply_rope(Q, cos, sin)
        K_rot = apply_rope(K, cos, sin)
        _causal_preproc(Q_rot, K_rot, V)
        _causal_fused(Q, K, V, cos=cos, sin=sin)
        torch.cuda.synchronize()

        # Pre-processing: rotate then run (includes two apply_rope calls)
        def run_preproc():
            Qr = apply_rope(Q, cos, sin)
            Kr = apply_rope(K, cos, sin)
            return _causal_preproc(Qr, Kr, V)

        def run_fused():
            return _causal_fused(Q, K, V, cos=cos, sin=sin)

        lat_pre  = _bench(run_preproc, warmup=WARMUP, iters=ITERS)
        lat_fused = _bench(run_fused,  warmup=WARMUP, iters=ITERS)

        flops = 4.0 * BATCH * NUM_HEADS * N * N * HEAD_DIM
        tfl_pre   = flops / (lat_pre   * 1e-3) / 1e12
        tfl_fused = flops / (lat_fused * 1e-3) / 1e12

        for method, lat, tfl in (
            ("preprocess", lat_pre,   tfl_pre),
            ("fused",      lat_fused, tfl_fused),
        ):
            row = [N, method, f"{lat:.3f}", f"{tfl:.2f}"]
            writer.writerow(row); fout.flush()
            print("  ".join(str(x) for x in row))

        del Q, K, V, cos, sin
        torch.cuda.empty_cache()

    fout.close()
    print(f"\n[ok] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
