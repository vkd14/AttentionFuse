"""Measure JIT first-call compile latency per attention variant.

AttnFuse compiles a Triton kernel on the first call per unique graph signature.
This script measures that one-time cost against subsequent cached-call latency.

Usage::

    python -m benchmarks.jit_compile_bench [--output results/jit_compile.csv]
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

import attnfuse as af
from attnfuse.runtime.kernel_cache import clear_cache


BATCH     = 2
NUM_HEADS = 12
HEAD_DIM  = 64
N         = 1024
DTYPE     = torch.float16
CACHED_ITERS = 50


def _make_inputs():
    g = torch.Generator(device="cuda").manual_seed(0)
    Q = torch.randn(BATCH, NUM_HEADS, N, HEAD_DIM, generator=g, device="cuda", dtype=DTYPE)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    return Q, K, V


def _time_first_and_cached(fn, *args, **kwargs):
    """Returns (first_call_ms, median_cached_ms)."""
    clear_cache()  # ensure fresh compile

    t0 = time.perf_counter()
    fn(*args, **kwargs)
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - t0) * 1e3

    # Cached calls
    events_s = [torch.cuda.Event(enable_timing=True) for _ in range(CACHED_ITERS)]
    events_e = [torch.cuda.Event(enable_timing=True) for _ in range(CACHED_ITERS)]
    for s, e in zip(events_s, events_e):
        s.record(); fn(*args, **kwargs); e.record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(events_s, events_e)]
    import statistics
    cached_ms = statistics.median(times)
    return first_ms, cached_ms


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/jit_compile.csv")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available"); return 1

    from attnfuse.rope_utils import build_rope_cache
    cos, sin = build_rope_cache(N, HEAD_DIM, device="cuda", dtype=DTYPE)

    @af.attention
    def dense(Q, K, V):
        return af.softmax(af.scaled_dot_product(Q, K)) @ V

    @af.attention
    def causal(Q, K, V):
        return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V

    @af.attention
    def sliding(Q, K, V):
        return af.softmax(af.sliding_window(af.scaled_dot_product(Q, K), 256)) @ V

    @af.attention
    def causal_alibi(Q, K, V):
        s = af.scaled_dot_product(Q, K)
        s = af.alibi(s, num_heads=NUM_HEADS)
        s = af.causal(s)
        return af.softmax(s) @ V

    @af.attention
    def causal_rope(Q, K, V):
        return af.softmax(af.causal(af.rope(Q, K))) @ V

    variants = [
        ("dense",        dense,        {}),
        ("causal",       causal,       {}),
        ("sliding_w256", sliding,      {}),
        ("causal_alibi", causal_alibi, {}),
        ("causal_rope",  causal_rope,  {"cos": cos, "sin": sin}),
    ]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fout = open(args.output, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["variant", "first_call_ms", "cached_call_ms", "compile_overhead_x"])
    print(f"# JIT compile-time  N={N}  device={torch.cuda.get_device_name(0)}")

    Q, K, V = _make_inputs()
    for name, fn, kw in variants:
        first_ms, cached_ms = _time_first_and_cached(fn, Q, K, V, **kw)
        overhead = first_ms / cached_ms if cached_ms > 0 else float("inf")
        row = [name, f"{first_ms:.1f}", f"{cached_ms:.3f}", f"{overhead:.0f}x"]
        writer.writerow(row); fout.flush()
        print(f"  {name:<16} first={first_ms:7.1f} ms  cached={cached_ms:.3f} ms  overhead={overhead:.0f}x")

    fout.close()
    print(f"\n[ok] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
