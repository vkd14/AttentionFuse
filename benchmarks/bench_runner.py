"""Benchmark runner.

Usage:

    python -m benchmarks.bench_runner                       # full sweep
    python -m benchmarks.bench_runner --variant causal      # one variant
    python -m benchmarks.bench_runner --seqlen 2048 --iters 100

Emits a CSV row per (variant, seqlen, baseline) to --output.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import torch

import attnfuse as af
from attnfuse.reference import naive_attention, sdpa_attention, flash_attention
from attnfuse.runtime.dispatch import _alibi_slopes
from .configs import (
    BERT_BASE, GPT2_SMALL, SEQLENS, BATCH_SIZE, WARMUP, ITERS, VARIANTS,
    SLIDING_WINDOW_SIZE,
)


# ---------------------------------------------------------------------------
# AttnFuse variant builders
# ---------------------------------------------------------------------------


def _build_attnfuse(variant: str, num_heads: int):
    if variant == "dense":
        @af.attention
        def fn(Q, K, V):
            return af.softmax(af.scaled_dot_product(Q, K)) @ V
        return fn
    if variant == "causal":
        @af.attention
        def fn(Q, K, V):
            return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V
        return fn
    if variant == "sliding_window":
        @af.attention
        def fn(Q, K, V):
            s = af.scaled_dot_product(Q, K)
            s = af.sliding_window(s, window_size=SLIDING_WINDOW_SIZE)
            return af.softmax(s) @ V
        return fn
    if variant == "causal_alibi":
        @af.attention
        def fn(Q, K, V):
            s = af.scaled_dot_product(Q, K)
            s = af.alibi(s, num_heads=num_heads)
            s = af.causal(s)
            return af.softmax(s) @ V
        return fn
    raise ValueError(f"unknown variant: {variant}")


# ---------------------------------------------------------------------------
# Baseline closures
# ---------------------------------------------------------------------------


def _build_naive(variant: str, num_heads: int) -> Callable:
    causal  = variant in ("causal", "causal_alibi")
    window  = SLIDING_WINDOW_SIZE if variant == "sliding_window" else None
    use_alibi = variant == "causal_alibi"

    def run(Q, K, V):
        slopes = _alibi_slopes(num_heads, str(Q.device), str(Q.dtype)) if use_alibi else None
        return naive_attention(Q, K, V, causal=causal, window=window, alibi_slopes=slopes)
    return run


def _build_sdpa(variant: str, num_heads: int) -> Callable:
    causal  = variant in ("causal", "causal_alibi")
    window  = SLIDING_WINDOW_SIZE if variant == "sliding_window" else None
    use_alibi = variant == "causal_alibi"

    def run(Q, K, V):
        slopes = _alibi_slopes(num_heads, str(Q.device), str(Q.dtype)) if use_alibi else None
        return sdpa_attention(Q, K, V, causal=causal, window=window, alibi_slopes=slopes)
    return run


def _build_flash(variant: str, num_heads: int, dtype: torch.dtype = torch.float16) -> Callable | None:
    """The hand-written reference only handles dense + causal in fp16/bf16."""
    if variant not in ("dense", "causal"):
        return None
    if dtype == torch.float32:
        # Reference Triton flash kernel overflows SMEM for fp32 tile sizes.
        return None
    causal = variant == "causal"
    return lambda Q, K, V: flash_attention(Q, K, V, causal=causal)


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------


@contextmanager
def _peak_memory():
    torch.cuda.reset_peak_memory_stats()
    yield
    # Caller reads torch.cuda.max_memory_allocated()


def _bench(fn: Callable, Q, K, V, warmup: int, iters: int) -> tuple[float, float]:
    """Returns (median latency in ms, peak memory in MB)."""
    # Warmup
    for _ in range(warmup):
        fn(Q, K, V)
    torch.cuda.synchronize()

    # Peak memory measurement
    torch.cuda.reset_peak_memory_stats()
    fn(Q, K, V)
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # Latency measurement (CUDA events are more accurate than wall clock)
    starters = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stoppers = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for s, e in zip(starters, stoppers):
        s.record()
        fn(Q, K, V)
        e.record()
    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(starters, stoppers)]
    return statistics.median(times_ms), peak_mb


def _tflops(latency_ms: float, B: int, H: int, N: int, D: int) -> float:
    """Approximate forward TFLOP/s: 2 matmuls of (N x D)·(D x N) and (N x N)·(N x D)."""
    flops = 4.0 * B * H * N * N * D
    return flops / (latency_ms * 1e-3) / 1e12


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------


def _make_inputs(B, H, N, D, dtype, device):
    g = torch.Generator(device=device).manual_seed(0)
    Q = torch.randn(B, H, N, D, generator=g, device=device, dtype=dtype)
    K = torch.randn(B, H, N, D, generator=g, device=device, dtype=dtype)
    V = torch.randn(B, H, N, D, generator=g, device=device, dtype=dtype)
    return Q, K, V


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=VARIANTS, default=None,
                   help="Run a single variant (default: all).")
    p.add_argument("--model", choices=("bert", "gpt2"), default=None)
    p.add_argument("--seqlen", type=int, default=None)
    p.add_argument("--batch",  type=int, default=BATCH_SIZE)
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--iters",  type=int, default=ITERS)
    p.add_argument("--dtype",  choices=("float16", "bfloat16", "float32"),
                   default="float16")
    p.add_argument("--output", default="results/eval.csv")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; aborting.", file=sys.stderr)
        return 1

    device = "cuda"
    dtype  = getattr(torch, args.dtype)

    variants = [args.variant] if args.variant else list(VARIANTS)
    seqlens  = [args.seqlen]  if args.seqlen  else list(SEQLENS)
    models   = ([BERT_BASE] if args.model == "bert" else
                [GPT2_SMALL] if args.model == "gpt2" else
                [BERT_BASE, GPT2_SMALL])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fout = open(args.output, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["model", "variant", "seqlen", "baseline",
                     "latency_ms", "peak_mem_mb", "tflops"])

    print(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# device={torch.cuda.get_device_name(0)} dtype={args.dtype}")

    for model in models:
        # GPT-2's natural variant is causal, BERT's is dense -- skip useless combos
        for variant in variants:
            if model.is_causal and variant == "dense":
                continue
            if (not model.is_causal) and variant in ("causal", "causal_alibi"):
                continue
            for N in seqlens:
                B, H, D = args.batch, model.num_heads, model.head_dim
                Q, K, V = _make_inputs(B, H, N, D, dtype, device)

                runners = {
                    "naive":      _build_naive(variant, H),
                    "sdpa":       _build_sdpa(variant, H),
                    "flash_ref":  _build_flash(variant, H, dtype),
                    "attnfuse":   _build_attnfuse(variant, H),
                }
                for name, fn in runners.items():
                    if fn is None:
                        continue
                    try:
                        lat, mem = _bench(fn, Q, K, V, args.warmup, args.iters)
                    except (torch.cuda.OutOfMemoryError, Exception) as exc:
                        if not isinstance(exc, torch.cuda.OutOfMemoryError):
                            print(f"    {name} error: {exc}")
                        torch.cuda.empty_cache()
                        lat, mem = float("nan"), float("nan")
                    tfl = _tflops(lat, B, H, N, D) if lat == lat else float("nan")
                    row = [model.name, variant, N, name, f"{lat:.3f}", f"{mem:.1f}", f"{tfl:.2f}"]
                    writer.writerow(row); fout.flush()
                    print("  ".join(str(c) for c in row))
                del Q, K, V
                torch.cuda.empty_cache()

    fout.close()
    print(f"\n[ok] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
