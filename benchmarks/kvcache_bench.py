"""KV-cache decoding benchmark — the production LLM inference path.

During autoregressive generation, each generation step has:
  Q.shape  = (B, H_q,    1,         D)    # one new token
  K.shape  = (B, H_kv,   cache_len, D)    # all past tokens
  V.shape  = (B, H_kv,   cache_len, D)

This is the most-called attention pattern in deployed LLMs. Per-call
latency directly affects time-to-first-token (TTFT) and time-between-
tokens (TBT) -- the user-visible latency.

We measure AttnFuse and flex_attention across realistic geometries:
  Llama-3-8B-MQA-equivalent  (H_q=32, H_kv=8,  D=128, group=4)
  Llama-3-70B                (H_q=64, H_kv=8,  D=128, group=8)
  Falcon-40B-MQA             (H_q=64, H_kv=1,  D=128, group=64)
With cache lengths 1024, 4096, 16384, 32768 (long-context decoding).
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
import warnings
from pathlib import Path

import torch
import attnfuse as af

try:
    from torch.nn.attention.flex_attention import flex_attention
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 128
    HAS_FLEX = True
except ImportError:
    HAS_FLEX = False

DTYPE  = torch.float16
WARMUP = 12
ITERS  = 60

GEOMETRIES = [
    ("llama3-8B",   32,  8, 128),
    ("llama3-70B",  64,  8, 128),
    ("falcon-MQA",  64,  1, 128),
]
CACHE_LENS = [1024, 4096, 16384, 32768]


@af.attention
def af_dense(Q, K, V):
    """Dense attention: every Q sees every K. The KV-cache decoder pattern."""
    return af.softmax(af.scaled_dot_product(Q, K)) @ V


def _flex_dense_gqa(Q, K, V):
    fn = torch.compile(flex_attention, dynamic=False)
    return lambda: fn(Q, K, V, enable_gqa=True)


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


def _flops_per_step(B: int, H_q: int, N_kv: int, D: int) -> float:
    """Decode step: B * H_q * 1 query attends to N_kv keys with D dim.
    Two matmuls (Q@K^T and P@V) = 4 * B * H_q * N_kv * D FLOPs total."""
    return 4.0 * B * H_q * 1 * N_kv * D


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--output", default="results/kvcache_bench.csv")
    args = p.parse_args()
    if not torch.cuda.is_available() or not HAS_FLEX:
        print("Need CUDA + PyTorch >=2.5"); return 1

    print(f"# KV-cache bench  device={torch.cuda.get_device_name(0)}")
    print(f"# B={args.batch} dtype={DTYPE}  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fout = open(args.output, "w", newline="")
    w = csv.writer(fout)
    w.writerow(["model", "H_q", "H_kv", "cache_len", "backend",
                "latency_us", "tflops", "speedup_vs_flex"])
    print(f"{'model':12s} {'cache':>6s} {'backend':10s} {'µs':>10s} {'TFLOPS':>8s}  speedup")

    B = args.batch
    for name, H_q, H_kv, D in GEOMETRIES:
        for N_kv in CACHE_LENS:
            try:
                g = torch.Generator(device="cuda").manual_seed(0)
                Q = torch.randn(B, H_q,     1, D, generator=g, device="cuda", dtype=DTYPE)
                K = torch.randn(B, H_kv, N_kv, D, generator=g, device="cuda", dtype=DTYPE)
                V = torch.randn(B, H_kv, N_kv, D, generator=g, device="cuda", dtype=DTYPE)
            except torch.cuda.OutOfMemoryError:
                print(f"  [skip {name} cache={N_kv}] OOM")
                continue

            try:
                af_ms = _bench_ms(lambda: af_dense(Q, K, V))
                af_tf = _flops_per_step(B, H_q, N_kv, D) / (af_ms * 1e-3) / 1e12
                af_us = af_ms * 1000.0
                af_ok = True
            except Exception as e:
                af_ms = float("nan"); af_us = float("nan"); af_tf = float("nan")
                af_ok = False; print(f"  AttnFuse failed: {str(e)[:80]}")

            try:
                flex_call = _flex_dense_gqa(Q, K, V)
                flex_ms = _bench_ms(flex_call)
                flex_us = flex_ms * 1000.0
                flex_tf = _flops_per_step(B, H_q, N_kv, D) / (flex_ms * 1e-3) / 1e12
                flex_ok = True
            except Exception as e:
                flex_ms = float("nan"); flex_us = float("nan"); flex_tf = float("nan")
                flex_ok = False; print(f"  flex failed: {str(e)[:80]}")

            speedup = (flex_ms / af_ms) if (af_ok and flex_ok) else float("nan")

            for backend, us, tf in [("attnfuse", af_us, af_tf),
                                     ("flex",     flex_us, flex_tf)]:
                sp = 1.0 if backend == "attnfuse" else speedup
                w.writerow([name, H_q, H_kv, N_kv, backend,
                            f"{us:.1f}" if not math.isnan(us) else "nan",
                            f"{tf:.3f}" if not math.isnan(tf) else "nan",
                            f"{sp:.2f}" if not math.isnan(sp) else "nan"])
                fout.flush()
                print(f"{name:12s} {N_kv:>6d} {backend:10s} {us:>9.1f}  {tf:>7.3f}  "
                      f"{'(ref)' if backend == 'attnfuse' else f'{sp:.2f}x'}")

            del Q, K, V; torch.cuda.empty_cache()

    fout.close()
    print(f"\n[ok] wrote {args.output}")
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    raise SystemExit(main())
