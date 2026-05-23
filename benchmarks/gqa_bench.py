"""Grouped-Query Attention (GQA) benchmark — Llama-3-style geometries.

Llama 3 uses GQA with the following ratios:
  Llama-3-8B:   H_q = 32, H_kv = 8   (group_size = 4)
  Llama-3-70B:  H_q = 64, H_kv = 8   (group_size = 8)

Mistral-7B uses 8 KV heads vs 32 query heads — same as Llama-3-8B.
Falcon-40B uses Multi-Query Attention: H_q = 64, H_kv = 1.

This benchmark measures AttnFuse and flex_attention on three LLM-realistic
shapes (causal-only, since GQA is overwhelmingly used in decoder-only
autoregressive models):

  Llama-3-8B-like     : H_q=32, H_kv= 8, D=128   (group=4)
  Llama-3-70B-like    : H_q=64, H_kv= 8, D=128   (group=8)
  Falcon-MQA          : H_q=64, H_kv= 1, D=128   (group=64)

For flex_attention, GQA is supported via the ``enable_gqa=True`` flag in
PyTorch 2.5+ (the kernel auto-broadcasts K and V over the group axis).

Note: this is a smaller geometry sweep than flex_bench because GQA shapes
are constrained by real-model dimensions.
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
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 64
    HAS_FLEX = True
except ImportError:
    HAS_FLEX = False

DTYPE  = torch.float16
WARMUP = 10
ITERS  = 50

# (name, H_q, H_kv, head_dim)
GEOMETRIES = [
    ("llama3-8B",   32,  8, 128),
    ("llama3-70B",  64,  8, 128),
    ("falcon-MQA",  64,  1, 128),
]
SEQLENS = [512, 1024, 2048, 4096]


@af.attention
def af_causal(Q, K, V):
    return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V


def _flex_causal_gqa(Q, K, V, N):
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    bm = create_block_mask(causal, B=None, H=None, Q_LEN=N, KV_LEN=N)
    fn = torch.compile(flex_attention, dynamic=False)
    # PyTorch 2.5+: enable_gqa=True broadcasts K/V across query head groups
    return lambda: fn(Q, K, V, block_mask=bm, enable_gqa=True)


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


def _causal_flops(B: int, H_q: int, N: int, D: int) -> float:
    """Causal attention: triangle is ~N^2/2; 4 FLOPs per (q, k) pair."""
    return 2.0 * B * H_q * N * N * D


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--output", default="results/gqa_bench.csv")
    args = p.parse_args()

    if not torch.cuda.is_available() or not HAS_FLEX:
        print("Need CUDA + PyTorch >=2.5"); return 1

    print(f"# GQA bench  device={torch.cuda.get_device_name(0)}")
    print(f"# torch={torch.__version__}  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fout = open(args.output, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["model", "H_q", "H_kv", "group_size", "seqlen",
                     "backend", "latency_ms", "tflops", "speedup_vs_flex"])
    print(f"{'model':14s} {'shape':24s} {'N':>5s} {'backend':10s} "
          f"{'ms':>10s} {'TFLOPS':>9s}")

    B = args.batch
    for name, H_q, H_kv, D in GEOMETRIES:
        group = H_q // H_kv
        for N in SEQLENS:
            g = torch.Generator(device="cuda").manual_seed(0)
            Q = torch.randn(B, H_q,  N, D, generator=g, device="cuda", dtype=DTYPE)
            K = torch.randn(B, H_kv, N, D, generator=g, device="cuda", dtype=DTYPE)
            V = torch.randn(B, H_kv, N, D, generator=g, device="cuda", dtype=DTYPE)

            # AttnFuse
            try:
                af_ms = _bench_ms(lambda: af_causal(Q, K, V))
                af_tf = _causal_flops(B, H_q, N, D) / (af_ms * 1e-3) / 1e12
                af_ok = True
            except Exception as e:
                af_ms = float("nan"); af_tf = float("nan"); af_ok = False
                print(f"  AttnFuse failed for {name} N={N}: {str(e)[:80]}")

            # flex_attention with enable_gqa
            try:
                flex_call = _flex_causal_gqa(Q, K, V, N)
                flex_ms = _bench_ms(flex_call)
                flex_tf = _causal_flops(B, H_q, N, D) / (flex_ms * 1e-3) / 1e12
                flex_ok = True
            except Exception as e:
                flex_ms = float("nan"); flex_tf = float("nan"); flex_ok = False
                print(f"  flex failed for {name} N={N}: {str(e)[:80]}")

            speedup = (flex_ms / af_ms) if (af_ok and flex_ok) else float("nan")

            for backend, ms, tf in [("attnfuse", af_ms, af_tf),
                                     ("flex",     flex_ms, flex_tf)]:
                sp = 1.0 if backend == "attnfuse" else speedup
                row = [name, H_q, H_kv, group, N, backend,
                       f"{ms:.3f}" if not math.isnan(ms) else "nan",
                       f"{tf:.2f}" if not math.isnan(tf) else "nan",
                       f"{sp:.2f}" if not math.isnan(sp) else "nan"]
                writer.writerow(row); fout.flush()
                shape_str = f"H_q={H_q} H_kv={H_kv} D={D}"
                print(f"{name:14s} {shape_str:24s} {N:>5d} {backend:10s} "
                      f"{ms:>9.3f}  {tf:>8.2f}")

            del Q, K, V
            torch.cuda.empty_cache()

    fout.close()
    print(f"\n[ok] wrote {args.output}")
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    raise SystemExit(main())
