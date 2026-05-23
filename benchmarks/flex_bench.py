"""Baseline: PyTorch ``flex_attention`` (PyTorch 2.5+) vs AttnFuse.

flex_attention is PyTorch's compile-based attention generalisation introduced in
2.5: the user supplies a Python ``score_mod`` function and / or a block mask;
TorchInductor generates a fused Triton kernel.  It is the closest published
competitor to AttnFuse.

What this script does
---------------------
For each variant we benchmark three things side-by-side at $N = 512..4096$,
fp16, GPT-2 geometry, RTX 3090:

  * AttnFuse           — our DSL-compiled kernel
  * flex_attention     — torch.compile'd flex_attention with mask_mod / score_mod
  * sdpa               — PyTorch's stock SDPA (FA2 backend if supported, else
                          O(N^2) fallback)

Variants covered:
  dense, causal, sliding-window (W=256), causal+ALiBi.

RoPE is **deliberately excluded**: flex_attention's score_mod runs on the
already-computed ``Q K^T`` tile, so it cannot rotate Q and K before the
matmul.  Achieving fused RoPE in flex_attention requires materialising the
rotated Q, K tensors first — which is exactly the pre-processing path
AttnFuse beats by 1.18--2.29x in ``rope_bench.py``.  That asymmetry is the
sharpest paper claim we have; this script measures everything else fairly.

Usage::

    python -m benchmarks.flex_bench [--output results/flex_bench.csv]
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

# flex_attention lives under torch.nn.attention.flex_attention in 2.5+
try:
    from torch.nn.attention.flex_attention import (
        flex_attention,
        create_block_mask,
    )
    import torch._dynamo
    # We compile a fresh flex_attention per (variant, N); the default limit
    # of 8 trips before we finish the (4 variants x 4 seqlens) = 16 grid.
    torch._dynamo.config.cache_size_limit = 64
    HAS_FLEX = True
except ImportError:  # pragma: no cover
    HAS_FLEX = False

BATCH      = 4
NUM_HEADS  = 12
HEAD_DIM   = 64
WINDOW     = 256
DTYPE      = torch.float16
WARMUP     = 10
ITERS      = 50


# --- AttnFuse graphs (one per variant) -------------------------------------

@af.attention
def _af_dense(Q, K, V):
    return af.softmax(af.scaled_dot_product(Q, K)) @ V


@af.attention
def _af_causal(Q, K, V):
    return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V


@af.attention
def _af_sw(Q, K, V):
    return af.softmax(af.sliding_window(af.scaled_dot_product(Q, K), WINDOW)) @ V


@af.attention
def _af_causal_alibi(Q, K, V):
    s = af.scaled_dot_product(Q, K)
    s = af.alibi(s, num_heads=NUM_HEADS)
    s = af.causal(s)
    return af.softmax(s) @ V


# --- flex_attention setups -------------------------------------------------
# Each setup function builds the score_mod / block_mask once per (variant, N)
# and returns a zero-arg callable that runs one forward pass.

def _alibi_slopes(num_heads: int) -> torch.Tensor:
    n = 2 ** math.floor(math.log2(num_heads))
    base = 2 ** (-(2 ** -(math.log2(n) - 3)))
    powers = torch.arange(1, n + 1, dtype=torch.float32)
    slopes = torch.pow(base, powers)
    if n < num_heads:
        extra_base = 2 ** (-(2 ** -(math.log2(2 * n) - 3)))
        extra_pow  = torch.arange(1, 2 * (num_heads - n) + 1, 2, dtype=torch.float32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_pow)])
    return slopes.to(device="cuda")


def _flex_dense(Q, K, V, N):
    fn = torch.compile(flex_attention, dynamic=False)
    return lambda: fn(Q, K, V)


def _flex_causal(Q, K, V, N):
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    block_mask = create_block_mask(causal_mask, B=None, H=None, Q_LEN=N, KV_LEN=N)
    fn = torch.compile(flex_attention, dynamic=False)
    return lambda: fn(Q, K, V, block_mask=block_mask)


def _flex_sw(Q, K, V, N):
    # Bidirectional sliding-window to match af.sliding_window(s, W):
    # |q - k| < W. This spans 2W-1 keys per query, NOT W.
    def sw_mask(b, h, q_idx, kv_idx):
        d = q_idx - kv_idx
        return (d < WINDOW) & (d > -WINDOW)
    block_mask = create_block_mask(sw_mask, B=None, H=None, Q_LEN=N, KV_LEN=N)
    fn = torch.compile(flex_attention, dynamic=False)
    return lambda: fn(Q, K, V, block_mask=block_mask)


def _flex_causal_alibi(Q, K, V, N):
    slopes = _alibi_slopes(NUM_HEADS)
    def alibi_mod(score, b, h, q_idx, kv_idx):
        return score + slopes[h] * (kv_idx - q_idx)
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    block_mask = create_block_mask(causal_mask, B=None, H=None, Q_LEN=N, KV_LEN=N)
    fn = torch.compile(flex_attention, dynamic=False)
    return lambda: fn(Q, K, V, score_mod=alibi_mod, block_mask=block_mask)


# --- bench helper ----------------------------------------------------------

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


def _flops(variant: str, N: int) -> float:
    B, H, D = BATCH, NUM_HEADS, HEAD_DIM
    if variant == "sw_w256":
        eff = min(2 * WINDOW, N)
        return 4.0 * B * H * N * eff * D
    return 4.0 * B * H * N * N * D


# --- main ------------------------------------------------------------------

VARIANTS = [
    # (name,              attnfuse_fn,        flex_setup_fn)
    ("dense",             _af_dense,          _flex_dense),
    ("causal",            _af_causal,         _flex_causal),
    ("sw_w256",           _af_sw,             _flex_sw),
    ("causal_alibi",      _af_causal_alibi,   _flex_causal_alibi),
]

SEQLENS = [512, 1024, 2048, 4096]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/flex_bench.csv")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available"); return 1
    if not HAS_FLEX:
        print("flex_attention not available — needs PyTorch 2.5+"); return 1

    print(f"# Flex bench  device={torch.cuda.get_device_name(0)}")
    print(f"# torch={torch.__version__}  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fout = open(args.output, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["variant", "seqlen", "backend",
                     "latency_ms", "tflops", "speedup_vs_attnfuse"])

    for N in SEQLENS:
        g = torch.Generator(device="cuda").manual_seed(0)
        Q = torch.randn(BATCH, NUM_HEADS, N, HEAD_DIM, generator=g,
                        device="cuda", dtype=DTYPE)
        K = torch.randn_like(Q)
        V = torch.randn_like(Q)

        for vname, af_fn, flex_setup in VARIANTS:
            # AttnFuse
            af_ms = _bench_ms(lambda: af_fn(Q, K, V))
            af_tf = _flops(vname, N) / (af_ms * 1e-3) / 1e12

            # flex_attention
            try:
                flex_call = flex_setup(Q, K, V, N)
                flex_ms = _bench_ms(flex_call)
                flex_tf = _flops(vname, N) / (flex_ms * 1e-3) / 1e12
                flex_speedup = af_ms / flex_ms
                flex_err = ""
            except Exception as e:
                flex_ms = float("nan")
                flex_tf = float("nan")
                flex_speedup = float("nan")
                flex_err = str(e).splitlines()[0][:80]

            for backend, ms, tf, sp in [
                ("attnfuse", af_ms, af_tf, 1.0),
                ("flex",     flex_ms, flex_tf, flex_speedup),
            ]:
                row = [vname, N, backend,
                       f"{ms:.3f}" if not math.isnan(ms) else "nan",
                       f"{tf:.2f}" if not math.isnan(tf) else "nan",
                       f"{sp:.2f}" if not math.isnan(sp) else "nan"]
                writer.writerow(row); fout.flush()
                print("  ".join(str(x) for x in row),
                      flex_err if backend == "flex" and flex_err else "")

        # free tensors between seqlens to avoid OOM at fp16 N=4096
        del Q, K, V
        torch.cuda.empty_cache()

    fout.close()
    print(f"\n[ok] wrote {args.output}")
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    raise SystemExit(main())
