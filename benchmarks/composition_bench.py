"""The 'structural-novelty' benchmark: compositions flex_attention cannot fuse.

This is the paper's headline experiment. flex_attention's ``score_mod``
hook runs on the already-computed ``Q @ K^T`` tile, so any positional
encoding that must be applied to ``Q`` or ``K`` *before* the matmul --
notably RoPE -- cannot be fused. To compare fairly we let flex_attention
do the rotation as a separate pre-processing step (which is exactly the
pre-process baseline AttnFuse already beats by 1.18--2.36x).

We measure four compositions, all in fp16, GPT-2 geometry, batch=4,
12 heads, head_dim=64, on the RTX 3090:

  1. causal                          (both kernels can fuse)
  2. causal + ALiBi                  (both kernels can fuse)
  3. causal + RoPE                   (only AttnFuse fuses)
  4. causal + RoPE + ALiBi           (only AttnFuse fuses)

For variants 3 and 4, the flex baseline calls ``apply_rope`` host-side,
then ``flex_attention`` with the appropriate ``score_mod`` / ``block_mask``.
AttnFuse handles everything in one fused kernel via the seven combinators.

Output: CSV with one row per (variant, seqlen, backend) trio.
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
from attnfuse.rope_utils import build_rope_cache, apply_rope

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 64
    HAS_FLEX = True
except ImportError:
    HAS_FLEX = False

BATCH      = 4
NUM_HEADS  = 12
HEAD_DIM   = 64
DTYPE      = torch.float16
WARMUP     = 10
ITERS      = 50


# --- AttnFuse compositions ------------------------------------------------

@af.attention
def _af_causal(Q, K, V):
    return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V

@af.attention
def _af_causal_alibi(Q, K, V):
    s = af.scaled_dot_product(Q, K)
    s = af.alibi(s, num_heads=NUM_HEADS)
    s = af.causal(s)
    return af.softmax(s) @ V

@af.attention
def _af_causal_rope(Q, K, V):
    s = af.rope(Q, K)
    s = af.causal(s)
    return af.softmax(s) @ V

@af.attention
def _af_causal_rope_alibi(Q, K, V):
    s = af.rope(Q, K)
    s = af.alibi(s, num_heads=NUM_HEADS)
    s = af.causal(s)
    return af.softmax(s) @ V


# --- flex_attention compositions ------------------------------------------
# For RoPE compositions we have NO CHOICE but to pre-process Q and K,
# because score_mod runs after the matmul. This is the structural
# limitation we exploit.

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


def _flex_causal(Q, K, V, N, cos, sin):
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    bm = create_block_mask(causal, B=None, H=None, Q_LEN=N, KV_LEN=N)
    fn = torch.compile(flex_attention, dynamic=False)
    return lambda: fn(Q, K, V, block_mask=bm)

def _flex_causal_alibi(Q, K, V, N, cos, sin):
    slopes = _alibi_slopes(NUM_HEADS)
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    def alibi_mod(score, b, h, q_idx, kv_idx):
        return score + slopes[h] * (kv_idx - q_idx)
    bm = create_block_mask(causal, B=None, H=None, Q_LEN=N, KV_LEN=N)
    fn = torch.compile(flex_attention, dynamic=False)
    return lambda: fn(Q, K, V, score_mod=alibi_mod, block_mask=bm)

def _flex_causal_rope(Q, K, V, N, cos, sin):
    """Best flex_attention can do for RoPE: pre-process Q and K as a
    separate kernel, then call flex_attention. Two extra GPU launches
    and one extra HBM round-trip per Q, K -- this is the gap we exploit."""
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    bm = create_block_mask(causal, B=None, H=None, Q_LEN=N, KV_LEN=N)
    fn = torch.compile(flex_attention, dynamic=False)
    def run():
        Qr = apply_rope(Q, cos, sin)
        Kr = apply_rope(K, cos, sin)
        return fn(Qr, Kr, V, block_mask=bm)
    return run

def _flex_causal_rope_alibi(Q, K, V, N, cos, sin):
    slopes = _alibi_slopes(NUM_HEADS)
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    def alibi_mod(score, b, h, q_idx, kv_idx):
        return score + slopes[h] * (kv_idx - q_idx)
    bm = create_block_mask(causal, B=None, H=None, Q_LEN=N, KV_LEN=N)
    fn = torch.compile(flex_attention, dynamic=False)
    def run():
        Qr = apply_rope(Q, cos, sin)
        Kr = apply_rope(K, cos, sin)
        return fn(Qr, Kr, V, score_mod=alibi_mod, block_mask=bm)
    return run


VARIANTS = [
    # (name,                 attnfuse_fn,            flex_setup_fn,         needs_rope)
    ("causal",               _af_causal,             _flex_causal,           False),
    ("causal_alibi",         _af_causal_alibi,       _flex_causal_alibi,     False),
    ("causal_rope",          _af_causal_rope,        _flex_causal_rope,      True),
    ("causal_rope_alibi",    _af_causal_rope_alibi,  _flex_causal_rope_alibi,True),
]

SEQLENS = [512, 1024, 2048, 4096]


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


def _flops(N: int) -> float:
    """Causal: 2 * 4 * B * H * N^2 * D / 2 = 4 * B * H * N^2 * D / 2."""
    # Causal attention: triangle is N(N+1)/2 ~ N^2/2; multiply by 4 for the
    # standard four-FLOP convention.
    return 2.0 * BATCH * NUM_HEADS * N * N * HEAD_DIM


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/composition_bench.csv")
    args = p.parse_args()

    if not torch.cuda.is_available() or not HAS_FLEX:
        print("Need CUDA + PyTorch >=2.5"); return 1

    print(f"# Composition bench  device={torch.cuda.get_device_name(0)}")
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
        K = torch.randn_like(Q); V = torch.randn_like(Q)
        cos, sin = build_rope_cache(N, HEAD_DIM, device="cuda", dtype=DTYPE)

        for vname, af_fn, flex_setup, needs_rope in VARIANTS:
            # AttnFuse path
            if needs_rope:
                af_call = lambda: af_fn(Q, K, V, cos=cos, sin=sin)
            else:
                af_call = lambda: af_fn(Q, K, V)
            af_ms = _bench_ms(af_call)
            af_tf = _flops(N) / (af_ms * 1e-3) / 1e12

            # flex path
            try:
                flex_call = flex_setup(Q, K, V, N, cos, sin)
                flex_ms = _bench_ms(flex_call)
                flex_tf = _flops(N) / (flex_ms * 1e-3) / 1e12
                speedup = flex_ms / af_ms
                err = ""
            except Exception as e:
                flex_ms = float("nan"); flex_tf = float("nan"); speedup = float("nan")
                err = str(e).splitlines()[0][:80]

            for backend, ms, tf in [("attnfuse", af_ms, af_tf),
                                     ("flex",     flex_ms, flex_tf)]:
                sp = 1.0 if backend == "attnfuse" else speedup
                row = [vname, N, backend,
                       f"{ms:.3f}" if not math.isnan(ms) else "nan",
                       f"{tf:.2f}" if not math.isnan(tf) else "nan",
                       f"{sp:.2f}" if not math.isnan(sp) else "nan"]
                writer.writerow(row); fout.flush()
                print("  ".join(str(x) for x in row),
                      err if backend == "flex" and err else "")

        del Q, K, V, cos, sin; torch.cuda.empty_cache()

    fout.close()
    print(f"\n[ok] wrote {args.output}")
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    raise SystemExit(main())
