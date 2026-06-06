"""Session 6 -- RoPE+causal vs flex on H100. The headline LLM-production case.

flex_attention cannot fuse RoPE: its score_mod hook fires AFTER the
QK^T matmul, so any positional encoding that must be applied to Q and
K BEFORE the matmul (which RoPE is) requires the user to materialise
rotated Q' and K' tensors before calling flex_attention. That is two
extra HBM round-trips plus two extra kernel launches per attention
layer.

AttnFuse rotates Q once outside the inner loop and rotates each K tile
inside the inner loop (in registers). Zero extra HBM traffic.

On RTX 3090 this fusion gave 2.10x speedup. This script measures the
same composition on H100 NVL via the dispatch path (i.e. through
@af.attention so the spike is the active kernel).

Run on H100:
    python -m benchmarks.hopper_rope_bench
"""
from __future__ import annotations

import argparse
import statistics
from typing import Callable

import torch

import attnfuse as af
from attnfuse.rope_utils import build_rope_cache, apply_rope

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    _HAS_FLEX = True
    _FLEX_COMPILED = torch.compile(flex_attention, dynamic=False, fullgraph=True)
except ImportError:
    _HAS_FLEX = False
    _FLEX_COMPILED = None


_FLEX_MASK_CACHE: dict = {}


def _causal_block_mask(N: int, device):
    key = (N, str(device))
    bm = _FLEX_MASK_CACHE.get(key)
    if bm is None:
        def causal(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx
        bm = create_block_mask(causal, B=None, H=None,
                                Q_LEN=N, KV_LEN=N, device=device)
        _FLEX_MASK_CACHE[key] = bm
    return bm


@af.attention
def _attnfuse_rope_causal(Q, K, V):
    s = af.rope(Q, K)
    s = af.causal(s)
    return af.softmax(s) @ V


def _flex_rope_causal(Q, K, V, cos, sin):
    """The flex equivalent: rotate outside, then causal-flex.

    This is exactly the workaround HuggingFace + PyTorch use today --
    a separate elementwise kernel materialises Q' and K', then flex
    runs on the rotated tensors. Two extra HBM round-trips and two
    extra kernel launches that AttnFuse avoids.
    """
    if not _HAS_FLEX:
        return None
    Q_rot = apply_rope(Q, cos, sin)
    K_rot = apply_rope(K, cos, sin)
    return _FLEX_COMPILED(Q_rot, K_rot, V, block_mask=_causal_block_mask(Q.shape[-2], Q.device))


def _time_kernel(fn: Callable, *args, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    events = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(*args); e.record()
        events.append((s, e))
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in events)


def _causal_flops(B, H, N, D) -> float:
    return 4.0 * B * H * (N * (N + 1) / 2) * D


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch",    type=int, default=4)
    p.add_argument("--heads",    type=int, default=12)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--Ns",       type=int, nargs="+",
                   default=[2048, 4096, 8192, 16384])
    p.add_argument("--warmup",   type=int, default=10)
    p.add_argument("--iters",    type=int, default=50)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available"); return 1

    print(f"GPU       : {torch.cuda.get_device_name(0)}")
    print(f"shape     : B={args.batch} H={args.heads} D={args.head_dim} fp16  RoPE+causal\n")

    print("=" * 86)
    print(f"{'N':>6}  {'attnfuse (ms)':>14}  {'flex+rotate (ms)':>17}  "
          f"{'AF TFLOPS':>10}  {'AF speedup':>11}  {'numerics':>10}")
    print("-" * 86)

    for N in args.Ns:
        g = torch.Generator(device="cuda").manual_seed(0)
        Q = torch.randn(args.batch, args.heads, N, args.head_dim,
                        generator=g, device="cuda", dtype=torch.float16)
        K = torch.randn_like(Q); V = torch.randn_like(Q)
        cos, sin = build_rope_cache(N, args.head_dim, device="cuda", dtype=torch.float16)

        # Numerics check at small N only (the N x N reference is expensive).
        if N <= 4096:
            out_af = _attnfuse_rope_causal(Q, K, V, cos=cos, sin=sin)
            # Reference path: rotate, then causal softmax(QK^T)V.
            Qr = apply_rope(Q, cos, sin)
            Kr = apply_rope(K, cos, sin)
            S = torch.einsum('bhmd,bhnd->bhmn', Qr.float(), Kr.float()) * (args.head_dim**-0.5)
            mask = torch.triu(torch.ones(N, N, device="cuda", dtype=torch.bool), diagonal=1)
            S = S.masked_fill(mask, float("-inf"))
            P = torch.softmax(S, dim=-1)
            ref = torch.einsum('bhmn,bhnd->bhmd', P, V.float()).to(torch.float16)
            err_str = f"{(out_af.float() - ref.float()).abs().max().item():.2e}"
            del S, P, ref, Qr, Kr; torch.cuda.empty_cache()
        else:
            err_str = "skipped"

        try:
            t_af  = _time_kernel(lambda q, k, v: _attnfuse_rope_causal(q, k, v, cos=cos, sin=sin),
                                  Q, K, V, warmup=args.warmup, iters=args.iters)
        except Exception as e:
            t_af = None
            print(f"  attnfuse crash at N={N}: {str(e)[:60]}")

        try:
            t_flex = _time_kernel(lambda q, k, v: _flex_rope_causal(q, k, v, cos, sin),
                                   Q, K, V, warmup=args.warmup, iters=args.iters) if _HAS_FLEX else None
        except Exception as e:
            t_flex = None

        flops    = _causal_flops(args.batch, args.heads, N, args.head_dim)
        af_tflps = flops / (t_af * 1e-3) / 1e12 if t_af else None
        speedup  = (t_flex / t_af) if (t_af and t_flex) else None

        print(f"{N:>6}  "
              f"{t_af or float('nan'):>12.3f}    "
              f"{t_flex or float('nan'):>15.3f}    "
              f"{af_tflps or 0:>8.1f}    "
              f"{speedup or 0:>9.2f}x   "
              f"{err_str:>10}")

        del Q, K, V, cos, sin
        torch.cuda.empty_cache()

    print("=" * 86)
    print()
    print("AF speedup > 1.0 means AttnFuse is faster than flex+pre-rotate at this N.")
    print("Reference (RTX 3090, N=4096, B=2 H=12 D=64): AttnFuse 2.10x faster than flex+pre-rotate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
