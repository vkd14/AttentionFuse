"""Session 4 -- spike vs production vs flex across N.

The Session 3 result at N=4096 is 1.10x behind flex (0.488 vs 0.443 ms).
The S3 flex ncu also showed HMMA at only 32.6% -- the structural ceiling
for FA-2 at this shape is much lower than the blueprint assumed.

This script asks: does the gap narrow at longer N? Two reasons to think
it might:

  1. The non-matmul fraction of the inner loop (exp, max, mask, online
     softmax) is amortised over more K-tiles per program at larger N.
     At N=4096 with BLOCK_M=128 there are 32 m-blocks; at N=16384 there
     are 128. Each m-block has one diagonal (masked) tile and (N-BLOCK_M)
     full (unmasked) tiles. As N grows, the full-tile fraction grows from
     ~96% (N=4096) to ~99% (N=16384). Non-matmul overhead per output
     row scales as O(1), tensor-core work as O(N).

  2. At larger N the L2 cache effect changes: the Q tile is reused over
     more K tiles, the relative cost of K/V loads (HBM-bound at small N)
     declines, and the kernel becomes more compute-bound.

If we beat flex at any N >= 8192, the paper story changes from "1.10x
behind flex" to "wins at long context, parity at short". That is a much
stronger position for any real LLM workload.

Run:
    python -m benchmarks.hopper_spike_scaling
"""
from __future__ import annotations

import argparse
import statistics
from typing import Callable

import torch

import attnfuse as af
from attnfuse.experimental.hopper_causal_fwd import hopper_causal_fwd

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
def _attnfuse_causal(Q, K, V):
    return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V


def _flex_causal(Q, K, V):
    if not _HAS_FLEX:
        return None
    return _FLEX_COMPILED(Q, K, V, block_mask=_causal_block_mask(Q.shape[-2], Q.device))


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


def _reference(Q, K, V) -> torch.Tensor:
    """fp32 reference. Skipped at large N to save 1+ GB temp memory."""
    B, H, N, D = Q.shape
    sm_scale = D ** -0.5
    S = torch.einsum("bhmd,bhnd->bhmn", Q.float(), K.float()) * sm_scale
    causal = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
    S = S.masked_fill(causal, float("-inf"))
    P = torch.softmax(S, dim=-1)
    return torch.einsum("bhmn,bhnd->bhmd", P, V.float()).to(Q.dtype)


def _causal_flops(B, H, N, D) -> float:
    return 4.0 * B * H * (N * (N + 1) / 2) * D


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch",    type=int, default=4)
    p.add_argument("--heads",    type=int, default=12)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--Ns",       type=int, nargs="+",
                   default=[1024, 2048, 4096, 8192, 16384])
    p.add_argument("--warmup",   type=int, default=10)
    p.add_argument("--iters",    type=int, default=50)
    p.add_argument("--check_until", type=int, default=8192,
                   help="Skip the fp32 numerics check at N > this. "
                        "The fp32 reference materialises an N x N attn matrix "
                        "per (b, h) which is ~1 GB at N=16384, fp4096.")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available"); return 1

    print(f"GPU       : {torch.cuda.get_device_name(0)}")
    print(f"shape     : B={args.batch} H={args.heads} D={args.head_dim} fp16  "
          f"(spike defaults: BM=128 BN=64 nw=8 ns=3)\n")

    print("=" * 92)
    print(f"{'N':>6}  {'spike (ms)':>11}  {'prod (ms)':>10}  {'flex (ms)':>10}  "
          f"{'spike TFLOPS':>13}  {'spk/flex':>9}  {'prd/flex':>9}  {'numerics':>10}")
    print("-" * 92)

    for N in args.Ns:
        g = torch.Generator(device="cuda").manual_seed(0)
        Q = torch.randn(args.batch, args.heads, N, args.head_dim,
                        generator=g, device="cuda", dtype=torch.float16)
        K = torch.randn_like(Q); V = torch.randn_like(Q)

        # Numerics
        if N <= args.check_until:
            try:
                ref = _reference(Q, K, V)
                err_spike = (hopper_causal_fwd(Q, K, V).float() - ref.float()).abs().max().item()
                err_str = f"{err_spike:.2e}"
                del ref
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                err_str = "OOM-ref"
        else:
            err_str = "skipped"

        # Timing
        try:
            t_spike = _time_kernel(hopper_causal_fwd, Q, K, V,
                                    warmup=args.warmup, iters=args.iters)
        except Exception as e:
            t_spike = None
            print(f"  spike crash at N={N}: {str(e)[:60]}")
        try:
            t_prod  = _time_kernel(_attnfuse_causal, Q, K, V,
                                    warmup=args.warmup, iters=args.iters)
        except Exception as e:
            t_prod = None
        try:
            t_flex  = _time_kernel(_flex_causal, Q, K, V,
                                    warmup=args.warmup, iters=args.iters) if _HAS_FLEX else None
        except Exception as e:
            t_flex = None

        flops = _causal_flops(args.batch, args.heads, N, args.head_dim)
        spike_tflops = flops / (t_spike * 1e-3) / 1e12 if t_spike else None
        spk_vs_flex = (t_spike / t_flex) if (t_spike and t_flex) else None
        prd_vs_flex = (t_prod  / t_flex) if (t_prod  and t_flex) else None

        print(f"{N:>6}  "
              f"{t_spike or float('nan'):>9.3f}    "
              f"{t_prod  or float('nan'):>8.3f}    "
              f"{t_flex  or float('nan'):>8.3f}    "
              f"{spike_tflops or 0:>11.1f}    "
              f"{spk_vs_flex or 0:>7.2f}x   "
              f"{prd_vs_flex or 0:>7.2f}x   "
              f"{err_str:>10}")

        del Q, K, V
        torch.cuda.empty_cache()

    print("=" * 92)
    print()
    print("Interpretation guide:")
    print("  spk/flex < 1.0   -> spike beats flex at this N")
    print("  spk/flex ~ 1.0   -> parity")
    print("  spk/flex > 1.1   -> meaningful gap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
