"""Hopper-spike driver: numerics + wall-clock vs the production causal kernel.

Phase 1 spike, Session 1. Measures whether the WGMMA-friendly tile pipeline
(BLOCK_M=128, BLOCK_N=128, num_warps=8, num_stages=3, FA-2 causal split)
closes the H100 gap to flex_attention on causal forward at D=64, fp16.

Run:

    python -m benchmarks.hopper_spike --seqlen 4096

Outputs a table of (kernel, latency_ms, tflops, max_abs_err, pct_of_flex).

To re-profile with ncu after a positive result:

    bash scripts/run_hopper_spike_ncu.sh    # writes results/ncu_hopper_*.csv
"""
from __future__ import annotations

import argparse
import math
import statistics
from typing import Callable

import torch

import attnfuse as af
from attnfuse.experimental.hopper_causal_fwd import hopper_causal_fwd

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    _HAS_FLEX = True
except ImportError:
    _HAS_FLEX = False


@af.attention
def _attnfuse_causal(Q, K, V):
    return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V


def _reference_attention(Q, K, V) -> torch.Tensor:
    """fp32 reference for numerics check."""
    B, H, N, D = Q.shape
    sm_scale = D ** -0.5
    S = torch.einsum("bhmd,bhnd->bhmn", Q.float(), K.float()) * sm_scale
    causal = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
    S = S.masked_fill(causal, float("-inf"))
    P = torch.softmax(S, dim=-1)
    return torch.einsum("bhmn,bhnd->bhmd", P, V.float()).to(Q.dtype)


def _flex_causal(Q, K, V) -> torch.Tensor:
    if not _HAS_FLEX:
        return None
    B, H, N, D = Q.shape

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    bm = create_block_mask(causal_mask, B=None, H=None, Q_LEN=N, KV_LEN=N, device=Q.device)
    return flex_attention(Q, K, V, block_mask=bm)


def _time_kernel(fn: Callable, *args, warmup: int = 10, iters: int = 50) -> float:
    """Return median latency in milliseconds via cuda events."""
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    events = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        events.append((start, end))
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in events]
    return statistics.median(times)


def _causal_flops(B: int, H: int, N: int, D: int) -> float:
    """Causal attention FLOP count (the standard 2 * 2 * (N*(N+1)/2) * D * B * H)."""
    # QK^T: 2 * BHND for each output element, only lower triangle = N(N+1)/2 pairs
    # PV : 2 * BHND for each output element, same pair count
    pairs = N * (N + 1) / 2
    return 4.0 * B * H * pairs * D


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters",  type=int, default=50)
    p.add_argument("--rtol",   type=float, default=5e-3)
    p.add_argument("--atol",   type=float, default=1e-2)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available"); return 1

    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    Q = torch.randn(args.batch, args.heads, args.seqlen, args.head_dim,
                    generator=g, device=dev, dtype=torch.float16)
    K = torch.randn_like(Q); V = torch.randn_like(Q)

    print(f"GPU       : {torch.cuda.get_device_name(0)}")
    print(f"shape     : B={args.batch} H={args.heads} N={args.seqlen} D={args.head_dim} fp16")
    print()

    # --- numerics ----------------------------------------------------
    print("=== numerics (max abs err vs fp32 reference) ===")
    ref = _reference_attention(Q, K, V)

    out_spike = hopper_causal_fwd(Q, K, V)
    err_spike = (out_spike.float() - ref.float()).abs().max().item()
    print(f"  hopper_spike  : {err_spike:.4e}")

    out_attnfuse = _attnfuse_causal(Q, K, V)
    err_attnfuse = (out_attnfuse.float() - ref.float()).abs().max().item()
    print(f"  attnfuse_prod : {err_attnfuse:.4e}")

    if _HAS_FLEX:
        out_flex = _flex_causal(Q, K, V)
        err_flex = (out_flex.float() - ref.float()).abs().max().item()
        print(f"  flex          : {err_flex:.4e}")
    print()

    parity_ok = err_spike < max(args.atol, args.rtol * ref.abs().max().item())
    if not parity_ok:
        print(f"FAIL: hopper_spike numerics off (err={err_spike:.4e})")
        return 2

    # --- latency ----------------------------------------------------
    print("=== latency (median over %d iters, warmup %d) ===" % (args.iters, args.warmup))
    flops = _causal_flops(args.batch, args.heads, args.seqlen, args.head_dim)

    t_spike    = _time_kernel(hopper_causal_fwd,     Q, K, V,
                              warmup=args.warmup, iters=args.iters)
    t_attnfuse = _time_kernel(_attnfuse_causal,      Q, K, V,
                              warmup=args.warmup, iters=args.iters)
    t_flex     = _time_kernel(_flex_causal,          Q, K, V,
                              warmup=args.warmup, iters=args.iters) if _HAS_FLEX else None

    def _row(name: str, t_ms: float | None):
        if t_ms is None:
            print(f"  {name:<14}  ---")
            return
        tflops = flops / (t_ms * 1e-3) / 1e12
        ratio  = (t_flex / t_ms) if t_flex else float("nan")
        print(f"  {name:<14}  {t_ms:.3f} ms   {tflops:6.1f} TFLOPS   "
              f"vs flex: {ratio:.2f}x")

    _row("hopper_spike",  t_spike)
    _row("attnfuse_prod", t_attnfuse)
    _row("flex",          t_flex)
    print()

    # --- success criterion ------------------------------------------
    if t_flex:
        gap_before = t_attnfuse / t_flex
        gap_after  = t_spike    / t_flex
        improvement = (t_attnfuse - t_spike) / t_attnfuse
        print(f"=== verdict ===")
        print(f"  attnfuse_prod vs flex  : {gap_before:.2f}x")
        print(f"  hopper_spike  vs flex  : {gap_after:.2f}x")
        print(f"  spike improvement      : {improvement*100:+.1f}%")
        if improvement >= 0.30:
            print("  WIN: spike beats production by >= 30%")
        elif improvement >= 0.15:
            print("  PARTIAL: spike beats production by 15-30%")
        else:
            print("  INCONCLUSIVE: spike improvement < 15%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
