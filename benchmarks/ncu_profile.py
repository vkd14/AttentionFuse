"""Nsight Compute profiling harness for AttnFuse kernels.

Replaces the analytical roofline numbers in the paper with measured
hardware-counter data: real tensor-core utilisation, achieved HBM
bandwidth, occupancy, warp-stall breakdown. The analytical numbers
are upper / lower bounds; ncu shows what the GPU *actually* did.

A companion shell script ``scripts/run_ncu_profile.sh`` runs ncu with
the right metric set and writes a CSV per variant. The script targets
modern Nsight Compute (CUDA 12.x+, sm_86 and sm_90); the old
Nsight Compute 2021.1 bundled with CUDA 11.3 lacks the driver-API
calls used by recent PyTorch and is not supported.

Suggested run set for the paper:

    bash scripts/run_ncu_profile.sh causal       4096 fp16
    bash scripts/run_ncu_profile.sh causal_alibi 4096 fp16
    bash scripts/run_ncu_profile.sh dense        4096 fp16
    bash scripts/run_ncu_profile.sh sliding_window 4096 fp16

Each writes results/ncu_<variant>_<N>.csv with the per-metric values.
"""
from __future__ import annotations

import argparse
import torch
import attnfuse as af


@af.attention
def _dense(Q, K, V):
    return af.softmax(af.scaled_dot_product(Q, K)) @ V


@af.attention
def _causal(Q, K, V):
    return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V


@af.attention
def _sw(Q, K, V):
    return af.softmax(af.sliding_window(af.scaled_dot_product(Q, K), 256)) @ V


@af.attention
def _alibi(Q, K, V):
    s = af.scaled_dot_product(Q, K)
    s = af.alibi(s, num_heads=12)
    s = af.causal(s)
    return af.softmax(s) @ V


VARIANTS = {
    "dense":         _dense,
    "causal":        _causal,
    "sliding_window": _sw,
    "causal_alibi":  _alibi,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=list(VARIANTS), default="causal")
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--warmup", type=int, default=3,
                   help="Warmup iters to compile + warm Triton cache. "
                        "ncu's measured launch is the FIRST launch AFTER warmup.")
    args = p.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    if not torch.cuda.is_available():
        print("CUDA not available"); return 1

    g = torch.Generator(device="cuda").manual_seed(0)
    Q = torch.randn(args.batch, args.heads, args.seqlen, args.head_dim,
                    generator=g, device="cuda", dtype=dtype)
    K = torch.randn_like(Q); V = torch.randn_like(Q)
    fn = VARIANTS[args.variant]

    # Warmup -- triggers JIT compile + populates Triton disk cache.
    # ncu will profile every kernel launch in the process, so we want
    # the warmup launches to be "uninteresting" and the FINAL launch
    # to be the one we actually care about.
    for _ in range(args.warmup):
        fn(Q, K, V)
    torch.cuda.synchronize()

    # The launch ncu measures.
    fn(Q, K, V)
    torch.cuda.synchronize()

    print(f"# profiled {args.variant} N={args.seqlen} B={args.batch} "
          f"H={args.heads} D={args.head_dim} {args.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
