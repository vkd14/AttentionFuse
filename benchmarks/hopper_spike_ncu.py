"""ncu-targetable single-kernel-launch driver for the Hopper spike.

Mirrors benchmarks/ncu_profile.py but launches the spike kernel instead.
ncu attaches with --kernel-name '_hopper_causal_fwd_kernel'.

Usage:
    bash scripts/run_hopper_spike_ncu.sh   # writes results/ncu/ncu_hopper_*.csv
"""
from __future__ import annotations

import argparse

import torch

from attnfuse.experimental.hopper_causal_fwd import hopper_causal_fwd


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seqlen",   type=int, default=4096)
    p.add_argument("--batch",    type=int, default=4)
    p.add_argument("--heads",    type=int, default=12)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--warmup",   type=int, default=4)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available"); return 1

    g = torch.Generator(device="cuda").manual_seed(0)
    Q = torch.randn(args.batch, args.heads, args.seqlen, args.head_dim,
                    generator=g, device="cuda", dtype=torch.float16)
    K = torch.randn_like(Q); V = torch.randn_like(Q)

    for _ in range(args.warmup):
        hopper_causal_fwd(Q, K, V)
    torch.cuda.synchronize()

    hopper_causal_fwd(Q, K, V)
    torch.cuda.synchronize()

    print(f"# profiled hopper_spike causal N={args.seqlen} B={args.batch} "
          f"H={args.heads} D={args.head_dim} fp16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
