"""ncu-targetable driver for the torch.compile'd flex_attention kernel.

Setting an HMMA-pipe upper bound: what's the best a torch.compile'd
Triton kernel achieves on this exact shape on H100? If flex sits at
~60% HMMA we know the spike has 35 percentage points of headroom and
the gap is structural. If flex sits at ~30% we're closer to the ceiling
than the wall-clock suggests.

Flex's compiled kernel name is dynamic (Inductor generates something
like `triton_per_fused__flex_attention_*`). The wrapper script in
``scripts/run_flex_ncu.sh`` uses a permissive regex (`triton.*flex.*`)
so ncu attaches to whatever Inductor named it.

Usage:
    bash scripts/run_flex_ncu.sh   # writes results/ncu/ncu_flex_4096_fp16.csv
"""
from __future__ import annotations

import argparse

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


_FLEX_COMPILED = torch.compile(flex_attention, dynamic=False, fullgraph=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seqlen",   type=int, default=4096)
    p.add_argument("--batch",    type=int, default=4)
    p.add_argument("--heads",    type=int, default=12)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--warmup",   type=int, default=6,
                   help="High enough to land torch.compile + autotune.")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available"); return 1

    g = torch.Generator(device="cuda").manual_seed(0)
    Q = torch.randn(args.batch, args.heads, args.seqlen, args.head_dim,
                    generator=g, device="cuda", dtype=torch.float16)
    K = torch.randn_like(Q); V = torch.randn_like(Q)

    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    bm = create_block_mask(causal, B=None, H=None,
                            Q_LEN=args.seqlen, KV_LEN=args.seqlen,
                            device="cuda")

    # Warmup -- the FIRST iteration triggers torch.compile + Inductor
    # autotuning + Triton JIT. We need enough iters that ncu's launch-skip
    # window covers the autotune launches AND the compiled-kernel landing
    # launch, so the measured launch is the steady-state flex kernel.
    for _ in range(args.warmup):
        _FLEX_COMPILED(Q, K, V, block_mask=bm)
    torch.cuda.synchronize()

    # The launch ncu measures.
    _FLEX_COMPILED(Q, K, V, block_mask=bm)
    torch.cuda.synchronize()

    print(f"# profiled flex_attention causal N={args.seqlen} B={args.batch} "
          f"H={args.heads} D={args.head_dim} fp16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
