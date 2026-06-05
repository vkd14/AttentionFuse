"""Session 3 Prong A -- parameter sweep over the Hopper spike kernel.

Sweeps the three Hopper-relevant launch parameters:
    num_stages in {2, 3, 4}
    BLOCK_N    in {64, 128, 256}
    num_warps  in {4, 8}
BLOCK_M is fixed at 128 (FA-2 outer-loop block; Hopper WGMMA aligned).

For each config the script:
  1. Validates numerics against an fp32 reference (must be within 5e-3).
  2. Measures wall-clock via cuda events (median of 50 iters, 10 warmup).
  3. Prints a table sorted by latency.

The cheapest configs likely have:
  * num_stages=2  -> lower SMEM/block -> >=2 blocks/SM -> higher occupancy
  * BLOCK_N=64    -> shorter dep chain per matmul -> fewer wait stalls
But these trade against arithmetic intensity per program. The sweep
tells us where the curve bottoms.

Run on H100:
    python -m benchmarks.hopper_spike_sweep --seqlen 4096
"""
from __future__ import annotations

import argparse
import itertools
import statistics
import time
from dataclasses import dataclass
from typing import Optional

import torch

from attnfuse.experimental.hopper_causal_fwd import hopper_causal_fwd


@dataclass
class _Row:
    block_n: int
    num_warps: int
    num_stages: int
    latency_ms: Optional[float]
    tflops:     Optional[float]
    max_err:    Optional[float]
    note:       str = ""


def _reference(Q, K, V) -> torch.Tensor:
    B, H, N, D = Q.shape
    sm_scale = D ** -0.5
    S = torch.einsum("bhmd,bhnd->bhmn", Q.float(), K.float()) * sm_scale
    causal = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
    S = S.masked_fill(causal, float("-inf"))
    P = torch.softmax(S, dim=-1)
    return torch.einsum("bhmn,bhnd->bhmd", P, V.float()).to(Q.dtype)


def _time_kernel(fn, Q, K, V, *, warmup=10, iters=50) -> float:
    for _ in range(warmup):
        fn(Q, K, V)
    torch.cuda.synchronize()
    events = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(Q, K, V); e.record()
        events.append((s, e))
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in events)


def _causal_flops(B, H, N, D) -> float:
    return 4.0 * B * H * (N * (N + 1) / 2) * D


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--rtol", type=float, default=5e-3)
    p.add_argument("--atol", type=float, default=1e-2)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available"); return 1

    g = torch.Generator(device="cuda").manual_seed(0)
    Q = torch.randn(args.batch, args.heads, args.seqlen, args.head_dim,
                    generator=g, device="cuda", dtype=torch.float16)
    K = torch.randn_like(Q); V = torch.randn_like(Q)
    ref = _reference(Q, K, V)
    flops = _causal_flops(args.batch, args.heads, args.seqlen, args.head_dim)

    print(f"GPU       : {torch.cuda.get_device_name(0)}")
    print(f"shape     : B={args.batch} H={args.heads} N={args.seqlen} "
          f"D={args.head_dim} fp16  (BLOCK_M=128 fixed)\n")

    rows: list[_Row] = []
    configs = list(itertools.product([2, 3, 4], [64, 128, 256], [4, 8]))
    for num_stages, block_n, num_warps in configs:
        # SMEM/block heuristic: warp_specialize keeps an extra buffer.
        # Empirically num_warps=4 with BLOCK_N=256 + num_stages=4 OOMs SMEM
        # on H100 (228 KB cap); skip configs we know will fail to compile.
        approx_smem_kb = (
            128 * args.head_dim * 2 +                      # Q tile
            num_stages * 2 * block_n * args.head_dim * 2   # K,V * stages
        ) / 1024
        if approx_smem_kb > 220:
            rows.append(_Row(block_n, num_warps, num_stages, None, None, None,
                             f"skipped (SMEM ~{approx_smem_kb:.0f} KB > 220)"))
            continue
        t0 = time.time()
        try:
            def _fn(q, k, v, bn=block_n, nw=num_warps, ns=num_stages):
                return hopper_causal_fwd(q, k, v,
                                          block_m=128, block_n=bn,
                                          num_warps=nw, num_stages=ns,
                                          warp_specialize=True)
            O = _fn(Q, K, V)
            err = (O.float() - ref.float()).abs().max().item()
            if err > max(args.atol, args.rtol * ref.abs().max().item()):
                rows.append(_Row(block_n, num_warps, num_stages, None, None, err,
                                 "FAIL numerics"))
                continue
            latency = _time_kernel(_fn, Q, K, V, warmup=10, iters=50)
            tflops  = flops / (latency * 1e-3) / 1e12
            rows.append(_Row(block_n, num_warps, num_stages, latency, tflops, err))
        except Exception as ex:
            msg = str(ex).split("\n")[0][:80]
            rows.append(_Row(block_n, num_warps, num_stages, None, None, None,
                             f"crash: {msg}"))
        print(f"  ns={num_stages} BN={block_n:3d} nw={num_warps}  "
              f"{rows[-1].latency_ms or float('nan'):>7.3f} ms  "
              f"{rows[-1].note or ''}  ({time.time()-t0:.1f}s)")

    # Sort by latency ascending. Failed rows go to the bottom.
    rows.sort(key=lambda r: r.latency_ms if r.latency_ms is not None else float('inf'))

    print()
    print("=" * 76)
    print(f"{'rank':>4}  {'ns':>2}  {'BN':>3}  {'nw':>2}  "
          f"{'latency':>10}  {'TFLOPS':>8}  {'err':>10}  note")
    print("-" * 76)
    for i, r in enumerate(rows, 1):
        lat = f"{r.latency_ms:.3f} ms" if r.latency_ms else "---"
        tfl = f"{r.tflops:.1f}"        if r.tflops    else "---"
        err = f"{r.max_err:.2e}"       if r.max_err   else "---"
        print(f"{i:>4}  {r.num_stages:>2}  {r.block_n:>3}  {r.num_warps:>2}  "
              f"{lat:>10}  {tfl:>8}  {err:>10}  {r.note}")
    print("=" * 76)

    # Reference targets
    print()
    print("Reference (from Session 2 on H100 NVL, same shape):")
    print("  attnfuse_prod : 0.931 ms    (Ampere kernel template)")
    print("  hopper_spike  : 0.694 ms    (BM=128 BN=128 nw=8 ns=3, ws=True no-op)")
    print("  flex          : 0.443 ms    (target -- gap to close)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
