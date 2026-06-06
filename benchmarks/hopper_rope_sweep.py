"""Session 7 Prong A -- tile sweep over the Hopper RoPE+causal spike.

Session 6 showed the default (BN=64, ns=2, nw=8) at 0.943 ms for N=4096
RoPE+causal, vs flex+pre-rotate at 0.835 ms (AttnFuse 0.89x, losing).
The flat 2x cost over plain causal across all N suggests per-tile
rotation overhead, not setup cost. This sweep checks whether a
different (BN, ns, nw) recovers performance.

Hypotheses to test:

  H1. num_stages=3 (the plain-causal winner) might fit even with RoPE's
      extra register pressure on H100 (1 MB regs/SM vs Ampere's 256 KB).
      The Session 6 default of ns=2 was a conservative choice ported
      from the Ampere RoPE table; Hopper may not need it.

  H2. Larger BN amortises cos/sin/K_rot_half loads over more matmul
      work per iteration. BN=128 or BN=256 might win even though those
      lost on plain causal.

  H3. nw=4 (4-warp blocks, matching flex's choice) might reduce the
      per-program reg pressure budget enough to keep ns=3 viable.

For each config: numerics check at the headline shape (N=4096), then
wall-clock at multiple N to see whether the speedup vs flex+pre-rotate
recovers at long context (where the Session 6 default lost ground).

Run:
    python -m benchmarks.hopper_rope_sweep --seqlen 4096
    python -m benchmarks.hopper_rope_sweep --seqlen 16384  (for the long N check)
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
from attnfuse.rope_utils import build_rope_cache, apply_rope

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    _HAS_FLEX = True
    _FLEX_COMPILED = torch.compile(flex_attention, dynamic=False, fullgraph=True)
except ImportError:
    _HAS_FLEX = False
    _FLEX_COMPILED = None


@dataclass
class _Row:
    block_n: int
    num_warps: int
    num_stages: int
    latency_ms: Optional[float]
    tflops:     Optional[float]
    max_err:    Optional[float]
    speedup:    Optional[float]    # vs flex+pre-rotate
    note:       str = ""


def _reference(Q, K, V, cos, sin) -> torch.Tensor:
    """fp32 reference: rotate Q, K, then causal softmax(QK^T)V."""
    B, H, N, D = Q.shape
    Qr = apply_rope(Q, cos, sin).float()
    Kr = apply_rope(K, cos, sin).float()
    S = torch.einsum("bhmd,bhnd->bhmn", Qr, Kr) * (D ** -0.5)
    causal = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
    S = S.masked_fill(causal, float("-inf"))
    P = torch.softmax(S, dim=-1)
    return torch.einsum("bhmn,bhnd->bhmd", P, V.float()).to(Q.dtype)


def _flex_rope_causal(Q, K, V, cos, sin, block_mask):
    Qr = apply_rope(Q, cos, sin)
    Kr = apply_rope(K, cos, sin)
    return _FLEX_COMPILED(Qr, Kr, V, block_mask=block_mask)


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


def _causal_block_mask(N, device):
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    return create_block_mask(causal, B=None, H=None, Q_LEN=N, KV_LEN=N, device=device)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--batch",    type=int, default=4)
    p.add_argument("--heads",    type=int, default=12)
    p.add_argument("--seqlen",   type=int, default=4096)
    p.add_argument("--head_dim", type=int, default=64)
    p.add_argument("--rtol",     type=float, default=5e-3)
    p.add_argument("--atol",     type=float, default=1e-2)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available"); return 1

    g = torch.Generator(device="cuda").manual_seed(0)
    Q = torch.randn(args.batch, args.heads, args.seqlen, args.head_dim,
                    generator=g, device="cuda", dtype=torch.float16)
    K = torch.randn_like(Q); V = torch.randn_like(Q)
    cos, sin = build_rope_cache(args.seqlen, args.head_dim,
                                 device="cuda", dtype=torch.float16)
    ref = _reference(Q, K, V, cos, sin)
    flops = _causal_flops(args.batch, args.heads, args.seqlen, args.head_dim)

    # flex+pre-rotate baseline for the speedup column.
    if _HAS_FLEX:
        bm = _causal_block_mask(args.seqlen, Q.device)
        t_flex = _time_kernel(lambda q, k, v: _flex_rope_causal(q, k, v, cos, sin, bm),
                               Q, K, V, warmup=10, iters=50)
    else:
        t_flex = None

    print(f"GPU       : {torch.cuda.get_device_name(0)}")
    print(f"shape     : B={args.batch} H={args.heads} N={args.seqlen} "
          f"D={args.head_dim} fp16  RoPE+causal  (BLOCK_M=128 fixed)")
    print(f"flex+rotate baseline: {t_flex:.3f} ms" if t_flex else "no flex available")
    print()

    rows: list[_Row] = []
    configs = list(itertools.product([2, 3], [32, 64, 128, 256], [4, 8]))
    for num_stages, block_n, num_warps in configs:
        # SMEM heuristic. RoPE itself loads cos/sin/K_rot_half into registers,
        # not SMEM, so the pipelined SMEM is the same as plain causal: just
        # Q + ns * (K + V).
        approx_smem_kb = (
            128 * args.head_dim * 2 +
            num_stages * 2 * block_n * args.head_dim * 2
        ) / 1024
        if approx_smem_kb > 220:
            rows.append(_Row(block_n, num_warps, num_stages, None, None, None, None,
                             f"skipped (SMEM ~{approx_smem_kb:.0f} KB > 220)"))
            continue
        t0 = time.time()
        try:
            def _fn(q, k, v, bn=block_n, nw=num_warps, ns=num_stages):
                return hopper_causal_fwd(q, k, v, cos=cos, sin=sin,
                                          block_m=128, block_n=bn,
                                          num_warps=nw, num_stages=ns,
                                          warp_specialize=True)
            O = _fn(Q, K, V)
            err = (O.float() - ref.float()).abs().max().item()
            if err > max(args.atol, args.rtol * ref.abs().max().item()):
                rows.append(_Row(block_n, num_warps, num_stages, None, None, err, None,
                                 "FAIL numerics"))
                continue
            latency = _time_kernel(_fn, Q, K, V, warmup=10, iters=50)
            tflops  = flops / (latency * 1e-3) / 1e12
            speedup = (t_flex / latency) if t_flex else None
            rows.append(_Row(block_n, num_warps, num_stages,
                              latency, tflops, err, speedup))
        except Exception as ex:
            msg = str(ex).split("\n")[0][:60]
            rows.append(_Row(block_n, num_warps, num_stages, None, None, None, None,
                             f"crash: {msg}"))
        last = rows[-1]
        print(f"  ns={num_stages} BN={block_n:3d} nw={num_warps}  "
              f"{(last.latency_ms or float('nan')):>7.3f} ms  "
              f"{last.note or f'speedup {last.speedup:.2f}x'}  ({time.time()-t0:.1f}s)")

    rows.sort(key=lambda r: r.latency_ms if r.latency_ms is not None else float('inf'))

    print()
    print("=" * 90)
    print(f"{'rank':>4}  {'ns':>2}  {'BN':>3}  {'nw':>2}  "
          f"{'latency':>10}  {'TFLOPS':>8}  {'speedup':>9}  {'err':>10}  note")
    print("-" * 90)
    for i, r in enumerate(rows, 1):
        lat = f"{r.latency_ms:.3f} ms" if r.latency_ms else "---"
        tfl = f"{r.tflops:.1f}"        if r.tflops    else "---"
        spd = f"{r.speedup:.2f}x"      if r.speedup   else "---"
        err = f"{r.max_err:.2e}"       if r.max_err   else "---"
        print(f"{i:>4}  {r.num_stages:>2}  {r.block_n:>3}  {r.num_warps:>2}  "
              f"{lat:>10}  {tfl:>8}  {spd:>9}  {err:>10}  {r.note}")
    print("=" * 90)

    print()
    print("Reference (Session 6 default: BN=64 ns=2 nw=8):")
    print(f"  AttnFuse RoPE+causal : 0.943 ms  (lost to flex by 12%)")
    print(f"  flex+pre-rotate      : 0.835 ms  (this run measured: "
          f"{t_flex:.3f} ms)" if t_flex else "")
    print()
    print("Plain causal reference (Session 5 winner):")
    print(f"  Plain causal spike   : 0.489 ms  (BN=64 ns=3 nw=8)")
    print(f"  RoPE overhead at S6  : 0.943 - 0.489 = 0.454 ms  (~93% over plain)")
    print("  If the sweep winner here drops RoPE overhead below 0.2 ms,")
    print("  we beat flex+pre-rotate cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
