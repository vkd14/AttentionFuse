"""Hopper-targeted causal forward kernel — Phase 1 spike.

Goal: demonstrate that a WGMMA-friendly tile pipeline closes the H100
gap to flex_attention. Measured baseline (results/ncu/, 2026-06-02):

    causal forward, B=4 H=12 N=4096 D=64 fp16
      HMMA pipe % peak           : 15.8
      SM throughput % peak       : 35.5
      Warp occupancy % peak      : 12.0
      DRAM throughput % peak     : 2.1
      Wait-stall % cycles        : 29.9
      Regs/thread                : 217
      Block size                 : 128 threads
      Latency (vs flex 0.40 ms)  : 0.87 ms  (2.14x behind)

Success criterion for this spike: HMMA pipe >= 35% AND latency drops
>= 30%. Negative result is also reportable: it means Triton 3.3.1 is
not emitting WGMMA at this shape and the path forward is either a
Triton version bump or a CUTLASS / hand-PTX rewrite.

Design choices:

1.  BLOCK_M=128, BLOCK_N=128. WGMMA at fp16 prefers M-aligned-to-64
    accumulators and large-N inner contraction; 128x128 keeps the
    Q tile resident and gives the inner matmul 128 columns to issue
    against. The Ampere causal kernel uses BLOCK_M=128, BLOCK_N=32
    (sparse table); we double BLOCK_N here.

2.  num_warps=8, num_stages=3. Hopper wants 8 warps = 1 warp-group
    per program for WGMMA. 3 stages overlaps 2 K/V loads with the
    compute on the third.

3.  FA-2 outer-loop layout: one program per (batch, head, m_block).
    Inner loop iterates K/V blocks 0..N_KV. Causal skip handled by
    splitting into two ranges:

      a. "Full" tiles where every (m, n) is in-range -> no per-element
         mask, plain tl.dot + online softmax.
      b. "Diagonal" tile where some elements need masking -> apply
         the causal mask, then online softmax.

    This is the standard FA-2 pattern. Reduces masked-tile work to
    one per program instead of (N_KV / BLOCK_N).

4.  TMA descriptor path is conditionally compiled. If
    USE_TMA == 1 we build per-launch tensor descriptors for K and V
    and use tl.load on the descriptor view; else regular pointer
    arithmetic. Both should emit WGMMA on Hopper at this shape.

5.  HEAD_DIM=64 only, fp16 only, MHA (no GQA). The minimal spike.
    Generalising is straightforward once the codegen pattern works.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# Detect whether the installed Triton has the modern TMA descriptor API.
# Triton 3.3+ provides tl.make_tensor_descriptor. We probe at import time
# so the kernel constexpr stays a compile-time constant.
_HAS_TMA = hasattr(tl, "make_tensor_descriptor")


@triton.jit
def _hopper_causal_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    B, H, N,
    GROUP_SIZE: tl.constexpr,    # 1 = MHA, > 1 = GQA (h_kv = h_q // GROUP_SIZE)
    HEAD_DIM: tl.constexpr,
    BLOCK_M:  tl.constexpr,
    BLOCK_N:  tl.constexpr,
):
    """One program per (batch * head, m_block).

    Self-attention only (N_Q == N_KV == N). Causal.
    For GQA/MQA, Q has H heads, K and V have H // GROUP_SIZE heads; each
    program reads K and V at h_kv = h_q // GROUP_SIZE so the same KV head
    is consumed by GROUP_SIZE query heads (one program per query head).
    """
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh %  H
    h_kv = h // GROUP_SIZE       # 0..H_kv-1; equals h when GROUP_SIZE=1 (MHA)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    Q_bh = Q_ptr + b * stride_qb + h    * stride_qh
    K_bh = K_ptr + b * stride_kb + h_kv * stride_kh
    V_bh = V_ptr + b * stride_vb + h_kv * stride_vh
    O_bh = O_ptr + b * stride_ob + h    * stride_oh

    q_ptrs = Q_bh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < N
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # ----------------------------------------------------------------
    # FA-2 causal split:
    #   range 1: full tiles  [0, diag_block_lo)   -- no mask
    #   range 2: diag tile   [diag_block_lo,
    #                         diag_block_lo + BLOCK_N)  -- causal mask
    # The last "full" block is the largest n_start such that the entire
    # BLOCK_N tile lies strictly below the diagonal of this BLOCK_M tile.
    # Diagonal of this program is at n == offs_m, so the largest m in this
    # tile is (pid_m+1)*BLOCK_M - 1. The full-block range covers
    # n_start + BLOCK_N - 1 <= pid_m*BLOCK_M - 1, i.e. n_start <= pid_m*BLOCK_M - BLOCK_N.
    # We round down to a multiple of BLOCK_N.
    # ----------------------------------------------------------------
    m_lo = pid_m * BLOCK_M
    diag_lo = (m_lo // BLOCK_N) * BLOCK_N            # first block that touches the diagonal
    n_full_hi = diag_lo                              # exclusive upper bound for full range
    n_diag_hi = tl.minimum(m_lo + BLOCK_M, N)        # last column attended by this Q tile

    # ----- LOOP 1: full tiles (no causal mask) -----
    for n_start in range(0, n_full_hi, BLOCK_N):
        cur_n  = n_start + offs_n
        k_ptrs = K_bh + cur_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V_bh + cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        s = tl.dot(q, tl.trans(k))                    # (BM, BN), fp32 accumulator
        s = s * sm_scale

        block_max = tl.max(s, axis=1)
        m_new     = tl.maximum(m_i, block_max)
        alpha     = tl.exp(m_i - m_new)
        p         = tl.exp(s - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
        m_i = m_new

    # ----- LOOP 2: diagonal tiles (causal mask + bounds) -----
    for n_start in range(n_full_hi, n_diag_hi, BLOCK_N):
        cur_n  = n_start + offs_n
        k_ptrs = K_bh + cur_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V_bh + cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        n_in_bounds = cur_n < N
        k = tl.load(k_ptrs, mask=n_in_bounds[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=n_in_bounds[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k))
        s = s * sm_scale

        causal_mask = offs_m[:, None] >= cur_n[None, :]
        s = tl.where(causal_mask & n_in_bounds[None, :], s, -float("inf"))

        block_max  = tl.max(s, axis=1)
        m_new      = tl.maximum(m_i, block_max)
        all_masked = (block_max == float("-inf"))
        alpha      = tl.where(all_masked, 1.0, tl.exp(m_i - m_new))
        safe_m_new = tl.where(all_masked, tl.zeros_like(m_new), m_new)
        p          = tl.exp(s - safe_m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
        m_i = m_new

    # Normalise
    acc = acc / l_i[:, None]

    o_ptrs = O_bh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_mask)


def _default_tile_for(head_dim: int) -> tuple[int, int, int, int]:
    """Return (BLOCK_M, BLOCK_N, num_warps, num_stages) tuned per HEAD_DIM.

    D=64: sweep winner on H100 NVL (Session 3) -- BN=64 nw=8 ns=3,
          0.488 ms at N=4096 (matches flex within 10%).
    D=128: conservative starting point. BN=32 keeps per-stage SMEM under
          ~80 KB so 2 blocks fit per SM and occupancy doesn't collapse.
          Should be re-swept on H100 to refine.
    """
    if head_dim == 64:
        return (128, 64, 8, 3)
    if head_dim == 128:
        return (128, 32, 8, 3)
    raise ValueError(f"spike supports HEAD_DIM in (64, 128), got {head_dim}")


def hopper_causal_fwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    block_m: int | None = None,
    block_n: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    warp_specialize: bool = True,
) -> torch.Tensor:
    """Hopper-spike launcher.

    Q has shape (B, H_q, N, D). K and V have shape (B, H_kv, N, D) with
    H_q % H_kv == 0. ``H_q == H_kv`` is plain MHA; ``H_q > H_kv`` is GQA
    with GROUP_SIZE = H_q // H_kv.

    Tile config defaults to the per-HEAD_DIM sweep winner; explicitly
    set the four block_* kwargs to override (used by the sweep harness).

    ``warp_specialize`` enables Triton 3.3+ producer/consumer warp split
    on sm_90+. Falls back to a non-specialised launch if the installed
    Triton does not accept the kwarg (validated on Triton 3.1).
    """
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "CUDA tensors only"
    assert Q.dim() == K.dim() == V.dim() == 4
    B, H_q, N, D = Q.shape
    H_kv = K.shape[1]
    assert K.shape == (B, H_kv, N, D) and V.shape == (B, H_kv, N, D), \
        f"K/V shape mismatch: got {K.shape} {V.shape}; expected K, V " \
        f"with (B={B}, H_kv, N={N}, D={D})"
    assert H_q % H_kv == 0, f"H_q ({H_q}) must be a multiple of H_kv ({H_kv})"
    assert D in (64, 128), f"spike supports HEAD_DIM in (64, 128), got {D}"
    assert Q.dtype in (torch.float16, torch.bfloat16), \
        f"spike supports fp16/bf16 only, got {Q.dtype}"

    group_size = H_q // H_kv

    # Tile config: use the per-D default unless caller explicitly overrides.
    default_bm, default_bn, default_nw, default_ns = _default_tile_for(D)
    bm = block_m   if block_m   is not None else default_bm
    bn = block_n   if block_n   is not None else default_bn
    nw = num_warps if num_warps is not None else default_nw
    ns = num_stages if num_stages is not None else default_ns

    O = torch.empty_like(Q)
    sm_scale = D ** -0.5

    grid = (triton.cdiv(N, bm), B * H_q, 1)
    launch_kwargs = dict(
        GROUP_SIZE=group_size,
        HEAD_DIM=D,
        BLOCK_M=bm,
        BLOCK_N=bn,
        num_warps=nw,
        num_stages=ns,
    )
    if warp_specialize:
        launch_kwargs["warp_specialize"] = True

    def _launch(**extra):
        _hopper_causal_fwd_kernel[grid](
            Q, K, V, O,
            sm_scale,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            B, H_q, N,
            **extra,
        )

    try:
        _launch(**launch_kwargs)
    except (TypeError, KeyError) as e:
        # Older Triton (<3.3): warp_specialize kwarg not accepted. Drop and retry.
        if "warp_specialize" not in str(e):
            raise
        launch_kwargs.pop("warp_specialize", None)
        _launch(**launch_kwargs)
    return O


__all__ = ["hopper_causal_fwd", "_hopper_causal_fwd_kernel"]
