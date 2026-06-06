"""Hopper-targeted causal forward kernel — Phase 1 spike.

Goal: demonstrate that a WGMMA-friendly tile pipeline closes the H100
gap to flex_attention. Measured H100 NVL results (results/ncu/):

  Plain causal at B=4 H=12 N=4096 D=64 fp16:
      Production (Ampere template)    : 0.931 ms   HMMA 15.8%
      Spike (Session 5 defaults)      : 0.489 ms   HMMA 29.4%
      flex_attention                  : 0.443 ms   HMMA 32.6%

The spike is at parity with flex's tensor-core time
(0.144 ms each); the residual 10% wall-clock gap is non-matmul FA-2
overhead, structural to the algorithm at this shape.

Session 6 adds RoPE+causal -- the production LLM attention pattern.
flex_attention cannot fuse RoPE (its score_mod hook fires after the
QK^T matmul), so the comparison is structural: AttnFuse fuses the
rotation, flex pays an extra HBM round-trip to materialise rotated
Q', K' tensors before the kernel. Expected speedup on H100 mirrors
the 2.10x measured on RTX 3090 for this composition.

Design choices (Session 6 additions in brackets):

1.  FA-2 outer-loop layout: one program per (batch, head, m_block).
    Inner loop iterates K/V blocks 0..N_KV. Causal handled by
    splitting into "full tiles" (no per-element mask) and one
    "diagonal tile" (causal mask + bounds).

2.  Tile config: per-HEAD_DIM defaults; see ``_default_tile_for``.
    [Session 6: RoPE adds cos/sin/rotated-half loads in registers
    per tile, raising register pressure. Use num_stages=2 instead of
    3 to keep within the per-thread reg budget on H100.]

3.  GROUP_SIZE constexpr: 1 for MHA, >1 for GQA (Llama-3 ratios 4:1, 8:1).

4.  HEAD_DIM in {64, 128}. D=64 sweep-validated (Session 3); D=128
    uses a conservative tile that should be re-swept on H100.

5.  [Session 6 -- ROPE_KIND constexpr:
        0 = no RoPE (the Session 1-5 path, unchanged at runtime).
        1 = NeoX-style fused RoPE: rotate Q once before the inner loop
            (loads cos/sin/Q-rotate-half in registers, no extra HBM
            round-trip vs flex's pre-rotated Q'); rotate each K tile
            in-place inside the inner loop. Same algebra as the
            production codegen.py path; ported byte-for-byte.]
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


_HAS_TMA = hasattr(tl, "make_tensor_descriptor")


@triton.jit
def _hopper_causal_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    cos_ptr, sin_ptr,                       # only read when ROPE_KIND == 1
    stride_rope_n, stride_rope_d,
    B, H, N,
    GROUP_SIZE: tl.constexpr,    # 1 = MHA, > 1 = GQA (h_kv = h_q // GROUP_SIZE)
    HEAD_DIM: tl.constexpr,
    BLOCK_M:  tl.constexpr,
    BLOCK_N:  tl.constexpr,
    ROPE_KIND: tl.constexpr,     # 0 = no RoPE, 1 = NeoX-style fused RoPE
):
    """One program per (batch * head, m_block). Self-attention. Causal."""
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh %  H
    h_kv = h // GROUP_SIZE

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # RoPE rotate-half index mapping: d < D/2 -> d+D/2, else -> d-D/2.
    # rot_sign: -1 for first half (negated x), +1 for second half.
    d_half     = HEAD_DIM // 2
    rot_offs_d = tl.where(offs_d < d_half, offs_d + d_half, offs_d - d_half)
    rot_sign   = tl.where(offs_d < d_half, -1.0, 1.0)

    Q_bh = Q_ptr + b * stride_qb + h    * stride_qh
    K_bh = K_ptr + b * stride_kb + h_kv * stride_kh
    V_bh = V_ptr + b * stride_vb + h_kv * stride_vh
    O_bh = O_ptr + b * stride_ob + h    * stride_oh

    q_ptrs = Q_bh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < N
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # ----- Fused RoPE: rotate Q tile ONCE before the inner loop -----
    if ROPE_KIND == 1:
        q_cos_ptrs = cos_ptr + offs_m[:, None] * stride_rope_n + offs_d[None, :] * stride_rope_d
        q_sin_ptrs = sin_ptr + offs_m[:, None] * stride_rope_n + offs_d[None, :] * stride_rope_d
        q_cos = tl.load(q_cos_ptrs, mask=q_mask, other=0.0)
        q_sin = tl.load(q_sin_ptrs, mask=q_mask, other=0.0)
        q_rh_ptrs = Q_bh + offs_m[:, None] * stride_qm + rot_offs_d[None, :] * stride_qd
        q_rot_half = tl.load(q_rh_ptrs, mask=q_mask, other=0.0)
        q = (q.to(tl.float32) * q_cos.to(tl.float32)
             + q_rot_half.to(tl.float32) * rot_sign[None, :] * q_sin.to(tl.float32)
             ).to(q.dtype)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    m_lo = pid_m * BLOCK_M
    diag_lo = (m_lo // BLOCK_N) * BLOCK_N
    n_full_hi = diag_lo
    n_diag_hi = tl.minimum(m_lo + BLOCK_M, N)

    # ----- LOOP 1: full tiles (no causal mask) -----
    for n_start in range(0, n_full_hi, BLOCK_N):
        cur_n  = n_start + offs_n
        k_ptrs = K_bh + cur_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V_bh + cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        # ----- Fused RoPE: rotate K tile in-place every iteration -----
        if ROPE_KIND == 1:
            k_cos_ptrs = cos_ptr + cur_n[:, None] * stride_rope_n + offs_d[None, :] * stride_rope_d
            k_sin_ptrs = sin_ptr + cur_n[:, None] * stride_rope_n + offs_d[None, :] * stride_rope_d
            k_cos = tl.load(k_cos_ptrs)
            k_sin = tl.load(k_sin_ptrs)
            k_rh_ptrs = K_bh + cur_n[:, None] * stride_kn + rot_offs_d[None, :] * stride_kd
            k_rot_half = tl.load(k_rh_ptrs)
            k = (k.to(tl.float32) * k_cos.to(tl.float32)
                 + k_rot_half.to(tl.float32) * rot_sign[None, :] * k_sin.to(tl.float32)
                 ).to(k.dtype)

        s = tl.dot(q, tl.trans(k))
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

        if ROPE_KIND == 1:
            k_cos_ptrs = cos_ptr + cur_n[:, None] * stride_rope_n + offs_d[None, :] * stride_rope_d
            k_sin_ptrs = sin_ptr + cur_n[:, None] * stride_rope_n + offs_d[None, :] * stride_rope_d
            k_cos = tl.load(k_cos_ptrs, mask=n_in_bounds[:, None], other=0.0)
            k_sin = tl.load(k_sin_ptrs, mask=n_in_bounds[:, None], other=0.0)
            k_rh_ptrs = K_bh + cur_n[:, None] * stride_kn + rot_offs_d[None, :] * stride_kd
            k_rot_half = tl.load(k_rh_ptrs, mask=n_in_bounds[:, None], other=0.0)
            k = (k.to(tl.float32) * k_cos.to(tl.float32)
                 + k_rot_half.to(tl.float32) * rot_sign[None, :] * k_sin.to(tl.float32)
                 ).to(k.dtype)

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

    acc = acc / l_i[:, None]

    o_ptrs = O_bh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_mask)


def _default_tile_for(head_dim: int, *, rope: bool = False) -> tuple[int, int, int, int]:
    """Return (BLOCK_M, BLOCK_N, num_warps, num_stages) tuned per HEAD_DIM.

    D=64 (no RoPE): sweep winner Session 3 -- BN=64 nw=8 ns=3,
        0.488 ms at N=4096 (matches flex within 10%).
    D=128 (no RoPE): conservative -- BN=32 nw=8 ns=3, ~2 blocks/SM.

    RoPE variants: add cos/sin/rotated-half loads in registers per tile.
    Pressure forces num_stages=2 to keep within Hopper's per-thread reg
    budget. Should be re-swept on H100; these are starting points that
    parallel the Ampere RoPE table's stage drop.
    """
    if head_dim == 64:
        return (128, 64, 8, 2) if rope else (128, 64, 8, 3)
    if head_dim == 128:
        return (128, 32, 8, 2) if rope else (128, 32, 8, 3)
    raise ValueError(f"spike supports HEAD_DIM in (64, 128), got {head_dim}")


def hopper_causal_fwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
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

    If ``cos`` and ``sin`` are provided (shape (N, D) or (1, 1, N, D)),
    the kernel runs the fused RoPE + causal path. Otherwise it runs the
    plain causal path (the Session 1-5 behaviour).

    Tile config defaults to the per-HEAD_DIM / per-RoPE picks from
    ``_default_tile_for``; explicitly set the four ``block_*`` kwargs
    to override (used by the sweep harness).

    ``warp_specialize`` enables Triton 3.3+ producer/consumer warp split
    on sm_90+. Falls back gracefully on older Triton.
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

    use_rope = cos is not None and sin is not None
    if (cos is None) != (sin is None):
        raise ValueError("cos and sin must both be provided or both omitted")

    if use_rope:
        # Accept (N, D) or (1, 1, N, D); squeeze to (N, D) for the kernel.
        c = cos.squeeze(0).squeeze(0) if cos.dim() == 4 else cos
        s = sin.squeeze(0).squeeze(0) if sin.dim() == 4 else sin
        assert c.shape == (N, D) and s.shape == (N, D), \
            f"cos/sin must be shape (N={N}, D={D}); got {c.shape} {s.shape}"
        assert c.dtype == Q.dtype and s.dtype == Q.dtype, \
            f"cos/sin dtype must match Q ({Q.dtype}); got {c.dtype} {s.dtype}"
        c = c.contiguous()
        s = s.contiguous()
    else:
        # Placeholder pointers (kernel never reads them when ROPE_KIND=0).
        c = torch.empty(1, device=Q.device, dtype=Q.dtype)
        s = torch.empty(1, device=Q.device, dtype=Q.dtype)

    group_size = H_q // H_kv
    default_bm, default_bn, default_nw, default_ns = _default_tile_for(D, rope=use_rope)
    bm = block_m   if block_m   is not None else default_bm
    bn = block_n   if block_n   is not None else default_bn
    nw = num_warps if num_warps is not None else default_nw
    ns = num_stages if num_stages is not None else default_ns

    O = torch.empty_like(Q)
    sm_scale = D ** -0.5

    # For non-RoPE we still pass valid stride args; kernel won't use them.
    rope_stride_n = c.stride(0) if use_rope else 0
    rope_stride_d = c.stride(1) if use_rope else 0

    grid = (triton.cdiv(N, bm), B * H_q, 1)
    launch_kwargs = dict(
        GROUP_SIZE=group_size,
        HEAD_DIM=D,
        BLOCK_M=bm,
        BLOCK_N=bn,
        ROPE_KIND=1 if use_rope else 0,
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
            c, s,
            rope_stride_n, rope_stride_d,
            B, H_q, N,
            **extra,
        )

    try:
        _launch(**launch_kwargs)
    except (TypeError, KeyError) as e:
        if "warp_specialize" not in str(e):
            raise
        launch_kwargs.pop("warp_specialize", None)
        _launch(**launch_kwargs)
    return O


__all__ = ["hopper_causal_fwd", "_hopper_causal_fwd_kernel"]
