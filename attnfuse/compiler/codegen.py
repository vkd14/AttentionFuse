"""Codegen: TiledKernel -> compiled Triton kernel.

We emit a single, parameterised kernel where mask / bias / norm choices are
selected at compile time via `tl.constexpr` flags. Triton specialises the
kernel per (MASK_KIND, BIAS_KIND, NORM_KIND, BLOCK_M, BLOCK_N, head_dim, dtype)
combination, so this is genuinely zero-overhead dispatch at run time.

The fused loop is the well-known online-softmax tile loop:

    for n_block in range(n_lo, n_hi, BLOCK_N):
        K_blk, V_blk = load
        S = Q_tile @ K_blk.T * scale
        S += bias                                  (constexpr-gated)
        S += additive_mask                         (constexpr-gated)
        m_new = max(m_i, rowmax(S))
        alpha = exp(m_i - m_new)
        p     = exp(S - m_new)
        l_new = alpha * l_i + rowsum(p)
        acc   = alpha * acc + p @ V_blk
        m_i, l_i = m_new, l_new
    O = acc / l_i

For the ReLU normalisation we drop the m_i/exp/rescale machinery and just
accumulate `acc += relu(S) @ V_blk`, dividing by `l_i = sum(relu(S))` at the end.
"""
from __future__ import annotations

from ..ir.high_level import MaskKind, BiasKind, NormKind
from ..ir.tiled import TiledKernel

# Encoding for constexpr flags (keep stable; codegen template hardcodes them).
_MASK_FULL, _MASK_CAUSAL, _MASK_SLIDING = 0, 1, 2
_BIAS_NONE, _BIAS_ALIBI, _BIAS_ADDITIVE = 0, 1, 2
_NORM_SOFTMAX, _NORM_RELU = 0, 1


def _encode_mask(k: MaskKind) -> int:
    return {MaskKind.FULL: _MASK_FULL,
            MaskKind.CAUSAL: _MASK_CAUSAL,
            MaskKind.SLIDING_WINDOW: _MASK_SLIDING}[k]


def _encode_bias(k: BiasKind | None) -> int:
    if k is None:
        return _BIAS_NONE
    return {BiasKind.ALIBI: _BIAS_ALIBI,
            BiasKind.ADDITIVE: _BIAS_ADDITIVE}[k]


def _encode_norm(k: NormKind) -> int:
    return {NormKind.SOFTMAX: _NORM_SOFTMAX,
            NormKind.RELU: _NORM_RELU}[k]


# ---------------------------------------------------------------------------
# Kernel source
# ---------------------------------------------------------------------------
#
# We define the kernel ONCE at module import time, then re-use it across
# TiledKernels via Triton's constexpr-based specialisation cache.

_KERNEL_SRC = '''
import triton
import triton.language as tl

@triton.jit
def attnfuse_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    B, H, N,
    alibi_slopes_ptr,                  # only read when BIAS_KIND == 1
    bias_ptr,                          # only read when BIAS_KIND == 2
    stride_biasb, stride_biash, stride_biasm, stride_biasn,
    cos_ptr, sin_ptr,                  # only read when ROPE_KIND == 1; shape (N, D)
    stride_rope_n, stride_rope_d,
    WINDOW: tl.constexpr,              # only read when MASK_KIND  == 2
    HEAD_DIM: tl.constexpr,
    BLOCK_M:  tl.constexpr,
    BLOCK_N:  tl.constexpr,
    MASK_KIND: tl.constexpr,           # 0 full, 1 causal, 2 sliding-window
    BIAS_KIND: tl.constexpr,           # 0 none, 1 alibi, 2 additive-external
    NORM_KIND: tl.constexpr,           # 0 softmax, 1 relu
    ROPE_KIND: tl.constexpr,           # 0 none, 1 fused RoPE (Su et al., 2021)
    SKIP_EMPTY: tl.constexpr,
):
    """One program per (batch, head, m_block)."""
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh %  H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # RoPE rotate-half index mapping: d < D/2 → d+D/2, else → d-D/2
    # rot_sign: -1 for first half (negated), +1 for second half
    d_half     = HEAD_DIM // 2
    rot_offs_d = tl.where(offs_d < d_half, offs_d + d_half, offs_d - d_half)
    rot_sign   = tl.where(offs_d < d_half, -1.0, 1.0)

    # Pointers to the (b, h) slice
    Q_bh = Q_ptr + b * stride_qb + h * stride_qh
    K_bh = K_ptr + b * stride_kb + h * stride_kh
    V_bh = V_ptr + b * stride_vb + h * stride_vh
    O_bh = O_ptr + b * stride_ob + h * stride_oh

    # Load Q tile (BLOCK_M, HEAD_DIM) once, keep in registers / SMEM
    q_ptrs = Q_bh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < N
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Fused RoPE: rotate Q tile before the inner loop
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

    # Online-softmax accumulators (in fp32 for numerical stability)
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # ALiBi: load this head's slope once
    if BIAS_KIND == 1:
        slope = tl.load(alibi_slopes_ptr + h)

    # Determine the n-block range for this m-block.
    # Note: combine constexpr conditions with separate if/elif rather than
    # `A and B` to avoid Triton tl.constexpr.__bool__ evaluation issues.
    if MASK_KIND == 1:                        # causal — always skip upper blocks
        n_lo = 0
        n_hi = tl.minimum((pid_m + 1) * BLOCK_M, N)
    elif MASK_KIND == 2:                      # sliding window — always trim both ends
        m_lo = pid_m * BLOCK_M
        m_hi = m_lo + BLOCK_M
        n_lo = tl.maximum(m_lo - WINDOW + 1, 0)
        n_hi = tl.minimum(m_hi + WINDOW, N)
    else:                                     # full / dense
        n_lo = 0
        n_hi = N

    # Round n_lo down to BLOCK_N boundary so the mask logic is uniform
    n_lo = (n_lo // BLOCK_N) * BLOCK_N

    if MASK_KIND == 2:
        # Sliding-window: split the inner loop into
        #   [n_lo, interior_lo)        — left boundary  (apply SW mask)
        #   [interior_lo, interior_hi) — interior        (NO mask: ~45% of tiles)
        #   [interior_hi, n_hi)        — right boundary (apply SW mask)
        # Interior tile [n_start, n_start+BLOCK_N) is "fully inside the window"
        # iff for ALL (m, n) in [m_lo, m_hi) x [n_start, n_start+BLOCK_N):
        # -W < m - n < W. Equivalently:
        #   n_start >= m_hi - W   (worst case: m = m_hi - 1)
        #   n_start <= m_lo + W - BLOCK_N   (worst case: m = m_lo, n = n_start+BLOCK_N-1)
        interior_lo_raw = tl.maximum(m_hi - WINDOW, n_lo)
        interior_lo     = ((interior_lo_raw + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
        interior_hi_max = m_lo + WINDOW - BLOCK_N
        interior_hi     = (interior_hi_max // BLOCK_N) * BLOCK_N + BLOCK_N
        interior_hi     = tl.minimum(interior_hi, n_hi)
        # Critical: the interior loop drops the n_in_bounds mask for speed,
        # so every cur_n in [interior_lo, interior_hi) MUST satisfy cur_n < N.
        # Clamp interior_hi to the last BLOCK_N-aligned position that is also
        # <= N; the trailing partial tile (when N % BLOCK_N != 0) is handled
        # by the right-boundary loop, which keeps the n_in_bounds mask.
        interior_hi     = tl.minimum(interior_hi, (N // BLOCK_N) * BLOCK_N)
        interior_lo     = tl.minimum(interior_lo, interior_hi)

        # ----- LOOP 1: left boundary (apply SW mask + safe softmax) -----
        for n_start in range(n_lo, interior_lo, BLOCK_N):
            cur_n = n_start + offs_n
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
            if BIAS_KIND == 1:
                dist = tl.abs(offs_m[:, None] - cur_n[None, :])
                s = s + (-slope * dist.to(tl.float32))
            elif BIAS_KIND == 2:
                b_ptrs = (bias_ptr + b * stride_biasb + h * stride_biash
                          + offs_m[:, None] * stride_biasm + cur_n[None, :] * stride_biasn)
                b_mask = (offs_m[:, None] < N) & (cur_n[None, :] < N)
                b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
                s = s + b_tile.to(tl.float32)
            d = offs_m[:, None] - cur_n[None, :]
            sw_mask = (d < WINDOW) & (d > -WINDOW)
            s = tl.where(sw_mask, s, -float("inf"))
            s = tl.where(n_in_bounds[None, :], s, -float("inf"))
            if NORM_KIND == 0:
                block_max  = tl.max(s, axis=1)
                m_new      = tl.maximum(m_i, block_max)
                all_masked = (block_max == float("-inf"))
                alpha      = tl.where(all_masked, 1.0, tl.exp(m_i - m_new))
                safe_m_new = tl.where(all_masked, tl.zeros_like(m_new), m_new)
                p          = tl.exp(s - safe_m_new[:, None])
                l_i = alpha * l_i + tl.sum(p, axis=1)
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
                m_i = m_new
            else:
                p = tl.maximum(s, 0.0)
                l_i = l_i + tl.sum(p, axis=1)
                acc = acc + tl.dot(p.to(v.dtype), v).to(tl.float32)

        # ----- LOOP 2: interior (no SW mask, plain online softmax) -----
        # In this range all keys are guaranteed in-window for every query row
        # and within [0, N), so no mask logic is needed at all.
        for n_start in range(interior_lo, interior_hi, BLOCK_N):
            cur_n = n_start + offs_n
            k_ptrs = K_bh + cur_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            v_ptrs = V_bh + cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
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
            if BIAS_KIND == 1:
                dist = tl.abs(offs_m[:, None] - cur_n[None, :])
                s = s + (-slope * dist.to(tl.float32))
            elif BIAS_KIND == 2:
                b_ptrs = (bias_ptr + b * stride_biasb + h * stride_biash
                          + offs_m[:, None] * stride_biasm + cur_n[None, :] * stride_biasn)
                b_tile = tl.load(b_ptrs)
                s = s + b_tile.to(tl.float32)
            if NORM_KIND == 0:
                m_new = tl.maximum(m_i, tl.max(s, axis=1))
                alpha = tl.exp(m_i - m_new)
                p     = tl.exp(s - m_new[:, None])
                l_i = alpha * l_i + tl.sum(p, axis=1)
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
                m_i = m_new
            else:
                p = tl.maximum(s, 0.0)
                l_i = l_i + tl.sum(p, axis=1)
                acc = acc + tl.dot(p.to(v.dtype), v).to(tl.float32)

        # ----- LOOP 3: right boundary (apply SW mask + safe softmax) -----
        for n_start in range(interior_hi, n_hi, BLOCK_N):
            cur_n = n_start + offs_n
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
            if BIAS_KIND == 1:
                dist = tl.abs(offs_m[:, None] - cur_n[None, :])
                s = s + (-slope * dist.to(tl.float32))
            elif BIAS_KIND == 2:
                b_ptrs = (bias_ptr + b * stride_biasb + h * stride_biash
                          + offs_m[:, None] * stride_biasm + cur_n[None, :] * stride_biasn)
                b_mask = (offs_m[:, None] < N) & (cur_n[None, :] < N)
                b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
                s = s + b_tile.to(tl.float32)
            d = offs_m[:, None] - cur_n[None, :]
            sw_mask = (d < WINDOW) & (d > -WINDOW)
            s = tl.where(sw_mask, s, -float("inf"))
            s = tl.where(n_in_bounds[None, :], s, -float("inf"))
            if NORM_KIND == 0:
                block_max  = tl.max(s, axis=1)
                m_new      = tl.maximum(m_i, block_max)
                all_masked = (block_max == float("-inf"))
                alpha      = tl.where(all_masked, 1.0, tl.exp(m_i - m_new))
                safe_m_new = tl.where(all_masked, tl.zeros_like(m_new), m_new)
                p          = tl.exp(s - safe_m_new[:, None])
                l_i = alpha * l_i + tl.sum(p, axis=1)
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
                m_i = m_new
            else:
                p = tl.maximum(s, 0.0)
                l_i = l_i + tl.sum(p, axis=1)
                acc = acc + tl.dot(p.to(v.dtype), v).to(tl.float32)
    else:
        # Dense / causal: single loop (MASK_KIND in {0, 1}); causal trims n_hi.
        for n_start in range(n_lo, n_hi, BLOCK_N):
            cur_n = n_start + offs_n
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
            if BIAS_KIND == 1:
                dist = tl.abs(offs_m[:, None] - cur_n[None, :])
                s = s + (-slope * dist.to(tl.float32))
            elif BIAS_KIND == 2:
                b_ptrs = (bias_ptr + b * stride_biasb + h * stride_biash
                          + offs_m[:, None] * stride_biasm + cur_n[None, :] * stride_biasn)
                b_mask = (offs_m[:, None] < N) & (cur_n[None, :] < N)
                b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
                s = s + b_tile.to(tl.float32)
            if MASK_KIND == 1:
                causal_mask = offs_m[:, None] >= cur_n[None, :]
                s = tl.where(causal_mask, s, -float("inf"))
            s = tl.where(n_in_bounds[None, :], s, -float("inf"))
            if NORM_KIND == 0:
                m_new = tl.maximum(m_i, tl.max(s, axis=1))
                alpha = tl.exp(m_i - m_new)
                p     = tl.exp(s - m_new[:, None])
                l_i   = alpha * l_i + tl.sum(p, axis=1)
                acc   = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
                m_i   = m_new
            else:
                p   = tl.maximum(s, 0.0)
                l_i = l_i + tl.sum(p, axis=1)
                acc = acc + tl.dot(p.to(v.dtype), v).to(tl.float32)

    # Final normalisation
    out = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]

    o_ptrs = O_bh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out.to(O_ptr.dtype.element_ty), mask=q_mask)
'''


def generate_triton_source(kernel: TiledKernel) -> str:
    """Return the kernel source string. Identical for every TiledKernel --
    specialisation happens via constexpr launch arguments."""
    return _KERNEL_SRC


def kernel_constexprs(kernel: TiledKernel) -> dict:
    """Constexpr launch kwargs for the generated kernel."""
    return {
        "HEAD_DIM": kernel.head_dim,
        "BLOCK_M": kernel.config.BLOCK_M,
        "BLOCK_N": kernel.config.BLOCK_N,
        "MASK_KIND": _encode_mask(kernel.mask_kind),
        "BIAS_KIND": _encode_bias(kernel.bias_kind),
        "NORM_KIND": _encode_norm(kernel.norm_kind),
        "ROPE_KIND": kernel.rope_kind,
        "SKIP_EMPTY": int(kernel.config.skip_full_mask_blocks),
        "WINDOW": kernel.mask_window or 0,
    }


def kernel_launch_meta(kernel: TiledKernel) -> dict:
    """Triton launch hints (num_warps, num_stages)."""
    return {
        "num_warps": kernel.config.num_warps,
        "num_stages": kernel.config.num_stages,
    }
