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
_BIAS_NONE, _BIAS_ALIBI = 0, 1
_NORM_SOFTMAX, _NORM_RELU = 0, 1


def _encode_mask(k: MaskKind) -> int:
    return {MaskKind.FULL: _MASK_FULL,
            MaskKind.CAUSAL: _MASK_CAUSAL,
            MaskKind.SLIDING_WINDOW: _MASK_SLIDING}[k]


def _encode_bias(k: BiasKind | None) -> int:
    if k is None:
        return _BIAS_NONE
    return {BiasKind.ALIBI: _BIAS_ALIBI}[k]


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
    WINDOW: tl.constexpr,              # only read when MASK_KIND  == 2
    HEAD_DIM: tl.constexpr,
    BLOCK_M:  tl.constexpr,
    BLOCK_N:  tl.constexpr,
    MASK_KIND: tl.constexpr,           # 0 full, 1 causal, 2 sliding-window
    BIAS_KIND: tl.constexpr,           # 0 none, 1 alibi
    NORM_KIND: tl.constexpr,           # 0 softmax, 1 relu
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

    # Pointers to the (b, h) slice
    Q_bh = Q_ptr + b * stride_qb + h * stride_qh
    K_bh = K_ptr + b * stride_kb + h * stride_kh
    V_bh = V_ptr + b * stride_vb + h * stride_vh
    O_bh = O_ptr + b * stride_ob + h * stride_oh

    # Load Q tile (BLOCK_M, HEAD_DIM) once, keep in registers / SMEM
    q_ptrs = Q_bh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < N
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Online-softmax accumulators (in fp32 for numerical stability)
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # ALiBi: load this head's slope once
    if BIAS_KIND == 1:
        slope = tl.load(alibi_slopes_ptr + h)

    # Determine the n-block range for this m-block
    if MASK_KIND == 1 and SKIP_EMPTY:        # causal
        n_lo = 0
        n_hi = tl.minimum((pid_m + 1) * BLOCK_M, N)
    elif MASK_KIND == 2 and SKIP_EMPTY:      # sliding window
        m_lo = pid_m * BLOCK_M
        m_hi = m_lo + BLOCK_M
        n_lo = tl.maximum(m_lo - WINDOW + 1, 0)
        n_hi = tl.minimum(m_hi + WINDOW, N)
    else:
        n_lo = 0
        n_hi = N

    # Round n_lo down to BLOCK_N boundary so the mask logic is uniform
    n_lo = (n_lo // BLOCK_N) * BLOCK_N

    for n_start in range(n_lo, n_hi, BLOCK_N):
        cur_n = n_start + offs_n
        k_ptrs = K_bh + cur_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V_bh + cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        n_in_bounds = cur_n < N

        k = tl.load(k_ptrs, mask=n_in_bounds[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=n_in_bounds[:, None], other=0.0)

        # S = q @ k.T  -> (BLOCK_M, BLOCK_N)
        s = tl.dot(q, tl.trans(k))
        s = s * sm_scale

        # ----- Bias -----
        if BIAS_KIND == 1:                                 # ALiBi
            # ALiBi adds  -slope * |i - j|  (or  -slope * (i - j)  for causal).
            # We use the unsigned |i - j| form; for causal, j > i is masked
            # out anyway so the sign-of-difference does not matter.
            dist = tl.abs(offs_m[:, None] - cur_n[None, :])
            s = s + (-slope * dist.to(tl.float32))

        # ----- Mask -----
        if MASK_KIND == 1:                                 # causal
            causal_mask = offs_m[:, None] >= cur_n[None, :]
            s = tl.where(causal_mask, s, -float("inf"))
        elif MASK_KIND == 2:                               # sliding window
            d = offs_m[:, None] - cur_n[None, :]
            sw_mask = (d < WINDOW) & (d > -WINDOW)
            s = tl.where(sw_mask, s, -float("inf"))

        # Out-of-N keys are masked too (handles N % BLOCK_N != 0)
        s = tl.where(n_in_bounds[None, :], s, -float("inf"))

        if NORM_KIND == 0:
            # ----- Online softmax -----
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p     = tl.exp(s - m_new[:, None])
            l_i   = alpha * l_i + tl.sum(p, axis=1)
            acc   = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
            m_i   = m_new
        else:
            # ----- ReLU attention -----
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
        "SKIP_EMPTY": int(kernel.config.skip_full_mask_blocks),
        "WINDOW": kernel.mask_window or 0,
    }


def kernel_launch_meta(kernel: TiledKernel) -> dict:
    """Triton launch hints (num_warps, num_stages)."""
    return {
        "num_warps": kernel.config.num_warps,
        "num_stages": kernel.config.num_stages,
    }
