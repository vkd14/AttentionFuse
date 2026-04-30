"""A manually-written FlashAttention-style Triton kernel (forward only).

This is the reference upper bound: it is the same algorithm AttnFuse generates
but written by hand for the dense-causal case only. We use it to measure how
close the AttnFuse-generated kernel comes to a hand-tuned implementation.

Adapted from the Triton tutorial reference (06-fused-attention.py); kept
deliberately minimal and *not* extended with our DSL features.
"""
from __future__ import annotations

import math
import torch

try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except ImportError:
    _TRITON_OK = False


if _TRITON_OK:

    @triton.jit
    def _flash_fwd(
        Q, K, V, sm_scale, Out,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        B, H, N,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        pid_m  = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh %  H

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)

        Qbh = Q + b * stride_qb + h * stride_qh
        Kbh = K + b * stride_kb + h * stride_kh
        Vbh = V + b * stride_vb + h * stride_vh
        Obh = Out + b * stride_ob + h * stride_oh

        q_ptrs = Qbh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        q_mask = offs_m[:, None] < N
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        n_hi = tl.minimum((pid_m + 1) * BLOCK_M, N) if IS_CAUSAL else N

        for n_start in range(0, n_hi, BLOCK_N):
            cur_n = n_start + offs_n
            n_in = cur_n < N
            k_ptrs = Kbh + cur_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
            v_ptrs = Vbh + cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            k = tl.load(k_ptrs, mask=n_in[:, None], other=0.0)
            v = tl.load(v_ptrs, mask=n_in[:, None], other=0.0)

            s = tl.dot(q, tl.trans(k)) * sm_scale
            if IS_CAUSAL:
                s = tl.where(offs_m[:, None] >= cur_n[None, :], s, -float("inf"))
            s = tl.where(n_in[None, :], s, -float("inf"))

            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            alpha = tl.exp(m_i - m_new)
            p     = tl.exp(s - m_new[:, None])
            l_i   = alpha * l_i + tl.sum(p, axis=1)
            acc   = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
            m_i   = m_new

        out = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
        o_ptrs = Obh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
        tl.store(o_ptrs, out.to(Out.dtype.element_ty), mask=q_mask)


def flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                    *, causal: bool = False) -> torch.Tensor:
    """Self-attention forward via the hand-written Triton kernel above."""
    if not _TRITON_OK:
        raise RuntimeError("Triton not available; cannot run the manual flash kernel.")

    B, H, N, D = Q.shape
    out = torch.empty_like(Q)

    BLOCK_M = 128
    BLOCK_N = 64 if D <= 96 else 32
    grid = ((N + BLOCK_M - 1) // BLOCK_M, B * H)

    _flash_fwd[grid](
        Q, K, V, 1.0 / math.sqrt(D), out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, N,
        HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, IS_CAUSAL=causal,
        num_warps=4, num_stages=3,
    )
    return out
