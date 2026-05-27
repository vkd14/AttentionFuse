"""Codegen for the FlashAttention-2-style backward kernels.

Algorithm (Dao 2023, FlashAttention-2 backward):

  Forward saves L = m + log(sum(exp(s - m)))  per row (B, H_q, N_q).

  Backward inputs:  Q, K, V, O, L, dO
  Backward outputs: dQ (B, H_q, N_q, D),  dK, dV (B, H_kv, N_kv, D)

  Pre-compute:      D = rowsum(dO * O)            shape (B, H_q, N_q)

  Per (i, j) tile:  S  = Q_i K_j^T * scale        (apply mask, bias)
                    P  = exp(S - L_i)             (recompute attention probs)
                    dV_j += P^T @ dO_i
                    dP = dO_i @ V_j^T
                    dS = P * (dP - D_i[:, None])
                    dK_j += dS^T @ Q_i * scale
                    dQ_i += dS @ K_j * scale

We compile TWO kernels:

  attnfuse_bwd_dkv_kernel
    Grid: (cdiv(N_kv, BLOCK_N),  B * H_kv)
    Holds K_j, V_j in registers across an m-loop that sweeps every
    (b, h_q-in-group) pair. Atomic-free GQA reduction: the inner loop
    sums over all H_q / H_kv query heads sharing this KV head.

  attnfuse_bwd_dq_kernel
    Grid: (cdiv(N_q, BLOCK_M), B * H_q)
    Holds Q_i, O_i, dO_i, L_i in registers across an n-loop.

Initial scope:
  * MASK_KIND in {0 (dense), 1 (causal)}     -- sliding-window is future work
  * BIAS_KIND in {0 (none), 1 (ALiBi)}       -- additive bias backward is straightforward
                                                if needed; no current users
  * NORM_KIND = 0 (softmax)
  * ROPE_KIND = 0 (no fused RoPE in backward; pre-process if needed)
"""

_BACKWARD_KERNEL_SRC = '''
import triton
import triton.language as tl


@triton.jit
def attnfuse_bwd_preproc_kernel(
    O_ptr, dO_ptr, D_ptr,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_db, stride_dh, stride_dm,
    B, H, N_Q,
    HEAD_DIM: tl.constexpr,
    BLOCK_M:  tl.constexpr,
):
    """Pre-compute D = rowsum(dO * O), shape (B, H_q, N_q) in fp32."""
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh %  H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    m_mask = offs_m[:, None] < N_Q

    o_ptrs  = O_ptr  + b * stride_ob  + h * stride_oh  + offs_m[:, None] * stride_om  + offs_d[None, :] * stride_od
    do_ptrs = dO_ptr + b * stride_dob + h * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
    o  = tl.load(o_ptrs,  mask=m_mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=m_mask, other=0.0).to(tl.float32)

    D = tl.sum(o * do, axis=1)
    d_ptrs = D_ptr + b * stride_db + h * stride_dh + offs_m * stride_dm
    tl.store(d_ptrs, D, mask=offs_m < N_Q)


@triton.jit
def attnfuse_bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr,
    dK_ptr, dV_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_lb, stride_lh, stride_lm,
    stride_db, stride_dh, stride_dm,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    B, H_q, H_kv, N_Q, N_KV,
    alibi_slopes_ptr,
    GROUP_SIZE: tl.constexpr,
    MASK_KIND:  tl.constexpr,
    BIAS_KIND:  tl.constexpr,
    HEAD_DIM:   tl.constexpr,
    BLOCK_M:    tl.constexpr,
    BLOCK_N:    tl.constexpr,
):
    """dK/dV kernel: one program per (n_block, b*H_kv).

    Each program holds (BLOCK_N, HEAD_DIM) K and V tiles and sweeps all
    query rows. dK and dV are accumulated in fp32 then written. For GQA
    each (n_block, b, h_kv) program inner-loops over the GROUP_SIZE Q heads
    that share this KV head, naturally summing their dK / dV contributions.
    """
    pid_n  = tl.program_id(0)
    pid_bk = tl.program_id(1)
    b    = pid_bk // H_kv
    h_kv = pid_bk %  H_kv

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # Load K_j, V_j once for this n-block
    K_bh = K_ptr + b * stride_kb + h_kv * stride_kh
    V_bh = V_ptr + b * stride_vb + h_kv * stride_vh
    k_ptrs = K_bh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    v_ptrs = V_bh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    n_in_bounds = offs_n < N_KV
    k = tl.load(k_ptrs, mask=n_in_bounds[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=n_in_bounds[:, None], other=0.0)

    # dK, dV accumulators (fp32)
    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    # Inner loop: for each Q head sharing this KV head, sum contributions
    for h_in_group in range(0, GROUP_SIZE):
        h_q = h_kv * GROUP_SIZE + h_in_group

        if BIAS_KIND == 1:
            slope = tl.load(alibi_slopes_ptr + h_q)

        Q_bh  = Q_ptr  + b * stride_qb  + h_q * stride_qh
        dO_bh = dO_ptr + b * stride_dob + h_q * stride_doh
        L_bh  = L_ptr  + b * stride_lb  + h_q * stride_lh
        D_bh  = D_ptr  + b * stride_db  + h_q * stride_dh

        # For causal masks we can skip query blocks entirely below pid_n*BLOCK_N
        # (no Q row at index m < pid_n*BLOCK_N can attend to ANY n in this block).
        if MASK_KIND == 1:
            m_lo = (pid_n * BLOCK_N) // BLOCK_M * BLOCK_M
        else:
            m_lo = 0
        m_hi = N_Q

        for m_start in range(m_lo, m_hi, BLOCK_M):
            cur_m = m_start + offs_m
            m_in_bounds = cur_m < N_Q

            q_ptrs  = Q_bh  + cur_m[:, None] * stride_qm  + offs_d[None, :] * stride_qd
            do_ptrs = dO_bh + cur_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
            q  = tl.load(q_ptrs,  mask=m_in_bounds[:, None], other=0.0)
            do = tl.load(do_ptrs, mask=m_in_bounds[:, None], other=0.0)

            l_ptrs = L_bh + cur_m * stride_lm
            d_ptrs = D_bh + cur_m * stride_dm
            L_row  = tl.load(l_ptrs, mask=m_in_bounds, other=0.0)
            D_row  = tl.load(d_ptrs, mask=m_in_bounds, other=0.0)

            # Recompute S = Q K^T * scale
            s = tl.dot(q, tl.trans(k)) * sm_scale

            # Bias
            if BIAS_KIND == 1:
                dist = tl.abs(cur_m[:, None] - offs_n[None, :])
                s = s + (-slope.to(tl.float32) * dist.to(tl.float32))

            # Mask
            if MASK_KIND == 1:                              # causal
                causal_mask = cur_m[:, None] >= offs_n[None, :]
                s = tl.where(causal_mask, s, -float("inf"))
            s = tl.where(m_in_bounds[:, None] & n_in_bounds[None, :],
                         s, -float("inf"))

            # P = exp(S - L) -- using saved log-sum-exp
            p = tl.exp(s - L_row[:, None])

            # dV += P^T @ dO
            dv = dv + tl.dot(tl.trans(p).to(do.dtype), do).to(tl.float32)

            # dP = dO @ V^T
            dp = tl.dot(do, tl.trans(v)).to(tl.float32)

            # dS = P * (dP - D[:, None])
            ds = p * (dp - D_row[:, None])

            # dK += dS^T @ Q * scale  (scale applied at end)
            dk = dk + tl.dot(tl.trans(ds).to(q.dtype), q).to(tl.float32)

    # Apply final scale to dK (Q^T @ dS was scaled in S = QK^T * scale, so dK has scale factor)
    dk = dk * sm_scale

    # Store dK, dV (additive into pre-zeroed output; safe because each
    # (b, h_kv, n_block) is owned by exactly one program)
    dK_bh = dK_ptr + b * stride_dkb + h_kv * stride_dkh
    dV_bh = dV_ptr + b * stride_dvb + h_kv * stride_dvh
    dk_ptrs = dK_bh + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
    dv_ptrs = dV_bh + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
    tl.store(dk_ptrs, dk.to(dK_ptr.dtype.element_ty), mask=n_in_bounds[:, None])
    tl.store(dv_ptrs, dv.to(dV_ptr.dtype.element_ty), mask=n_in_bounds[:, None])


@triton.jit
def attnfuse_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr,
    dQ_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_lb, stride_lh, stride_lm,
    stride_db, stride_dh, stride_dm,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    B, H_q, H_kv, N_Q, N_KV,
    alibi_slopes_ptr,
    GROUP_SIZE: tl.constexpr,
    MASK_KIND:  tl.constexpr,
    BIAS_KIND:  tl.constexpr,
    HEAD_DIM:   tl.constexpr,
    BLOCK_M:    tl.constexpr,
    BLOCK_N:    tl.constexpr,
):
    """dQ kernel: one program per (m_block, b*H_q)."""
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b   = pid_bh // H_q
    h_q = pid_bh %  H_q
    h_kv = h_q // GROUP_SIZE

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    m_in_bounds = offs_m < N_Q

    if BIAS_KIND == 1:
        slope = tl.load(alibi_slopes_ptr + h_q)

    # Load Q_i, dO_i, L_i, D_i once
    Q_bh  = Q_ptr  + b * stride_qb  + h_q * stride_qh
    dO_bh = dO_ptr + b * stride_dob + h_q * stride_doh
    L_bh  = L_ptr  + b * stride_lb  + h_q * stride_lh
    D_bh  = D_ptr  + b * stride_db  + h_q * stride_dh

    q_ptrs  = Q_bh  + offs_m[:, None] * stride_qm  + offs_d[None, :] * stride_qd
    do_ptrs = dO_bh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
    q  = tl.load(q_ptrs,  mask=m_in_bounds[:, None], other=0.0)
    do = tl.load(do_ptrs, mask=m_in_bounds[:, None], other=0.0)

    L_row = tl.load(L_bh + offs_m * stride_lm, mask=m_in_bounds, other=0.0)
    D_row = tl.load(D_bh + offs_m * stride_dm, mask=m_in_bounds, other=0.0)

    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    if MASK_KIND == 1:
        n_hi = tl.minimum((pid_m + 1) * BLOCK_M, N_KV)
    else:
        n_hi = N_KV

    K_bh = K_ptr + b * stride_kb + h_kv * stride_kh
    V_bh = V_ptr + b * stride_vb + h_kv * stride_vh

    for n_start in range(0, n_hi, BLOCK_N):
        cur_n = n_start + offs_n
        n_in_bounds = cur_n < N_KV

        k_ptrs = K_bh + cur_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V_bh + cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=n_in_bounds[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=n_in_bounds[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k)) * sm_scale

        if BIAS_KIND == 1:
            dist = tl.abs(offs_m[:, None] - cur_n[None, :])
            s = s + (-slope.to(tl.float32) * dist.to(tl.float32))

        if MASK_KIND == 1:
            causal_mask = offs_m[:, None] >= cur_n[None, :]
            s = tl.where(causal_mask, s, -float("inf"))
        s = tl.where(m_in_bounds[:, None] & n_in_bounds[None, :],
                     s, -float("inf"))

        p = tl.exp(s - L_row[:, None])
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - D_row[:, None])
        # ds @ k via tensor cores. Triton 3.1's tl.dot(fp32, fp32) silently
        # produces wrong results for this reduction direction; bf16 inputs
        # use the tensor-core path correctly and bf16 has fp32's exponent
        # range so the cast is lossless except for mantissa rounding.
        dq = dq + tl.dot(ds.to(tl.bfloat16), k.to(tl.bfloat16)).to(tl.float32)

    dq = dq * sm_scale
    dQ_bh = dQ_ptr + b * stride_dqb + h_q * stride_dqh
    dq_ptrs = dQ_bh + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(dQ_ptr.dtype.element_ty), mask=m_in_bounds[:, None])
'''


def get_backward_source() -> str:
    return _BACKWARD_KERNEL_SRC
