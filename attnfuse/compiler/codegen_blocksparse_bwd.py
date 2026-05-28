"""Backward kernels for block-sparse attention.

Mirrors the dense FA2-style backward in attnfuse/compiler/codegen_backward.py
but iterates over the active block lists stored on the BlockMask:

  * dK/dV kernel
      Grid: (n_kv_blocks, B * H_kv)
      For each n-block, loops over the active m-blocks for that
      n-block (the K-major transpose list q_indices / q_num_blocks).
      Accumulates dK and dV; for GQA sums across the GROUP_SIZE Q heads
      in the inner loop (same trick as the dense backward).

  * dQ kernel
      Grid: (n_q_blocks, B * H_q)
      For each m-block, loops over the active n-blocks for that
      m-block (the Q-major list -- shared with forward).
      Accumulates dQ.

The same Triton-3.1 fp32-dot workaround (bf16 cast on the dQ matmul)
is used here -- the bug is in tl.dot's compilation, not in our kernel
structure, and the workaround is unchanged.
"""

_BLOCKSPARSE_BACKWARD_SRC = '''
import triton
import triton.language as tl


@triton.jit
def attnfuse_bs_bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr,
    dK_ptr, dV_ptr,
    # K-major active list: for each n-block, the m-blocks that attend to it
    q_num_blocks_ptr,         # (n_kv_blocks,) int32
    q_indices_ptr,            # (n_kv_blocks, max_q) int32
    stride_qi_n, stride_qi_m,
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
    BIAS_KIND:  tl.constexpr,
    HEAD_DIM:   tl.constexpr,
    BLOCK_M:    tl.constexpr,
    BLOCK_N:    tl.constexpr,
):
    pid_n  = tl.program_id(0)
    pid_bk = tl.program_id(1)
    b    = pid_bk // H_kv
    h_kv = pid_bk %  H_kv

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    K_bh = K_ptr + b * stride_kb + h_kv * stride_kh
    V_bh = V_ptr + b * stride_vb + h_kv * stride_vh
    k_ptrs = K_bh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    v_ptrs = V_bh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    n_in_bounds = offs_n < N_KV
    k = tl.load(k_ptrs, mask=n_in_bounds[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=n_in_bounds[:, None], other=0.0)

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)

    q_count = tl.load(q_num_blocks_ptr + pid_n)
    q_row_base = q_indices_ptr + pid_n * stride_qi_n

    for h_in_group in range(0, GROUP_SIZE):
        h_q = h_kv * GROUP_SIZE + h_in_group

        if BIAS_KIND == 1:
            slope = tl.load(alibi_slopes_ptr + h_q)

        Q_bh  = Q_ptr  + b * stride_qb  + h_q * stride_qh
        dO_bh = dO_ptr + b * stride_dob + h_q * stride_doh
        L_bh  = L_ptr  + b * stride_lb  + h_q * stride_lh
        D_bh  = D_ptr  + b * stride_db  + h_q * stride_dh

        for i in range(q_count):
            m_block = tl.load(q_row_base + i * stride_qi_m)
            cur_m = m_block * BLOCK_M + offs_m
            m_in_bounds = cur_m < N_Q

            q_ptrs  = Q_bh  + cur_m[:, None] * stride_qm  + offs_d[None, :] * stride_qd
            do_ptrs = dO_bh + cur_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
            q  = tl.load(q_ptrs,  mask=m_in_bounds[:, None], other=0.0)
            do = tl.load(do_ptrs, mask=m_in_bounds[:, None], other=0.0)

            L_row = tl.load(L_bh + cur_m * stride_lm, mask=m_in_bounds, other=0.0)
            D_row = tl.load(D_bh + cur_m * stride_dm, mask=m_in_bounds, other=0.0)

            s = tl.dot(q, tl.trans(k)) * sm_scale
            if BIAS_KIND == 1:
                dist = tl.abs(cur_m[:, None] - offs_n[None, :])
                s = s + (-slope.to(tl.float32) * dist.to(tl.float32))
            s = tl.where(m_in_bounds[:, None] & n_in_bounds[None, :],
                         s, -float("inf"))
            p = tl.exp(s - L_row[:, None])

            dv = dv + tl.dot(tl.trans(p).to(do.dtype), do).to(tl.float32)
            dp = tl.dot(do, tl.trans(v)).to(tl.float32)
            ds = p * (dp - D_row[:, None])
            dk = dk + tl.dot(tl.trans(ds).to(q.dtype), q).to(tl.float32)

    dk = dk * sm_scale

    dK_bh = dK_ptr + b * stride_dkb + h_kv * stride_dkh
    dV_bh = dV_ptr + b * stride_dvb + h_kv * stride_dvh
    dk_ptrs = dK_bh + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
    dv_ptrs = dV_bh + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
    tl.store(dk_ptrs, dk.to(dK_ptr.dtype.element_ty), mask=n_in_bounds[:, None])
    tl.store(dv_ptrs, dv.to(dV_ptr.dtype.element_ty), mask=n_in_bounds[:, None])


@triton.jit
def attnfuse_bs_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr,
    dQ_ptr,
    # Q-major active list (same as forward)
    kv_num_blocks_ptr,        # (n_q_blocks,) int32
    kv_indices_ptr,           # (n_q_blocks, max_kv) int32
    stride_kvi_q, stride_kvi_n,
    full_kv_num_ptr,
    full_kv_idx_ptr,
    stride_fkvi_q, stride_fkvi_n,
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
    BIAS_KIND:  tl.constexpr,
    HEAD_DIM:   tl.constexpr,
    BLOCK_M:    tl.constexpr,
    BLOCK_N:    tl.constexpr,
):
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

    K_bh = K_ptr + b * stride_kb + h_kv * stride_kh
    V_bh = V_ptr + b * stride_vb + h_kv * stride_vh

    # Iterate both "full" and "partial" active lists -- in the v1 mask all
    # active tiles end up in full_kv_idx so the partial loop is usually
    # empty, but we cover both for completeness.
    full_count = tl.load(full_kv_num_ptr + pid_m)
    full_base = full_kv_idx_ptr + pid_m * stride_fkvi_q
    for i in range(full_count):
        n_block = tl.load(full_base + i * stride_fkvi_n)
        cur_n = n_block * BLOCK_N + offs_n
        n_in_bounds = cur_n < N_KV
        k_ptrs = K_bh + cur_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V_bh + cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=n_in_bounds[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=n_in_bounds[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k)) * sm_scale
        if BIAS_KIND == 1:
            dist = tl.abs(offs_m[:, None] - cur_n[None, :])
            s = s + (-slope.to(tl.float32) * dist.to(tl.float32))
        s = tl.where(m_in_bounds[:, None] & n_in_bounds[None, :],
                     s, -float("inf"))
        p = tl.exp(s - L_row[:, None])
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - D_row[:, None])
        # Same Triton-3.1 bf16-cast workaround as the dense backward.
        dq = dq + tl.dot(ds.to(tl.bfloat16), k.to(tl.bfloat16)).to(tl.float32)

    partial_count = tl.load(kv_num_blocks_ptr + pid_m)
    partial_base = kv_indices_ptr + pid_m * stride_kvi_q
    for i in range(partial_count):
        n_block = tl.load(partial_base + i * stride_kvi_n)
        cur_n = n_block * BLOCK_N + offs_n
        n_in_bounds = cur_n < N_KV
        k_ptrs = K_bh + cur_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V_bh + cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=n_in_bounds[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=n_in_bounds[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k)) * sm_scale
        if BIAS_KIND == 1:
            dist = tl.abs(offs_m[:, None] - cur_n[None, :])
            s = s + (-slope.to(tl.float32) * dist.to(tl.float32))
        s = tl.where(m_in_bounds[:, None] & n_in_bounds[None, :],
                     s, -float("inf"))
        p = tl.exp(s - L_row[:, None])
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - D_row[:, None])
        dq = dq + tl.dot(ds.to(tl.bfloat16), k.to(tl.bfloat16)).to(tl.float32)

    dq = dq * sm_scale
    dQ_bh = dQ_ptr + b * stride_dqb + h_q * stride_dqh
    dq_ptrs = dQ_bh + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq.to(dQ_ptr.dtype.element_ty), mask=m_in_bounds[:, None])
'''


def get_blocksparse_backward_source() -> str:
    return _BLOCKSPARSE_BACKWARD_SRC
