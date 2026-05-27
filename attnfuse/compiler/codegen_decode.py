"""Codegen for Flash Decoding (split-K attention) with tensor-core matmul.

The standard attention kernel launches one program per (batch, query-head,
m_block). For autoregressive decoding (Q.N=1) this gives only B*H_q programs
-- typically 32 for an Llama-3-8B-style model -- which under-utilises the
RTX 3090's 82 SMs and scales linearly with cache length.

Flash Decoding (Dao 2023) fixes this by:
  1. Splitting the KV axis across NUM_SPLITS parallel programs.
  2. Computing partial (m, l, acc) per chunk.
  3. Merging them with the log-sum-exp algebra in a small combine kernel.

This implementation additionally **batches BLOCK_H Q heads per program**
so each program loads K and V exactly once (for GQA, the BLOCK_H heads
that share a KV head reuse the same K/V tile). To keep Triton's
``tl.dot`` happy (which requires M, N, K >= 16) we always set BLOCK_H = 16,
cyclically replicating Q-head indices when GROUP_SIZE < 16 (typical GQA
case) and masking the partial-result stores so duplicate rows are
silently dropped. This recovers full tensor-core throughput on the
Q @ K^T and P @ V matmuls.

Two kernels live here:

  attnfuse_decode_split_kernel
    Grid: (NUM_SPLITS, B * (H_q / chunk_size))
    where chunk_size = min(GROUP_SIZE, BLOCK_H). Each program computes
    partial (m, l, acc) for one KV chunk over `chunk_size` Q heads.

  attnfuse_decode_combine_kernel
    Grid: (B * H_q,)
    For each (b, h_q) reads all NUM_SPLITS partial outputs and merges
    them via log-sum-exp into the final O[b, h_q, :].

Restrictions of the initial scope:
  * Dense mask (MASK_KIND = 0)
  * No additive-tensor bias (BIAS_KIND in {0, 1})  -- ALiBi is supported
  * No fused RoPE inside the decode kernel  (preprocess if needed)
  * Softmax norm only
  * Q.N == 1 (the canonical decode shape)
"""

_DECODE_KERNEL_SRC = '''
import triton
import triton.language as tl

@triton.jit
def attnfuse_decode_split_kernel(
    Q_ptr, K_ptr, V_ptr,
    Wm_ptr, Wl_ptr, Wacc_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_wm_s, stride_wm_b, stride_wm_h,
    stride_wl_s, stride_wl_b, stride_wl_h,
    stride_wacc_s, stride_wacc_b, stride_wacc_h, stride_wacc_d,
    B, H, N_KV,
    alibi_slopes_ptr,
    GROUP_SIZE: tl.constexpr,           # constexpr so the GQA/MQA branch specialises
    BIAS_KIND: tl.constexpr,
    HEAD_DIM:  tl.constexpr,
    BLOCK_N:   tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_H:   tl.constexpr,            # padded to 16 for tl.dot compatibility
):
    """One program per (split_id, batch * head-chunk).

    Each program owns ``chunk_size`` consecutive query heads that all
    share a single ``h_kv``. Q is loaded into a (BLOCK_H, HEAD_DIM) tile
    where BLOCK_H >= chunk_size (cyclic-replicated as needed); the dot
    products use tensor cores; partial-result stores are masked so only
    the first ``chunk_size`` rows write to the workspace.
    """
    split_id = tl.program_id(0)
    pid_bh   = tl.program_id(1)

    # chunk_size = min(GROUP_SIZE, BLOCK_H). For GQA (group_size 4 / 8),
    # chunk_size = group_size and one program covers one whole group with
    # BLOCK_H - chunk_size padding rows. For MQA (group_size >= 16),
    # chunk_size = BLOCK_H and multiple programs cover one big group.
    if GROUP_SIZE <= BLOCK_H:
        chunk_size: tl.constexpr = GROUP_SIZE
    else:
        chunk_size: tl.constexpr = BLOCK_H
    n_chunks  = H // chunk_size

    b      = pid_bh // n_chunks
    chunk  = pid_bh %  n_chunks
    h_base = chunk * chunk_size

    # Per-row Q head index. For GROUP_SIZE < BLOCK_H we cyclic-replicate
    # the group; for GROUP_SIZE >= BLOCK_H we just take BLOCK_H
    # consecutive heads (all in the same group).
    offs_within = tl.arange(0, BLOCK_H)
    if GROUP_SIZE < BLOCK_H:
        offs_h_local = offs_within % GROUP_SIZE
        is_unique    = offs_within < GROUP_SIZE
    else:
        offs_h_local = offs_within
        is_unique    = offs_within < BLOCK_H            # all rows real
    offs_h = h_base + offs_h_local

    # h_kv is the same for every head in this program
    h_kv = h_base // GROUP_SIZE

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # Load Q tile: (BLOCK_H, HEAD_DIM) -- one row per Q head (or replicated)
    q_ptrs = Q_ptr + b * stride_qb + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs)

    # Split bounds along the KV axis
    split_len = (N_KV + NUM_SPLITS - 1) // NUM_SPLITS
    n_lo = split_id * split_len
    n_hi = tl.minimum(n_lo + split_len, N_KV)
    n_lo = (n_lo // BLOCK_N) * BLOCK_N

    K_bh = K_ptr + b * stride_kb + h_kv * stride_kh
    V_bh = V_ptr + b * stride_vb + h_kv * stride_vh

    if BIAS_KIND == 1:
        slope = tl.load(alibi_slopes_ptr + offs_h)      # (BLOCK_H,) (replicated for GQA)

    # Online-softmax accumulators (per row)
    m_i = tl.full([BLOCK_H], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, HEAD_DIM], dtype=tl.float32)

    for n_start in range(n_lo, n_hi, BLOCK_N):
        cur_n = n_start + offs_n
        n_in_bounds = (cur_n < n_hi) & (cur_n >= n_lo)

        k_ptrs = K_bh + cur_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V_bh + cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=n_in_bounds[:, None], other=0.0)   # (BLOCK_N, HEAD_DIM)
        v = tl.load(v_ptrs, mask=n_in_bounds[:, None], other=0.0)

        # Q @ K^T using tensor cores: (BLOCK_H, HEAD_DIM) @ (HEAD_DIM, BLOCK_N)
        s = tl.dot(q, tl.trans(k)) * sm_scale

        if BIAS_KIND == 1:
            dist = tl.abs((N_KV - 1) - cur_n)
            s = s + (-slope[:, None].to(tl.float32) *
                     dist[None, :].to(tl.float32))

        s = tl.where(n_in_bounds[None, :], s, -float("inf"))

        # Online softmax update (per row)
        block_max = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, block_max)
        is_all_masked = (block_max == float("-inf"))
        alpha = tl.where(is_all_masked, 1.0, tl.exp(m_i - m_new))
        safe_m_new = tl.where(is_all_masked, tl.zeros_like(m_new), m_new)
        p = tl.exp(s - safe_m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        # P @ V using tensor cores: (BLOCK_H, BLOCK_N) @ (BLOCK_N, HEAD_DIM)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
        m_i = m_new

    # Write partial (m, l, acc) but only for is_unique rows -- duplicates
    # produced by the GQA-replication trick are silently dropped here.
    wm_ptr   = Wm_ptr  + split_id * stride_wm_s  + b * stride_wm_b  + offs_h * stride_wm_h
    wl_ptr   = Wl_ptr  + split_id * stride_wl_s  + b * stride_wl_b  + offs_h * stride_wl_h
    wacc_ptr = (Wacc_ptr + split_id * stride_wacc_s + b * stride_wacc_b
                + offs_h[:, None] * stride_wacc_h + offs_d[None, :] * stride_wacc_d)
    tl.store(wm_ptr,   m_i, mask=is_unique)
    tl.store(wl_ptr,   l_i, mask=is_unique)
    tl.store(wacc_ptr, acc, mask=is_unique[:, None])


@triton.jit
def attnfuse_decode_combine_kernel(
    Wm_ptr, Wl_ptr, Wacc_ptr,
    O_ptr,
    stride_wm_s, stride_wm_b, stride_wm_h,
    stride_wl_s, stride_wl_b, stride_wl_h,
    stride_wacc_s, stride_wacc_b, stride_wacc_h, stride_wacc_d,
    stride_ob, stride_oh, stride_od,
    B, H,
    HEAD_DIM:  tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """Combine NUM_SPLITS partial (m, l, acc) tuples for one (b, h) into one output row."""
    pid_bh = tl.program_id(0)
    b = pid_bh // H
    h = pid_bh %  H

    offs_d = tl.arange(0, HEAD_DIM)
    offs_s = tl.arange(0, NUM_SPLITS)

    mp = Wm_ptr + offs_s * stride_wm_s + b * stride_wm_b + h * stride_wm_h
    lp = Wl_ptr + offs_s * stride_wl_s + b * stride_wl_b + h * stride_wl_h
    m_all = tl.load(mp)                                  # (NUM_SPLITS,)
    l_all = tl.load(lp)

    m_final = tl.max(m_all, axis=0)
    alpha   = tl.exp(m_all - m_final)
    l_final = tl.sum(alpha * l_all, axis=0)

    wacc_p = (Wacc_ptr
              + offs_s[:, None] * stride_wacc_s
              + b * stride_wacc_b
              + h * stride_wacc_h
              + offs_d[None, :] * stride_wacc_d)
    acc_all = tl.load(wacc_p)                            # (NUM_SPLITS, HEAD_DIM)
    weighted = acc_all * alpha[:, None]
    acc_final = tl.sum(weighted, axis=0) / l_final

    o_ptr = O_ptr + b * stride_ob + h * stride_oh + offs_d * stride_od
    tl.store(o_ptr, acc_final.to(O_ptr.dtype.element_ty))
'''


def get_decode_source() -> str:
    """Return the Triton source string for the two Flash Decoding kernels."""
    return _DECODE_KERNEL_SRC
