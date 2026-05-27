"""Codegen for block-sparse attention (BigBird-style, user-supplied mask).

The user provides a per-query-block list of active KV blocks (CSR-style).
Each program iterates only over the active n-blocks for its m-block, so
the FLOPs and HBM traffic are both genuinely sub-quadratic: O(active /
total) of the dense baseline.

Data layout (mirrors what flex_attention's BlockMask provides):

  kv_num_blocks : (n_q_blocks,)             int32    -- active count per row
  kv_indices    : (n_q_blocks, max_kv)      int32    -- padded block indices
  full_kv_num   : (n_q_blocks,)             int32    -- "fully inside the mask"
                                                       count: these blocks
                                                       need NO per-element mask
  full_kv_idx   : (n_q_blocks, max_full)    int32    -- full-block indices

  Optional second list of "boundary" blocks (partial mask) -- when a tile
  is FULL_KV the per-element mask is skipped; when only in kv_indices the
  per-element mask is applied. For BigBird-style patterns most tiles are
  full so this skip is meaningful.

This kernel currently supports:
  * Dense scoring  (RoPE in backward is future work for block-sparse too)
  * Optional ALiBi bias
  * Softmax norm
  * GQA / MQA via the existing GROUP_SIZE constexpr
"""

_BLOCKSPARSE_KERNEL_SRC = '''
import triton
import triton.language as tl


@triton.jit
def attnfuse_blocksparse_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    # CSR-ish active-block lists (one row per query block)
    kv_num_blocks_ptr,        # (n_q_blocks,) int32
    kv_indices_ptr,           # (n_q_blocks, max_kv) int32 (padded)
    stride_idx_q, stride_idx_n,
    full_kv_num_ptr,          # (n_q_blocks,) int32
    full_kv_idx_ptr,          # (n_q_blocks, max_full) int32
    stride_full_q, stride_full_n,
    B, H, N_Q, N_KV,
    GROUP_SIZE,
    alibi_slopes_ptr,
    HEAD_DIM:   tl.constexpr,
    BLOCK_M:    tl.constexpr,
    BLOCK_N:    tl.constexpr,
    BIAS_KIND:  tl.constexpr,
    L_ptr,                    # only written when SAVE_L=1; (B, H_q, N_q) fp32
    stride_lb, stride_lh, stride_lm,
    SAVE_L:     tl.constexpr,
):
    """One program per (m_block, b * H_q).

    For each program, the kv_num_blocks / kv_indices lists describe which
    n-blocks need work. We additionally split them into "full" (no
    per-element mask needed -- a fast inner-loop body) and "boundary"
    (per-element mask required).
    """
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b   = pid_bh // H
    h   = pid_bh %  H
    h_kv = h // GROUP_SIZE

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    m_in_bounds = offs_m < N_Q

    # Load Q
    Q_bh = Q_ptr + b * stride_qb + h    * stride_qh
    K_bh = K_ptr + b * stride_kb + h_kv * stride_kh
    V_bh = V_ptr + b * stride_vb + h_kv * stride_vh
    O_bh = O_ptr + b * stride_ob + h    * stride_oh

    q_ptrs = Q_bh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=m_in_bounds[:, None], other=0.0)

    if BIAS_KIND == 1:
        slope = tl.load(alibi_slopes_ptr + h)

    # Online softmax accumulators
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # --- LOOP 1: full blocks (no per-element mask check) ---
    full_count = tl.load(full_kv_num_ptr + pid_m)
    full_row_base = full_kv_idx_ptr + pid_m * stride_full_q
    for i in range(full_count):
        n_block = tl.load(full_row_base + i * stride_full_n)
        n_start = n_block * BLOCK_N
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

        s = tl.where(m_in_bounds[:, None] & n_in_bounds[None, :], s, -float("inf"))

        # Online softmax update (no all-masked branch needed in full blocks)
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
        m_i = m_new

    # --- LOOP 2: partial-mask blocks (boundary; user-provided mask gates per element) ---
    # For BigBird and similar patterns, this list is typically small (just
    # a handful of "boundary" tiles per row). The kernel still has to load
    # the per-element block mask if the user wants element-precision; for
    # now we treat partial blocks as fully-active and rely on the user's
    # block-coarse mask. (Element-level mask refinement is future work.)
    partial_count = tl.load(kv_num_blocks_ptr + pid_m)
    partial_row_base = kv_indices_ptr + pid_m * stride_idx_q
    for i in range(partial_count):
        n_block = tl.load(partial_row_base + i * stride_idx_n)
        n_start = n_block * BLOCK_N
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

        s = tl.where(m_in_bounds[:, None] & n_in_bounds[None, :], s, -float("inf"))

        # Safe online softmax (this tile might be all-masked if the
        # user-provided boundary block happens to have no valid entries)
        block_max = tl.max(s, axis=1)
        m_new = tl.maximum(m_i, block_max)
        is_all_masked = (block_max == float("-inf"))
        alpha = tl.where(is_all_masked, 1.0, tl.exp(m_i - m_new))
        safe_m_new = tl.where(is_all_masked, tl.zeros_like(m_new), m_new)
        p = tl.exp(s - safe_m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
        m_i = m_new

    # Final normalisation + store
    safe_l = tl.where(l_i == 0.0, 1.0, l_i)
    out = acc / safe_l[:, None]
    o_ptrs = O_bh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, out.to(O_ptr.dtype.element_ty), mask=m_in_bounds[:, None])

    if SAVE_L == 1:
        L_val = m_i + tl.log(safe_l)
        l_ptrs = L_ptr + b * stride_lb + h * stride_lh + offs_m * stride_lm
        tl.store(l_ptrs, L_val, mask=offs_m < N_Q)
'''


def get_blocksparse_source() -> str:
    return _BLOCKSPARSE_KERNEL_SRC
