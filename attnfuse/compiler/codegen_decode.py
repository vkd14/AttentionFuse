"""Codegen for Flash Decoding (split-K attention).

The standard attention kernel launches one program per (batch, query-head,
m_block). For autoregressive decoding (Q.N=1) this gives only B*H_q programs,
typically 32 for an Llama-3-8B-style model. The RTX 3090 has 82 SMs, so
roughly half are idle and the per-program inner loop scales linearly with
cache length.

Flash Decoding (Dao 2023) fixes this by splitting the KV axis across many
parallel programs and combining their partial results via online-softmax
merging. With NUM_SPLITS programs per (b, h), total programs scale as
NUM_SPLITS * B * H_q -- enough to saturate all SMs.

Two kernels live here:

  attnfuse_decode_split_kernel
    Grid: (NUM_SPLITS, B * H_q)
    Each program processes a contiguous chunk of K, V of length
    ceil(N_kv / NUM_SPLITS). Outputs partial m, l, and acc per chunk.

  attnfuse_decode_combine_kernel
    Grid: (B * H_q,)
    For each (b, h) reads all NUM_SPLITS partial outputs and merges
    them using the log-sum-exp trick (same algebra as the online-softmax
    accumulator update, applied across split boundaries).

Restrictions of this initial Flash Decoding path (covers the production
use case; other variants stay on the original kernel):
  * Dense / no mask (MASK_KIND = 0)
  * No additive bias (BIAS_KIND in {0, 1})  -- ALiBi works fine
  * No RoPE (ROPE_KIND = 0)                 -- TODO: fused-RoPE decode
  * Softmax norm only (NORM_KIND = 0)
  * Q.N == 1 (the canonical decode shape)
"""

_DECODE_KERNEL_SRC = '''
import triton
import triton.language as tl

@triton.jit
def attnfuse_decode_split_kernel(
    Q_ptr, K_ptr, V_ptr,
    # Workspace outputs: partial m (max), l (sum-exp), acc (output);
    # shape (NUM_SPLITS, B, H_q, HEAD_DIM) for acc, (NUM_SPLITS, B, H_q) for m/l
    Wm_ptr, Wl_ptr, Wacc_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_wm_s, stride_wm_b, stride_wm_h,
    stride_wl_s, stride_wl_b, stride_wl_h,
    stride_wacc_s, stride_wacc_b, stride_wacc_h, stride_wacc_d,
    B, H, N_KV, GROUP_SIZE,
    alibi_slopes_ptr,
    BIAS_KIND: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N:  tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    """One program per (split_id, batch * query_head). Processes one KV chunk.

    Each program covers cur_n in [split_id * split_len, (split_id+1) * split_len),
    where split_len = cdiv(N_KV, NUM_SPLITS), and writes its partial m, l, acc
    to the workspace.
    """
    split_id = tl.program_id(0)
    pid_bh   = tl.program_id(1)
    b    = pid_bh // H
    h    = pid_bh %  H
    h_kv = h // GROUP_SIZE

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # Load Q (single query row -- Q.N = 1 is the decode shape)
    q_ptrs = Q_ptr + b * stride_qb + h * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptrs)                                    # shape (HEAD_DIM,)

    # Split bounds along the KV axis
    split_len = (N_KV + NUM_SPLITS - 1) // NUM_SPLITS
    n_lo = split_id * split_len
    n_hi = tl.minimum(n_lo + split_len, N_KV)
    n_lo = (n_lo // BLOCK_N) * BLOCK_N                     # BLOCK_N-align for masked loads

    K_bh = K_ptr + b * stride_kb + h_kv * stride_kh
    V_bh = V_ptr + b * stride_vb + h_kv * stride_vh

    if BIAS_KIND == 1:
        slope = tl.load(alibi_slopes_ptr + h)

    # Online-softmax accumulators (scalars for the single-query case)
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for n_start in range(n_lo, n_hi, BLOCK_N):
        cur_n = n_start + offs_n
        n_in_bounds = (cur_n < n_hi) & (cur_n >= n_lo)     # within OUR split AND within N_KV

        k_ptrs = K_bh + cur_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V_bh + cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=n_in_bounds[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=n_in_bounds[:, None], other=0.0)

        # s = q . k   ->  (BLOCK_N,)
        s = tl.sum(q[None, :].to(tl.float32) * k.to(tl.float32), axis=1) * sm_scale

        if BIAS_KIND == 1:
            # ALiBi for a single query at the trailing position (N_KV - 1):
            # dist = (N_KV - 1) - cur_n   (always >= 0 in the typical decode case)
            dist = tl.abs((N_KV - 1) - cur_n)
            s = s + (-slope.to(tl.float32) * dist.to(tl.float32))

        s = tl.where(n_in_bounds, s, -float("inf"))

        # Online softmax update
        block_max = tl.max(s, axis=0)
        m_new = tl.maximum(m_i, block_max)
        # If THIS split has no in-bounds keys, block_max = -inf, so we leave state untouched
        is_all_masked = (block_max == float("-inf"))
        alpha = tl.where(is_all_masked, 1.0, tl.exp(m_i - m_new))
        safe_m_new = tl.where(is_all_masked, 0.0, m_new)
        p = tl.exp(s - safe_m_new)
        l_i = alpha * l_i + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v.to(tl.float32), axis=0)
        m_i = m_new

    # Write partial m, l, acc to workspace
    wm_ptr  = Wm_ptr  + split_id * stride_wm_s  + b * stride_wm_b  + h * stride_wm_h
    wl_ptr  = Wl_ptr  + split_id * stride_wl_s  + b * stride_wl_b  + h * stride_wl_h
    wacc_p  = Wacc_ptr + split_id * stride_wacc_s + b * stride_wacc_b + h * stride_wacc_h \\
              + offs_d * stride_wacc_d
    tl.store(wm_ptr, m_i)
    tl.store(wl_ptr, l_i)
    tl.store(wacc_p, acc)


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

    # Load all splits' m and l
    mp = Wm_ptr + offs_s * stride_wm_s + b * stride_wm_b + h * stride_wm_h
    lp = Wl_ptr + offs_s * stride_wl_s + b * stride_wl_b + h * stride_wl_h
    m_all = tl.load(mp)                                    # (NUM_SPLITS,)
    l_all = tl.load(lp)                                    # (NUM_SPLITS,)

    # Reduction: m_final = max(m_all); alpha_i = exp(m_i - m_final);
    # l_final = sum(alpha_i * l_i); acc_final = sum(alpha_i * acc_i) / l_final
    m_final = tl.max(m_all, axis=0)
    alpha   = tl.exp(m_all - m_final)
    l_final = tl.sum(alpha * l_all, axis=0)

    # Load and combine acc
    # wacc[s, b, h, d] = Wacc_ptr + s*stride_s + b*stride_b + h*stride_h + d*stride_d
    wacc_p = (Wacc_ptr
              + offs_s[:, None] * stride_wacc_s
              + b * stride_wacc_b
              + h * stride_wacc_h
              + offs_d[None, :] * stride_wacc_d)
    acc_all = tl.load(wacc_p)                              # (NUM_SPLITS, HEAD_DIM)
    weighted = acc_all * alpha[:, None]
    acc_final = tl.sum(weighted, axis=0) / l_final

    # Write final output
    o_ptr = O_ptr + b * stride_ob + h * stride_oh + offs_d * stride_od
    tl.store(o_ptr, acc_final.to(O_ptr.dtype.element_ty))
'''


def get_decode_source() -> str:
    """Return the Triton source string for the two Flash Decoding kernels."""
    return _DECODE_KERNEL_SRC
