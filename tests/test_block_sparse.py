"""Block-sparse mask correctness tests.

Covers the af.block_sparse() combinator and the create_block_mask
helper. Each test runs AttnFuse against a hand-computed PyTorch
reference that applies the same block-aligned mask element-wise.

Scope reminder: the v1 block-sparse path treats every tile that has
ANY active element as fully active (no element-level mask refinement).
Tests therefore use block-aligned mask patterns -- the standard
BigBird-style coarse mask -- not fine-grained per-element masks like
plain causal.
"""
from __future__ import annotations

import pytest
import torch

import attnfuse as af


pytestmark = pytest.mark.gpu
TOL = 2e-2

B, H, N, D = 2, 4, 256, 64
BLOCK_M = BLOCK_N = 64


@af.attention
def bs_attn(Q, K, V):
    s = af.scaled_dot_product(Q, K)
    s = af.block_sparse(s)
    return af.softmax(s) @ V


def _reference_with_elem_mask(Q, K, V, elem_mask):
    """Naive eager reference: mask the scores element-wise, softmax, then @ V."""
    S = torch.einsum("bhid,bhjd->bhij", Q.float(), K.float()) / (D ** 0.5)
    S = S.masked_fill(~elem_mask, float("-inf"))
    P = torch.softmax(S, dim=-1).to(Q.dtype)
    return P @ V


def _elem_mask_from(pred):
    """Materialise a 2D bool mask of shape (N, N) on the GPU."""
    qi = torch.arange(N)[:, None].expand(N, N)
    kj = torch.arange(N)[None, :].expand(N, N)
    return pred(qi, kj).to("cuda")


@pytest.fixture
def qkv():
    torch.manual_seed(0)
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    return Q, K, V


def test_block_sparse_all_active(qkv):
    """Every block active -> equivalent to dense attention."""
    Q, K, V = qkv

    def all_true(q, kv): return torch.ones_like(q, dtype=torch.bool)
    bm = af.create_block_mask(all_true, N, N, BLOCK_M, BLOCK_N)

    out_bs = bs_attn(Q, K, V, block_mask=bm)
    out_ref = _reference_with_elem_mask(Q, K, V, _elem_mask_from(all_true))
    err = (out_bs - out_ref).abs().max().item()
    assert err < TOL, f"all_true: max|err|={err:.3e}"
    assert bm.n_active == (N // BLOCK_M) ** 2


def test_block_sparse_block_local(qkv):
    """Block-aligned local window: |qb - kb| <= 1."""
    Q, K, V = qkv

    def aligned_local(q, kv):
        return ((q // BLOCK_M) - (kv // BLOCK_N)).abs() <= 1
    bm = af.create_block_mask(aligned_local, N, N, BLOCK_M, BLOCK_N)

    out_bs = bs_attn(Q, K, V, block_mask=bm)
    out_ref = _reference_with_elem_mask(Q, K, V, _elem_mask_from(aligned_local))
    err = (out_bs - out_ref).abs().max().item()
    assert err < TOL, f"block-local: max|err|={err:.3e}"
    # Tridiagonal active set: 3*n_blocks - 2 = 10 for n_blocks=4
    n_blocks = N // BLOCK_M
    assert bm.n_active == 3 * n_blocks - 2


def test_block_sparse_bigbird_pattern(qkv):
    """BigBird-style: global rows + global cols + local window + scattered."""
    Q, K, V = qkv

    def bigbird(q, kv):
        qb = q // BLOCK_M
        kb = kv // BLOCK_N
        return (qb == 0) | (kb == 0) | ((qb - kb).abs() <= 1)
    bm = af.create_block_mask(bigbird, N, N, BLOCK_M, BLOCK_N)

    out_bs = bs_attn(Q, K, V, block_mask=bm)
    out_ref = _reference_with_elem_mask(Q, K, V, _elem_mask_from(bigbird))
    err = (out_bs - out_ref).abs().max().item()
    assert err < TOL, f"bigbird: max|err|={err:.3e}"


def test_block_sparse_strided(qkv):
    """Strided pattern: every other block."""
    Q, K, V = qkv

    def strided(q, kv):
        qb = q // BLOCK_M
        kb = kv // BLOCK_N
        return (qb + kb) % 2 == 0
    bm = af.create_block_mask(strided, N, N, BLOCK_M, BLOCK_N)

    out_bs = bs_attn(Q, K, V, block_mask=bm)
    out_ref = _reference_with_elem_mask(Q, K, V, _elem_mask_from(strided))
    err = (out_bs - out_ref).abs().max().item()
    assert err < TOL, f"strided: max|err|={err:.3e}"
    # Half the grid is active: 8 of 16
    n_blocks = N // BLOCK_M
    assert bm.n_active == (n_blocks * n_blocks) // 2 + (n_blocks % 2)


def test_block_sparse_missing_mask_raises(qkv):
    """Forgetting the block_mask kwarg should give a clear error."""
    Q, K, V = qkv
    with pytest.raises(RuntimeError, match="block_mask"):
        bs_attn(Q, K, V)
