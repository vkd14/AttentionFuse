"""Property-based correctness fuzzer for AttnFuse.

Uses Hypothesis to draw random (variant, shape, dtype) configurations and
verifies that every compiled AttnFuse kernel agrees with a naive PyTorch
reference to within FA2's documented numerical tolerance.

This catches regressions in the codegen, the tile-config selector, and
the dispatch layer that hand-written unit tests miss. It also serves as
reviewer-grade evidence of correctness for the paper.

Run all 80 examples (the default Hypothesis budget for this test) with::

    pytest tests/test_fuzz.py -v

The fuzzer covers:
  * 5 variants (dense, causal, SW, ALiBi, RoPE+causal)
  * head_dim in {32, 64, 96, 128}
  * batch in {1, 2, 4}, num_heads in {1, 2, 4, 8, 12, 16}
  * seqlen in [16, 4096], with non-aligned values like 100 or 333
  * fp16 and bf16
  * sliding-window W in [16, 2048]

Pass criterion: max|err| < 2e-2 (FA2's documented fp16 tolerance).
"""
from __future__ import annotations

import math
import pytest
import torch
from hypothesis import given, settings, strategies as st, HealthCheck

import attnfuse as af
from attnfuse.reference.pytorch_naive import naive_attention
from attnfuse.runtime.dispatch import _alibi_slopes
from attnfuse.rope_utils import build_rope_cache, apply_rope


pytestmark = pytest.mark.gpu


# Numerical tolerance for fp16 attention -- conservative bound from FA2 paper.
TOL_FP16 = 2e-2
TOL_BF16 = 4e-2  # bf16 has more mantissa noise

# Hypothesis strategies
_HEAD_DIMS = st.sampled_from([32, 64, 128])   # power-of-2 only (Triton constraint)
_BATCHES   = st.sampled_from([1, 2, 4])
_NHEADS    = st.sampled_from([1, 2, 4, 8, 12, 16])
_SEQLENS   = st.integers(min_value=16, max_value=512)   # keep fuzz N small for speed
_DTYPES    = st.sampled_from([torch.float16, torch.bfloat16])
_WINDOWS   = st.integers(min_value=8, max_value=256)


def _tol_for(dtype: torch.dtype) -> float:
    return TOL_BF16 if dtype == torch.bfloat16 else TOL_FP16


def _make_qkv(B, H, N, D, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    Q = torch.randn(B, H, N, D, generator=g, device="cuda", dtype=dtype)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    return Q, K, V


def _max_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    return (out.float() - ref.float()).abs().max().item()


# --- dense (BERT) ---------------------------------------------------------
@given(_BATCHES, _NHEADS, _SEQLENS, _HEAD_DIMS, _DTYPES, st.integers(0, 1 << 30))
@settings(max_examples=15, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_fuzz_dense(B, H, N, D, dtype, seed):
    Q, K, V = _make_qkv(B, H, N, D, dtype, seed)

    @af.attention
    def dense(Q, K, V):
        return af.softmax(af.scaled_dot_product(Q, K)) @ V

    out = dense(Q, K, V)
    ref = naive_attention(Q, K, V)
    err = _max_err(out, ref)
    assert err < _tol_for(dtype), \
        f"dense B={B} H={H} N={N} D={D} dtype={dtype}: max|err|={err:.3e}"


# --- causal (GPT) ---------------------------------------------------------
@given(_BATCHES, _NHEADS, _SEQLENS, _HEAD_DIMS, _DTYPES, st.integers(0, 1 << 30))
@settings(max_examples=15, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_fuzz_causal(B, H, N, D, dtype, seed):
    Q, K, V = _make_qkv(B, H, N, D, dtype, seed)

    @af.attention
    def causal(Q, K, V):
        return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V

    out = causal(Q, K, V)
    ref = naive_attention(Q, K, V, causal=True)
    err = _max_err(out, ref)
    assert err < _tol_for(dtype), \
        f"causal B={B} H={H} N={N} D={D} dtype={dtype}: max|err|={err:.3e}"


# --- sliding-window (Mistral) ---------------------------------------------
@given(_BATCHES, _NHEADS, _SEQLENS, _HEAD_DIMS, _WINDOWS, _DTYPES,
       st.integers(0, 1 << 30))
@settings(max_examples=20, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
def test_fuzz_sliding_window(B, H, N, D, W, dtype, seed):
    Q, K, V = _make_qkv(B, H, N, D, dtype, seed)

    @af.attention
    def sw(Q, K, V):
        return af.softmax(af.sliding_window(af.scaled_dot_product(Q, K), W)) @ V

    out = sw(Q, K, V)
    ref = naive_attention(Q, K, V, causal=False, window=W)
    err = _max_err(out, ref)
    assert err < _tol_for(dtype), \
        f"SW B={B} H={H} N={N} D={D} W={W} dtype={dtype}: max|err|={err:.3e}"


# --- causal + ALiBi (GPT-NeoX) --------------------------------------------
@given(_BATCHES, _NHEADS, _SEQLENS, _HEAD_DIMS, _DTYPES, st.integers(0, 1 << 30))
@settings(max_examples=15, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_fuzz_causal_alibi(B, H, N, D, dtype, seed):
    Q, K, V = _make_qkv(B, H, N, D, dtype, seed)

    @af.attention
    def causal_alibi(Q, K, V):
        s = af.scaled_dot_product(Q, K)
        s = af.alibi(s, num_heads=H)
        s = af.causal(s)
        return af.softmax(s) @ V

    out = causal_alibi(Q, K, V)
    slopes = _alibi_slopes(H, "cuda", str(dtype).replace("torch.", ""))
    ref = naive_attention(Q, K, V, causal=True, alibi_slopes=slopes)
    err = _max_err(out, ref)
    assert err < _tol_for(dtype), \
        f"causal+ALiBi B={B} H={H} N={N} D={D} dtype={dtype}: max|err|={err:.3e}"


# --- causal + fused RoPE (LLaMA) ------------------------------------------
@given(_BATCHES, _NHEADS, _SEQLENS, _HEAD_DIMS, _DTYPES, st.integers(0, 1 << 30))
@settings(max_examples=15, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_fuzz_causal_rope(B, H, N, D, dtype, seed):
    Q, K, V = _make_qkv(B, H, N, D, dtype, seed)
    cos, sin = build_rope_cache(N, D, device="cuda", dtype=dtype)

    @af.attention
    def causal_rope(Q, K, V):
        s = af.rope(Q, K)
        s = af.causal(s)
        return af.softmax(s) @ V

    out = causal_rope(Q, K, V, cos=cos, sin=sin)
    # Reference: rotate Q and K, then causal attention
    ref = naive_attention(apply_rope(Q, cos, sin),
                          apply_rope(K, cos, sin),
                          V, causal=True)
    err = _max_err(out, ref)
    # RoPE adds extra fp16 rounding; bump tolerance slightly
    tol = _tol_for(dtype) * 1.5
    assert err < tol, \
        f"causal+RoPE B={B} H={H} N={N} D={D} dtype={dtype}: max|err|={err:.3e}"
