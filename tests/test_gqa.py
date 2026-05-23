"""GQA / MQA correctness tests.

Modern LLMs (Llama 3, Mistral, Mixtral, Falcon) use Grouped-Query Attention:
the query tensor has H_q heads but K, V have only H_kv heads, with each
KV head shared by H_q // H_kv query heads. The extreme case H_kv = 1 is
Multi-Query Attention (MQA).

These tests verify that AttnFuse's GQA path matches the naive PyTorch
reference where K, V are expanded to match Q's head count before
attention.

Coverage:
  * MHA (H_kv = H_q): default behaviour, group_size = 1
  * GQA (H_q = 4 * H_kv): Llama-3-70B-style ratio
  * GQA (H_q = 8 * H_kv): Llama-3-8B-style ratio
  * MQA (H_kv = 1): one shared KV head across all query heads
Plus all five attention variants (dense, causal, SW, ALiBi, RoPE+causal).
"""
from __future__ import annotations

import pytest
import torch

import attnfuse as af
from attnfuse.reference.pytorch_naive import naive_attention
from attnfuse.runtime.dispatch import _alibi_slopes
from attnfuse.rope_utils import build_rope_cache, apply_rope


pytestmark = pytest.mark.gpu
TOL = 2e-2  # FA2 documented fp16 tolerance


def _expand_kv(t: torch.Tensor, group_size: int) -> torch.Tensor:
    """Repeat each KV head ``group_size`` times along the head axis."""
    B, H_kv, N, D = t.shape
    return (t.unsqueeze(2)
             .expand(B, H_kv, group_size, N, D)
             .reshape(B, H_kv * group_size, N, D)
             .contiguous())


# Parametrize over (H_q, H_kv) pairs
@pytest.fixture(params=[(8, 8), (8, 2), (8, 1), (12, 4), (12, 1)],
                ids=lambda p: f"H_q={p[0]}-H_kv={p[1]}")
def head_config(request):
    return request.param


def _make_qkv(B, H_q, H_kv, N, D, dtype=torch.float16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    Q = torch.randn(B, H_q,  N, D, generator=g, device="cuda", dtype=dtype)
    K = torch.randn(B, H_kv, N, D, generator=g, device="cuda", dtype=dtype)
    V = torch.randn(B, H_kv, N, D, generator=g, device="cuda", dtype=dtype)
    return Q, K, V


def test_gqa_dense(head_config):
    H_q, H_kv = head_config
    B, N, D = 2, 256, 64
    Q, K, V = _make_qkv(B, H_q, H_kv, N, D)
    group_size = H_q // H_kv

    @af.attention
    def dense(Q, K, V):
        return af.softmax(af.scaled_dot_product(Q, K)) @ V

    out = dense(Q, K, V)
    ref = naive_attention(Q, _expand_kv(K, group_size), _expand_kv(V, group_size))
    err = (out.float() - ref.float()).abs().max().item()
    assert err < TOL, f"H_q={H_q} H_kv={H_kv}: max|err|={err:.3e}"


def test_gqa_causal(head_config):
    H_q, H_kv = head_config
    B, N, D = 2, 256, 64
    Q, K, V = _make_qkv(B, H_q, H_kv, N, D)
    group_size = H_q // H_kv

    @af.attention
    def causal(Q, K, V):
        return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V

    out = causal(Q, K, V)
    ref = naive_attention(Q, _expand_kv(K, group_size), _expand_kv(V, group_size),
                          causal=True)
    err = (out.float() - ref.float()).abs().max().item()
    assert err < TOL, f"H_q={H_q} H_kv={H_kv}: max|err|={err:.3e}"


def test_gqa_sliding_window(head_config):
    H_q, H_kv = head_config
    B, N, D = 2, 256, 64
    W = 64
    Q, K, V = _make_qkv(B, H_q, H_kv, N, D)
    group_size = H_q // H_kv

    @af.attention
    def sw(Q, K, V):
        return af.softmax(af.sliding_window(af.scaled_dot_product(Q, K), W)) @ V

    out = sw(Q, K, V)
    ref = naive_attention(Q, _expand_kv(K, group_size), _expand_kv(V, group_size),
                          window=W)
    err = (out.float() - ref.float()).abs().max().item()
    assert err < TOL, f"H_q={H_q} H_kv={H_kv}: max|err|={err:.3e}"


def test_gqa_causal_alibi(head_config):
    H_q, H_kv = head_config
    B, N, D = 2, 256, 64
    Q, K, V = _make_qkv(B, H_q, H_kv, N, D)
    group_size = H_q // H_kv

    @af.attention
    def causal_alibi(Q, K, V):
        s = af.scaled_dot_product(Q, K)
        s = af.alibi(s, num_heads=H_q)
        s = af.causal(s)
        return af.softmax(s) @ V

    out = causal_alibi(Q, K, V)
    slopes = _alibi_slopes(H_q, "cuda", "float16")
    ref = naive_attention(Q, _expand_kv(K, group_size), _expand_kv(V, group_size),
                          causal=True, alibi_slopes=slopes)
    err = (out.float() - ref.float()).abs().max().item()
    assert err < TOL, f"H_q={H_q} H_kv={H_kv}: max|err|={err:.3e}"


def test_gqa_causal_rope(head_config):
    H_q, H_kv = head_config
    B, N, D = 2, 256, 64
    Q, K, V = _make_qkv(B, H_q, H_kv, N, D)
    group_size = H_q // H_kv
    cos, sin = build_rope_cache(N, D, device="cuda", dtype=torch.float16)

    @af.attention
    def causal_rope(Q, K, V):
        s = af.rope(Q, K)
        s = af.causal(s)
        return af.softmax(s) @ V

    out = causal_rope(Q, K, V, cos=cos, sin=sin)
    # Reference: rotate Q and K (in their original head dims), expand, attend
    Q_rot = apply_rope(Q, cos, sin)
    K_rot = apply_rope(K, cos, sin)
    ref = naive_attention(Q_rot, _expand_kv(K_rot, group_size),
                          _expand_kv(V, group_size), causal=True)
    err = (out.float() - ref.float()).abs().max().item()
    # RoPE adds fp16 rounding noise; loosen slightly
    assert err < TOL * 1.5, f"H_q={H_q} H_kv={H_kv}: max|err|={err:.3e}"
