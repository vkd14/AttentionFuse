"""Cross-attention / KV-cache decoding tests.

The dispatch layer used to require ``Q.shape == K.shape``, which made
autoregressive decoding impossible: at inference time each generation
step has a single new token (``N_q = 1``) attending to a large KV cache
(``N_kv = past_tokens + 1``). After this change, ``N_q != N_kv`` is
permitted for the dense (no-mask) variant -- the canonical KV-cache
configuration.

Coverage:
  * Encoder-decoder style: N_q = 64, N_kv = 256 (T5 / BART pattern)
  * Single-token decode:   N_q = 1,  N_kv = 4096 (LLM inference)
  * Multi-token chunked:   N_q = 32, N_kv = 1024 (prefill + N steps)
  * Combined with GQA:     Llama-3-8B (H_q=32, H_kv=8) + KV-cache
"""
from __future__ import annotations

import pytest
import torch

import attnfuse as af
from attnfuse.reference.pytorch_naive import naive_attention


pytestmark = pytest.mark.gpu
TOL = 2e-2


def _expand_kv(t: torch.Tensor, group_size: int) -> torch.Tensor:
    B, H_kv, N, D = t.shape
    return (t.unsqueeze(2)
             .expand(B, H_kv, group_size, N, D)
             .reshape(B, H_kv * group_size, N, D)
             .contiguous())


@af.attention
def cross_dense(Q, K, V):
    return af.softmax(af.scaled_dot_product(Q, K)) @ V


def test_cross_encoder_decoder():
    """Encoder-decoder cross-attention: decoder attends to encoder."""
    B, H, N_q, N_kv, D = 2, 8, 64, 256, 64
    Q = torch.randn(B, H, N_q,  D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H, N_kv, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, N_kv, D, device="cuda", dtype=torch.float16)
    out = cross_dense(Q, K, V)
    ref = naive_attention(Q, K, V)
    err = (out.float() - ref.float()).abs().max().item()
    assert err < TOL, f"cross-attn N_q={N_q} N_kv={N_kv}: max|err|={err:.3e}"


def test_kv_cache_single_token():
    """LLM inference: one new token attends to the full KV cache."""
    B, H, D = 2, 12, 64
    N_kv = 4096   # large KV cache
    Q = torch.randn(B, H, 1,    D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H, N_kv, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, N_kv, D, device="cuda", dtype=torch.float16)
    out = cross_dense(Q, K, V)
    ref = naive_attention(Q, K, V)
    err = (out.float() - ref.float()).abs().max().item()
    assert err < TOL, f"KV-cache decode N_kv={N_kv}: max|err|={err:.3e}"


def test_kv_cache_chunked():
    """Prefill + chunked decoding pattern: 32 new tokens, 1024-cache."""
    B, H, D = 1, 12, 64
    Q = torch.randn(B, H,   32, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H, 1024, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H, 1024, D, device="cuda", dtype=torch.float16)
    out = cross_dense(Q, K, V)
    ref = naive_attention(Q, K, V)
    err = (out.float() - ref.float()).abs().max().item()
    assert err < TOL, f"chunked: max|err|={err:.3e}"


def test_kv_cache_with_gqa():
    """Llama-3-8B-style decoding: GQA + KV cache."""
    B, D = 1, 128
    H_q, H_kv = 32, 8       # group_size = 4
    Q = torch.randn(B, H_q,    1, D, device="cuda", dtype=torch.float16)
    K = torch.randn(B, H_kv, 2048, D, device="cuda", dtype=torch.float16)
    V = torch.randn(B, H_kv, 2048, D, device="cuda", dtype=torch.float16)
    out = cross_dense(Q, K, V)
    # Reference: expand KV to match Q heads, then dense attention
    ref = naive_attention(Q, _expand_kv(K, H_q // H_kv), _expand_kv(V, H_q // H_kv))
    err = (out.float() - ref.float()).abs().max().item()
    assert err < TOL, f"GQA + KV-cache: max|err|={err:.3e}"


def test_cross_attention_rejects_causal_when_mismatched_seqlens():
    """Causal cross-attention with N_q != N_kv should raise (semantics ambiguous)."""
    Q = torch.randn(1, 4,  16, 64, device="cuda", dtype=torch.float16)
    K = torch.randn(1, 4, 256, 64, device="cuda", dtype=torch.float16)
    V = torch.randn(1, 4, 256, 64, device="cuda", dtype=torch.float16)

    @af.attention
    def causal_cross(Q, K, V):
        return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V

    with pytest.raises(ValueError, match="Cross-attention"):
        causal_cross(Q, K, V)
