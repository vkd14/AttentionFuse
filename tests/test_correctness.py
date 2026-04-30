"""GPU correctness tests: AttnFuse output must match PyTorch reference.

Tolerances:
    fp16/bf16   → atol=2e-2 (softmax accumulates a lot of small errors)
    fp32        → atol=2e-3 (online-softmax vs standard softmax rounding)
"""
import math
import pytest
import torch

import attnfuse as af
from attnfuse.reference import naive_attention
from attnfuse.runtime.dispatch import _alibi_slopes


pytestmark = pytest.mark.gpu


def _inputs(B=1, H=4, N=128, D=64, dtype=torch.float16):
    g = torch.Generator(device="cuda").manual_seed(0)
    return tuple(
        torch.randn(B, H, N, D, generator=g, device="cuda", dtype=dtype)
        for _ in range(3)
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_dense(dtype):
    @af.attention
    def fn(Q, K, V):
        return af.softmax(af.scaled_dot_product(Q, K)) @ V

    Q, K, V = _inputs(dtype=dtype)
    got = fn(Q, K, V)
    want = naive_attention(Q, K, V)
    atol = 2e-3 if dtype is torch.float32 else 2e-2
    assert torch.allclose(got, want, atol=atol, rtol=atol)


def test_causal():
    @af.attention
    def fn(Q, K, V):
        return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V

    Q, K, V = _inputs()
    got  = fn(Q, K, V)
    want = naive_attention(Q, K, V, causal=True)
    assert torch.allclose(got, want, atol=2e-2, rtol=2e-2)


def test_sliding_window():
    @af.attention
    def fn(Q, K, V):
        s = af.scaled_dot_product(Q, K)
        s = af.sliding_window(s, window_size=32)
        return af.softmax(s) @ V

    Q, K, V = _inputs(N=128)
    got  = fn(Q, K, V)
    want = naive_attention(Q, K, V, window=32)
    assert torch.allclose(got, want, atol=2e-2, rtol=2e-2)


def test_causal_alibi():
    H = 4
    @af.attention
    def fn(Q, K, V):
        s = af.scaled_dot_product(Q, K)
        s = af.alibi(s, num_heads=H)
        s = af.causal(s)
        return af.softmax(s) @ V

    Q, K, V = _inputs(H=H)
    slopes = _alibi_slopes(H, "cuda", "float16")
    got  = fn(Q, K, V)
    want = naive_attention(Q, K, V, causal=True, alibi_slopes=slopes)
    assert torch.allclose(got, want, atol=2e-2, rtol=2e-2)
