"""30-second smoke test: import, trace, lower, dispatch one tiny kernel."""
import pytest
import torch

import attnfuse as af


def test_import():
    assert af.__version__


def test_trace_dense_cpu():
    """Tracing must work without CUDA/Triton."""
    @af.attention
    def fn(Q, K, V):
        return af.softmax(af.scaled_dot_product(Q, K)) @ V

    Q = torch.zeros(1, 2, 16, 32)
    K = torch.zeros(1, 2, 16, 32)
    V = torch.zeros(1, 2, 16, 32)
    g = fn(Q, K, V, return_graph=True)
    assert g.q.head_dim == 32
    assert len(g.signature()) == 16


def test_trace_causal_alibi_cpu():
    @af.attention
    def fn(Q, K, V):
        s = af.scaled_dot_product(Q, K)
        s = af.alibi(s, num_heads=2)
        s = af.causal(s)
        return af.softmax(s) @ V

    Q = torch.zeros(1, 2, 16, 32); K = Q.clone(); V = Q.clone()
    g = fn(Q, K, V, return_graph=True)
    assert len(g.collect_masks()) == 1
    assert len(g.collect_biases()) == 1


@pytest.mark.gpu
def test_dispatch_dense_gpu():
    @af.attention
    def fn(Q, K, V):
        return af.softmax(af.scaled_dot_product(Q, K)) @ V

    Q = torch.randn(1, 2, 64, 32, device="cuda", dtype=torch.float16)
    K = torch.randn_like(Q); V = torch.randn_like(Q)
    out = fn(Q, K, V)
    assert out.shape == Q.shape
    assert torch.isfinite(out).all()
