"""DSL surface tests: combinators, tracer, error messages."""
import pytest
import torch

import attnfuse as af
from attnfuse.ir.high_level import MaskKind, BiasKind, NormKind


def _zeros(D=32, N=16):
    z = torch.zeros(1, 2, N, D)
    return z, z.clone(), z.clone()


def test_dense_graph_shape():
    @af.attention
    def fn(Q, K, V):
        return af.softmax(af.scaled_dot_product(Q, K)) @ V

    g = fn(*_zeros(), return_graph=True)
    assert g.score().scale is None
    assert {m.kind for m in g.collect_masks()} == set()  # no mask op
    assert g.norm().kind == NormKind.SOFTMAX


def test_causal_graph_shape():
    @af.attention
    def fn(Q, K, V):
        return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V

    g = fn(*_zeros(), return_graph=True)
    masks = g.collect_masks()
    assert len(masks) == 1 and masks[0].kind == MaskKind.CAUSAL


def test_sliding_window_validates_size():
    with pytest.raises(ValueError):
        @af.attention
        def fn(Q, K, V):
            s = af.sliding_window(af.scaled_dot_product(Q, K), window_size=0)
            return af.softmax(s) @ V
        fn(*_zeros(), return_graph=True)


def test_alibi_validates_heads():
    with pytest.raises(ValueError):
        @af.attention
        def fn(Q, K, V):
            s = af.alibi(af.scaled_dot_product(Q, K), num_heads=0)
            return af.softmax(s) @ V
        fn(*_zeros(), return_graph=True)


def test_missing_value_projection_message():
    @af.attention
    def fn(Q, K, V):
        return af.softmax(af.scaled_dot_product(Q, K))   # forgot @ V

    with pytest.raises(TypeError, match="value projection"):
        fn(*_zeros(), return_graph=True)


def test_signature_stable_across_traces():
    @af.attention
    def fn1(Q, K, V):
        return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V

    @af.attention
    def fn2(Q, K, V):
        return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V

    sig1 = fn1(*_zeros(), return_graph=True).signature()
    sig2 = fn2(*_zeros(), return_graph=True).signature()
    assert sig1 == sig2

    @af.attention
    def fn3(Q, K, V):
        # Different mask: signature must differ
        return af.softmax(af.scaled_dot_product(Q, K)) @ V

    sig3 = fn3(*_zeros(), return_graph=True).signature()
    assert sig3 != sig1
