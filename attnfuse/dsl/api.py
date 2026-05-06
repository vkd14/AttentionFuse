"""User-facing API for AttnFuse.

The `@attention` decorator traces a Python function that uses combinator calls
on symbolic tensors and returns a callable that JIT-compiles a fused Triton
kernel on first use.
"""
from __future__ import annotations

import functools
import os
from typing import Callable

from ..ir.high_level import (
    Expr,
    TensorSym,
    ScoreOp,
    MaskOp,
    BiasOp,
    NormOp,
    MatMulPV,
    Graph,
    ScoreKind,
    MaskKind,
    BiasKind,
    NormKind,
)
from .tracer import trace_attention_fn

# ---------------------------------------------------------------------------
# Combinators -- each returns an IR node and is callable from inside @attention
# ---------------------------------------------------------------------------


def scaled_dot_product(Q: TensorSym, K: TensorSym, scale: float | None = None) -> Expr:
    """S = (Q @ K.T) * scale.   `scale` defaults to 1/sqrt(head_dim)."""
    return ScoreOp(kind=ScoreKind.SCALED_DOT, q=Q, k=K, scale=scale)


def rope(Q: TensorSym, K: TensorSym, scale: float | None = None) -> Expr:
    """S = (RoPE(Q) @ RoPE(K).T) * scale — fused rotary positional encoding.

    Pass the precomputed tables at call time via ``cos=`` and ``sin=`` kwargs::

        out = my_attn(Q, K, V, cos=cos_table, sin=sin_table)

    Both tables should have shape (N, D) or (1, 1, N, D) and the same dtype as Q.
    """
    return ScoreOp(kind=ScoreKind.SCALED_DOT, q=Q, k=K, scale=scale, rope=True)


def additive_bias(scores: Expr) -> Expr:
    """S' = S + bias  where bias is a (B, H, N, N) tensor injected at call time.

    Pass the actual tensor at call time via the ``bias=`` keyword argument::

        out = my_attn(Q, K, V, bias=bias_tensor)
    """
    return BiasOp(kind=BiasKind.ADDITIVE, scores=scores, bias=None)


def causal(scores: Expr) -> Expr:
    """Strictly-lower-triangular mask: S[i, j] += -inf when j > i."""
    return MaskOp(kind=MaskKind.CAUSAL, scores=scores)


def sliding_window(scores: Expr, window_size: int) -> Expr:
    """Local attention: S[i, j] += -inf when |i - j| >= window_size."""
    if window_size <= 0:
        raise ValueError("sliding_window: window_size must be positive")
    return MaskOp(kind=MaskKind.SLIDING_WINDOW, scores=scores, window=window_size)


def full(scores: Expr) -> Expr:
    """No-op mask. Present so `mask=full` is a valid combinator slot."""
    return MaskOp(kind=MaskKind.FULL, scores=scores)


def alibi(scores: Expr, num_heads: int) -> Expr:
    """ALiBi linear positional bias (Press et al., 2021).

    bias[h, i, j] = -slope[h] * |i - j|    for bidirectional, or
    bias[h, i, j] = -slope[h] * (i - j)    for causal (clamped to >= 0)
    """
    if num_heads <= 0:
        raise ValueError("alibi: num_heads must be positive")
    return BiasOp(kind=BiasKind.ALIBI, scores=scores, num_heads=num_heads)


def softmax(scores: Expr) -> Expr:
    """Row-wise stable softmax (online / streaming softmax in the kernel)."""
    return NormOp(kind=NormKind.SOFTMAX, scores=scores)


def relu_attention(scores: Expr) -> Expr:
    """ReLU normalisation (Wortsman et al., 2023): max(S, 0) instead of softmax."""
    return NormOp(kind=NormKind.RELU, scores=scores)


# ---------------------------------------------------------------------------
# @attention decorator
# ---------------------------------------------------------------------------


def attention(fn: Callable) -> Callable:
    """Decorator: trace a Python function written with AttnFuse combinators.

    The decorated function is callable with real `torch.Tensor` Q/K/V; on the
    first call we trace once (no GPU work) to recover the high-level IR, hand
    it to the compiler, and cache the compiled kernel keyed by (graph-hash,
    dtype, head_dim).  Subsequent calls dispatch directly to the kernel.
    """
    # Defer compiler import to avoid pulling Triton into the import path of
    # tools that only want to inspect IR (e.g. tests on CPU-only machines).
    graph: Graph | None = None

    @functools.wraps(fn)
    def wrapper(Q, K, V, *, bias=None, cos=None, sin=None,
                return_graph: bool = False, **kwargs):
        nonlocal graph
        if graph is None:
            graph = trace_attention_fn(fn, Q, K, V)
            if os.environ.get("ATTNFUSE_DEBUG"):
                from ..ir.printer import format_graph
                print("[AttnFuse] high-level IR:")
                print(format_graph(graph))
        if return_graph:
            return graph

        from ..runtime.dispatch import run_attention
        return run_attention(graph, Q, K, V, bias=bias, cos=cos, sin=sin, **kwargs)

    wrapper.graph_fn = fn  # type: ignore[attr-defined]
    return wrapper
