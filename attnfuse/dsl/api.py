"""User-facing API for AttnFuse.

The `@attention` decorator traces a Python function that uses combinator calls
on symbolic tensors and returns a callable that JIT-compiles a fused Triton
kernel on first use.
"""
from __future__ import annotations

import functools
import os
from typing import Callable

import torch

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


def block_sparse(scores: Expr) -> Expr:
    """User-supplied block-sparse mask (BigBird / FlexAttention style).

    The mask is provided at call time via ``block_mask=<BlockMask>`` kwarg.
    Use :func:`attnfuse.create_block_mask` to build a BlockMask from a
    Python predicate. The kernel iterates only over the active (m, n)
    block pairs, so FLOPs and HBM traffic are both genuinely sub-quadratic.
    """
    return MaskOp(kind=MaskKind.BLOCK_SPARSE, scores=scores)


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

    The decorated function is callable with real `torch.Tensor` Q/K/V; we
    trace once per distinct (head_dim, dtype) pair, hand the graph to the
    compiler, and cache the compiled kernel keyed by the graph signature.
    Re-tracing on shape change is essentially free (it does no GPU work)
    and only happens when the user calls the same function with a new
    head_dim or dtype -- a rare event in normal use.
    """
    graphs: dict[tuple, Graph] = {}

    @functools.wraps(fn)
    def wrapper(Q, K, V, *, bias=None, cos=None, sin=None,
                block_mask=None, return_graph: bool = False, **kwargs):
        # Key the cached graph by the dimensions that drive codegen
        # specialisation (head_dim becomes a tl.constexpr, dtype gates
        # the tile-config table, has-RoPE is a graph property already).
        key = (Q.shape[-1], str(Q.dtype))
        graph = graphs.get(key)
        if graph is None:
            graph = trace_attention_fn(fn, Q, K, V)
            graphs[key] = graph
            if os.environ.get("ATTNFUSE_DEBUG"):
                from ..ir.printer import format_graph
                print(f"[AttnFuse] high-level IR (head_dim={key[0]} dtype={key[1]}):")
                print(format_graph(graph))
        if return_graph:
            return graph

        # Block-sparse path: dedicated kernel that iterates over the
        # user-supplied active-block lists in the BlockMask.
        masks = {m.kind for m in graph.collect_masks()}
        if MaskKind.BLOCK_SPARSE in masks:
            if block_mask is None:
                raise RuntimeError(
                    "Graph uses af.block_sparse(); pass a BlockMask via "
                    "block_mask=<BlockMask> at call time. Build one via "
                    "attnfuse.create_block_mask(predicate, Q_LEN, KV_LEN)."
                )
            from ..runtime.block_mask import run_block_sparse, BlockMask as _BM
            if not isinstance(block_mask, _BM):
                raise TypeError("block_mask must be an attnfuse.BlockMask")
            # ALiBi if present in graph
            biases = {b.kind for b in graph.collect_biases()}
            from ..ir.high_level import BiasKind as _BK
            bias_kind_int = 1 if _BK.ALIBI in biases else 0
            sm_scale = 1.0 / (Q.shape[-1] ** 0.5)
            return run_block_sparse(Q, K, V, block_mask, sm_scale,
                                     bias_kind=bias_kind_int)

        # Route through autograd.Function when any input requires gradient.
        # Inference paths (no requires_grad) skip the L allocation entirely
        # via the direct run_attention call.
        needs_grad = (
            isinstance(Q, torch.Tensor) and Q.requires_grad
        ) or (
            isinstance(K, torch.Tensor) and K.requires_grad
        ) or (
            isinstance(V, torch.Tensor) and V.requires_grad
        )
        if needs_grad:
            from ..runtime.autograd import AttnFuseFunction
            from ..runtime.backward import can_backward
            if can_backward(graph):
                return AttnFuseFunction.apply(graph, Q, K, V, bias, cos, sin)
            # Outside the supported backward scope: fall through to the
            # inference path so forward still works (no gradient will flow).

        from ..runtime.dispatch import run_attention
        return run_attention(graph, Q, K, V, bias=bias, cos=cos, sin=sin, **kwargs)

    wrapper.graph_fn = fn  # type: ignore[attr-defined]
    return wrapper
