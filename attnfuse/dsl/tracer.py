"""Tracer: run a user attention function on symbolic tensors and recover the IR.

Tracing strategy
----------------
We do NOT execute on real GPU memory. We pass `TensorSym` placeholders through
the user function. Combinators (`scaled_dot_product`, `causal`, `softmax`, ...)
return IR nodes (`Expr` subclasses). The user is also expected to write the
final `probs @ V` step using `@` -- we override `Expr.__matmul__` to capture it.

The output Expr is then walked into a `Graph` object.
"""
from __future__ import annotations

from typing import Callable

from ..ir.high_level import (
    Expr,
    TensorSym,
    Graph,
    MatMulPV,
    NormOp,
)


def _make_sym(name: str, ref) -> TensorSym:
    """Build a symbolic tensor that mirrors the shape/dtype of a real torch.Tensor.

    Accepted real-tensor layouts are (B, H, N, D) (PyTorch SDPA convention).
    """
    shape = tuple(ref.shape)
    if len(shape) != 4:
        raise ValueError(
            f"AttnFuse expects Q/K/V of shape (B, H, N, D); got {shape}"
        )
    dtype = str(ref.dtype).replace("torch.", "")
    return TensorSym(
        name=name,
        batch=shape[0],
        num_heads=shape[1],
        seqlen=shape[2],
        head_dim=shape[3],
        dtype=dtype,
    )


def trace_attention_fn(fn: Callable, Q, K, V) -> Graph:
    """Run `fn(Q_sym, K_sym, V_sym)` once to capture its IR.

    The user-written function must:
      * call combinators on Q/K to produce a score Expr,
      * (optionally) apply mask/bias combinators,
      * call a normalisation combinator (`softmax` or `relu_attention`),
      * return `probs @ V` (where `probs` is the normalised score Expr).
    """
    Qs = _make_sym("Q", Q)
    Ks = _make_sym("K", K)
    Vs = _make_sym("V", V)

    if Qs.head_dim != Ks.head_dim:
        raise ValueError(
            f"head_dim mismatch: Q.head_dim={Qs.head_dim} K.head_dim={Ks.head_dim}"
        )
    if Vs.head_dim != Qs.head_dim and Vs.head_dim not in (Qs.head_dim,):
        # K and V can in principle have different head_dim from Q (cross attn);
        # for now AttnFuse only supports self-attention head dims.
        raise ValueError("AttnFuse currently requires Q.head_dim == V.head_dim")

    out = fn(Qs, Ks, Vs)

    if not isinstance(out, MatMulPV):
        # Tolerate the user calling softmax(...) and forgetting `@ V` -- give a
        # readable diagnostic instead of a confusing AttributeError later.
        raise TypeError(
            "Attention function must return `probs @ V`; got "
            f"{type(out).__name__}.  Did you forget the value projection?"
        )

    # Sanity: norm op must dominate the matmul-by-V.
    if not isinstance(out.probs, NormOp):
        raise TypeError(
            "The attention probabilities passed to `@ V` must come from a "
            "normalisation combinator (af.softmax / af.relu_attention)."
        )

    return Graph(q=Qs, k=Ks, v=Vs, root=out)
