"""Fusion pass.

For the attention kernels AttnFuse generates the only fusion that matters is

    score -> (mask) -> (bias) -> softmax -> @V

i.e. fuse the entire row of operators into a single tile loop with online
softmax (Milakov & Gimelshein, 2018). This pass is therefore conceptually a
*recognition* step rather than a transformation: it walks the graph and asserts
that everything between the score op and the final matmul is fusion-eligible
(no data-dependent branching, no cross-row reductions other than softmax).

If the graph violates a fusion precondition we raise -- the user's spec was
malformed, and silently falling back to a non-fused kernel would defeat the
purpose of the project.
"""
from __future__ import annotations

from ..ir.high_level import (
    Graph, ScoreOp, MaskOp, BiasOp, NormOp, MatMulPV, NormKind,
)


class FusionError(RuntimeError):
    pass


def fuse_score_softmax(graph: Graph) -> Graph:
    """Verify the graph is shaped as a fusable attention pipeline."""
    root = graph.root
    if not isinstance(root, MatMulPV):
        raise FusionError("Graph root must be MatMulPV (probs @ V).")

    norm = root.probs
    if not isinstance(norm, NormOp):
        raise FusionError(
            "MatMulPV.probs must be the output of a normalisation combinator."
        )
    if norm.kind not in (NormKind.SOFTMAX, NormKind.RELU):
        raise FusionError(f"Unsupported normalisation: {norm.kind}")

    # Walk down: norm -> (zero or more mask/bias) -> score.  Anything else
    # rejects the graph.
    cursor = norm.scores
    while isinstance(cursor, (MaskOp, BiasOp)):
        cursor = cursor.scores  # both have .scores referring to upstream Expr

    if not isinstance(cursor, ScoreOp):
        raise FusionError(
            f"Expected the chain norm <- (mask|bias)* <- score; "
            f"hit {type(cursor).__name__} instead."
        )

    # No transformation needed -- the lowering pass already walks the graph in
    # this order. Fusion is implicit in the codegen template.
    return graph
