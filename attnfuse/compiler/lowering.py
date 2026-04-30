"""Lowering: high-level Graph -> TiledKernel.

The lowering pass collapses the dataflow graph into a small bag of attributes
that codegen consumes. Because every supported variant fuses into a single
tiled loop, the lowered form is intentionally flat -- no node-by-node dispatch
in codegen.
"""
from __future__ import annotations

import math

from ..ir.high_level import (
    Graph, ScoreOp, MaskOp, BiasOp, NormOp, ScoreKind, MaskKind, BiasKind, NormKind,
)
from ..ir.tiled import TiledKernel
from .tiling import choose_tile_config
from .fusion import fuse_score_softmax


def lower_to_tiled(graph: Graph) -> TiledKernel:
    """Lower a verified graph to a TiledKernel."""
    graph = fuse_score_softmax(graph)

    score = graph.score()
    if score.kind is not ScoreKind.SCALED_DOT:
        raise NotImplementedError(f"Score kind {score.kind} not yet supported")
    head_dim = graph.q.head_dim
    score_scale = (
        score.scale if score.scale is not None else 1.0 / math.sqrt(head_dim)
    )

    masks = graph.collect_masks()
    if len(masks) > 1:
        # Fold multiple masks into the strictest one. Today the only
        # combination we permit is sliding_window inside causal -- both shrink
        # the eligible-keys range, and SLIDING_WINDOW is strictly inside CAUSAL
        # for any (i, j) with j > i, so we keep SLIDING_WINDOW and warn.
        kinds = {m.kind for m in masks}
        if kinds == {MaskKind.CAUSAL, MaskKind.SLIDING_WINDOW}:
            sw = next(m for m in masks if m.kind is MaskKind.SLIDING_WINDOW)
            mask_kind, mask_window = MaskKind.SLIDING_WINDOW, sw.window
        else:
            raise NotImplementedError(
                f"Mask combination {kinds} not yet supported"
            )
    elif masks:
        mask_kind = masks[0].kind
        mask_window = masks[0].window
    else:
        mask_kind, mask_window = MaskKind.FULL, None

    biases = graph.collect_biases()
    if len(biases) > 1:
        raise NotImplementedError("Multiple bias ops in one graph not yet supported")
    if biases:
        bias_kind = biases[0].kind
        bias_num_heads = biases[0].num_heads
        if bias_kind is BiasKind.ADDITIVE:
            raise NotImplementedError(
                "Externally-supplied additive bias not yet wired into the runtime; "
                "use af.alibi() instead, or extend dispatch.run_attention to pass it."
            )
    else:
        bias_kind, bias_num_heads = None, None

    norm = graph.norm()

    cfg = choose_tile_config(graph)
    if norm.kind is NormKind.RELU:
        # Streaming softmax accumulators are unused in the ReLU path.
        cfg.use_streaming_softmax = False

    kernel = TiledKernel(
        head_dim=head_dim,
        dtype=graph.q.dtype,
        score_scale=score_scale,
        mask_kind=mask_kind,
        mask_window=mask_window,
        bias_kind=bias_kind,
        bias_num_heads=bias_num_heads,
        norm_kind=norm.kind,
        config=cfg,
        cache_key=graph.signature(),
    )
    return kernel
