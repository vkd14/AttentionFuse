"""Two-level IR for AttnFuse.

Level 1 -- `high_level`: dataflow graph of the attention spec.
Level 2 -- `tiled`     : explicit tiled-loop form ready for Triton codegen.
"""
from .high_level import (
    Expr, TensorSym, ScoreOp, MaskOp, BiasOp, NormOp, MatMulPV, Graph,
    ScoreKind, MaskKind, BiasKind, NormKind,
)
from .tiled import TiledKernel, TileConfig
from .printer import format_graph, format_tiled

__all__ = [
    "Expr", "TensorSym", "ScoreOp", "MaskOp", "BiasOp", "NormOp", "MatMulPV", "Graph",
    "ScoreKind", "MaskKind", "BiasKind", "NormKind",
    "TiledKernel", "TileConfig",
    "format_graph", "format_tiled",
]
