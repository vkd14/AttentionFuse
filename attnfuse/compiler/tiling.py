"""Tiling-analysis pass.

For RTX 3090 Ti (Ampere, 84 SM × 128 KB SMEM, 64K registers/SM) the sweet
spot for fused-attention forward (no backward) is:

    head_dim   BLOCK_M   BLOCK_N   warps   stages
    -------------------------------------------
       32       128        64        4        3
       64       128        64        4        3
       96       128        64        4        3      (a bit register-bound)
      128       128        32        8        2      (register-bound; smaller N)
      256        64        32        4        2      (rare; OOM-ish)

The numbers below were chosen by sweeping in Triton 2.2 + CUDA 12.1 on a 3090 Ti.
Re-tune if you change the GPU.
"""
from __future__ import annotations

from ..ir.high_level import Graph, MaskKind, ScoreOp
from ..ir.tiled import TileConfig

# fp16/bf16 configs (2 bytes/element — 3 pipeline stages fit in 101 KB SMEM).
_AMPERE_TABLE_F16: dict[int, TileConfig] = {
    32:  TileConfig(BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=3),
    64:  TileConfig(BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=3),
    96:  TileConfig(BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=3),
    128: TileConfig(BLOCK_M=128, BLOCK_N=32, num_warps=8, num_stages=2),
    256: TileConfig(BLOCK_M=64,  BLOCK_N=32, num_warps=4, num_stages=2),
}

# RoPE adds cos/sin loads for both Q and K, increasing SMEM pressure.
# Use num_stages=2 and smaller BLOCK_N to stay within 101 KB.
_AMPERE_TABLE_F16_ROPE: dict[int, TileConfig] = {
    32:  TileConfig(BLOCK_M=128, BLOCK_N=32, num_warps=4, num_stages=2),
    64:  TileConfig(BLOCK_M=128, BLOCK_N=32, num_warps=4, num_stages=2),
    96:  TileConfig(BLOCK_M=128, BLOCK_N=32, num_warps=4, num_stages=2),
    128: TileConfig(BLOCK_M=64,  BLOCK_N=32, num_warps=4, num_stages=2),
    256: TileConfig(BLOCK_M=64,  BLOCK_N=16, num_warps=4, num_stages=2),
}

_AMPERE_TABLE_F32_ROPE: dict[int, TileConfig] = {
    32:  TileConfig(BLOCK_M=64,  BLOCK_N=32, num_warps=4, num_stages=2),
    64:  TileConfig(BLOCK_M=64,  BLOCK_N=32, num_warps=4, num_stages=2),
    96:  TileConfig(BLOCK_M=64,  BLOCK_N=32, num_warps=4, num_stages=2),
    128: TileConfig(BLOCK_M=64,  BLOCK_N=16, num_warps=4, num_stages=2),
    256: TileConfig(BLOCK_M=32,  BLOCK_N=16, num_warps=4, num_stages=2),
}

# fp32 configs (4 bytes/element — 3 stages would need 128 KB, over the 101 KB limit;
# drop to 2 stages so peak SMEM = 2*(BN*D + BN*D)*4 + BM*D*4 ≤ 101 KB).
_AMPERE_TABLE_F32: dict[int, TileConfig] = {
    32:  TileConfig(BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=2),
    64:  TileConfig(BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=2),
    96:  TileConfig(BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=2),
    128: TileConfig(BLOCK_M=64,  BLOCK_N=32, num_warps=8, num_stages=2),
    256: TileConfig(BLOCK_M=64,  BLOCK_N=32, num_warps=4, num_stages=2),
}


def _graph_has_rope(graph: Graph) -> bool:
    score_nodes = [n for n, _ in graph.walk() if isinstance(n, ScoreOp)]
    return any(s.rope for s in score_nodes)


def choose_tile_config(graph: Graph) -> TileConfig:
    """Return a tile config tuned for `graph` on RTX 3090 Ti."""
    head_dim = graph.q.head_dim
    has_rope = _graph_has_rope(graph)

    if graph.q.dtype == "float32":
        table = _AMPERE_TABLE_F32_ROPE if has_rope else _AMPERE_TABLE_F32
    else:
        table = _AMPERE_TABLE_F16_ROPE if has_rope else _AMPERE_TABLE_F16

    cfg = table.get(head_dim)
    if cfg is None:
        # Conservative fallback: small blocks, low warp count.
        cfg = TileConfig(BLOCK_M=64, BLOCK_N=32, num_warps=4, num_stages=2)

    # Mask-aware skipping is most useful for causal / sliding-window: empty
    # n_blocks above the diagonal (or outside the window) can be skipped.
    masks = {m.kind for m in graph.collect_masks()}
    if masks <= {MaskKind.FULL}:
        cfg.skip_full_mask_blocks = False

    return cfg
