"""Tiled-loop IR.

After the lowering pass we no longer think in dataflow nodes; we think in tile
loops over the (M, N) score matrix. The tiled IR is a small structured record
that the codegen pass walks linearly to produce a Triton kernel string.

A `TiledKernel` is best read top-to-bottom:

    FOR m_block IN range(0, M, BLOCK_M):
        load Q[m_block]                      (tile in SMEM)
        init  m_i = -inf, l_i = 0, acc = 0   (online softmax accumulators)
        FOR n_block IN range(<n_lo>, <n_hi>, BLOCK_N):
            load K[n_block], V[n_block]
            S = (Q @ K.T) * scale
            S += <bias>            (optional)
            S += <mask>            (optional, written as additive -inf mask)
            <update online softmax: m_new, l_new, acc>
        write O[m_block] = acc / l_new
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .high_level import MaskKind, BiasKind, NormKind


# ---------------------------------------------------------------------------
# Tile-size config
# ---------------------------------------------------------------------------


@dataclass
class TileConfig:
    """Block sizes + Triton launch hints. Filled in by the tiling pass."""

    BLOCK_M: int = 128
    BLOCK_N: int = 64
    num_warps: int = 4
    num_stages: int = 3

    # Codegen-level switches
    use_streaming_softmax: bool = True  # always True for SOFTMAX
    skip_full_mask_blocks: bool = True  # skip n_blocks fully masked-out (causal/sw)


# ---------------------------------------------------------------------------
# Lowered kernel
# ---------------------------------------------------------------------------


@dataclass
class TiledKernel:
    """Codegen-ready description of a single fused attention kernel."""

    # Shape information lifted from the high-level graph
    head_dim: int
    dtype: str            # "float16" / "bfloat16" / "float32"

    # Score
    score_scale: Optional[float]  # None -> 1/sqrt(head_dim)

    # Mask
    mask_kind: MaskKind = MaskKind.FULL
    mask_window: Optional[int] = None     # for SLIDING_WINDOW

    # Bias
    bias_kind: Optional[BiasKind] = None
    bias_num_heads: Optional[int] = None  # for ALIBI

    # Norm
    norm_kind: NormKind = NormKind.SOFTMAX

    # Tile / launch config
    config: TileConfig = field(default_factory=TileConfig)

    # Positional encoding
    rope_kind: int = 0   # 0 = no RoPE, 1 = fused RoPE (Su et al., 2021)

    # The compiler stamps the cache key here so the runtime can dedup.
    cache_key: str = ""
