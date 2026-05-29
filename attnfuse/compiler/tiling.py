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

# fp16/bf16 default (dense / no-mask) configs. 3 pipeline stages fit in 101 KB SMEM.
# Tuned on RTX 3090 (sm_86) via benchmarks/config_sweep.py.
#
# NOTE: Triton's `tl.arange(0, HEAD_DIM)` requires HEAD_DIM to be a power of 2,
# so only {32, 64, 128, 256} are real supported values. Non-power-of-2 head dims
# (e.g. 96 for some research models) would need separate handling and are
# explicitly out of scope for this release.
_AMPERE_TABLE_F16: dict[int, TileConfig] = {
    32:  TileConfig(BLOCK_M=128, BLOCK_N=64, num_warps=8, num_stages=2),
    64:  TileConfig(BLOCK_M=128, BLOCK_N=64, num_warps=8, num_stages=2),
    128: TileConfig(BLOCK_M=128, BLOCK_N=32, num_warps=8, num_stages=2),
    256: TileConfig(BLOCK_M=64,  BLOCK_N=32, num_warps=4, num_stages=2),
}

# For variants with reduced work per query (causal, sliding-window, ALiBi),
# smaller tiles (BM=64, BN=32) give substantially better SM utilisation
# because more parallel tiles fit on each SM. The sweep showed ~17-25%
# latency improvement at N=4096 with these configs, taking AttnFuse
# from 86% of flex_attention to >100% on sliding-window.
_AMPERE_TABLE_F16_SPARSE: dict[int, TileConfig] = {
    32:  TileConfig(BLOCK_M=64,  BLOCK_N=32, num_warps=4, num_stages=3),
    64:  TileConfig(BLOCK_M=64,  BLOCK_N=32, num_warps=4, num_stages=3),
    # HEAD_DIM=128 wants BIGGER BLOCK_M to amortise the cost of the wider
    # Q register footprint. Discovered via benchmarks/gqa_config_sweep.py
    # on Llama-3-8B geometry; 16-19% faster than BM=64.
    128: TileConfig(BLOCK_M=128, BLOCK_N=32, num_warps=4, num_stages=2),
    256: TileConfig(BLOCK_M=64,  BLOCK_N=32, num_warps=4, num_stages=2),
}

# RoPE adds cos/sin loads for both Q and K, increasing SMEM pressure.
# Use num_stages=2 and smaller BLOCK_N to stay within 101 KB.
_AMPERE_TABLE_F16_ROPE: dict[int, TileConfig] = {
    32:  TileConfig(BLOCK_M=128, BLOCK_N=32, num_warps=4, num_stages=2),
    64:  TileConfig(BLOCK_M=128, BLOCK_N=32, num_warps=4, num_stages=2),
    128: TileConfig(BLOCK_M=64,  BLOCK_N=32, num_warps=4, num_stages=2),
    256: TileConfig(BLOCK_M=64,  BLOCK_N=16, num_warps=4, num_stages=2),
}

_AMPERE_TABLE_F32_ROPE: dict[int, TileConfig] = {
    32:  TileConfig(BLOCK_M=64,  BLOCK_N=32, num_warps=4, num_stages=2),
    64:  TileConfig(BLOCK_M=64,  BLOCK_N=32, num_warps=4, num_stages=2),
    128: TileConfig(BLOCK_M=64,  BLOCK_N=16, num_warps=4, num_stages=2),
    256: TileConfig(BLOCK_M=32,  BLOCK_N=16, num_warps=4, num_stages=2),
}

# fp32 configs (4 bytes/element — 3 stages would need 128 KB, over the 101 KB limit;
# drop to 2 stages so peak SMEM = 2*(BN*D + BN*D)*4 + BM*D*4 ≤ 101 KB).
_AMPERE_TABLE_F32: dict[int, TileConfig] = {
    32:  TileConfig(BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=2),
    64:  TileConfig(BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=2),
    128: TileConfig(BLOCK_M=64,  BLOCK_N=32, num_warps=8, num_stages=2),
    256: TileConfig(BLOCK_M=64,  BLOCK_N=32, num_warps=4, num_stages=2),
}


def _detect_hopper() -> bool:
    """True on sm_90 (Hopper, H100 / H200). Used to pick a separate tile table
    that exploits the larger SMEM (228 KB vs Ampere's 101 KB), more pipeline
    stages, and bigger tile sizes that Hopper's WGMMA atomically supports.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 9
    except Exception:
        return False


_IS_HOPPER = _detect_hopper()


# Hopper (sm_90) tile table. Hopper has ~228 KB of SMEM per SM, double-rate
# matmul via WGMMA, TMA descriptors for async memory ops, and a higher
# default num_stages sweet spot. Bigger BLOCK_M takes advantage of WGMMA's
# 64-aligned M dimension; larger num_stages amortises the higher SM count
# (132 SMs vs Ampere's 82). These defaults are starting points -- a
# benchmark sweep on H100 hardware should refine them.
# Updated 2026-05-29 based on the empirical sweep on H100 SXM5:
# Hopper's 228 KB SMEM lets BLOCK_M=256 hold a large Q tile without spilling;
# deeper num_stages exploits the WGMMA pipeline.
_HOPPER_TABLE_F16: dict[int, TileConfig] = {
    32:  TileConfig(BLOCK_M=128, BLOCK_N=64, num_warps=8, num_stages=3),
    64:  TileConfig(BLOCK_M=128, BLOCK_N=64, num_warps=8, num_stages=3),
    128: TileConfig(BLOCK_M=128, BLOCK_N=64, num_warps=8, num_stages=3),
    256: TileConfig(BLOCK_M=64,  BLOCK_N=64, num_warps=8, num_stages=2),
}

# Causal / SW / ALiBi -- the sweep showed BLOCK_M=256 wins on causal and
# ALiBi at large N because the bigger Q tile amortises the Q-load cost
# across more in-window KV blocks. Sliding-window prefers BLOCK_M=64
# because its narrow per-query work doesn't justify the bigger Q tile.
_HOPPER_TABLE_F16_SPARSE: dict[int, TileConfig] = {
    32:  TileConfig(BLOCK_M=256, BLOCK_N=64, num_warps=4, num_stages=2),
    64:  TileConfig(BLOCK_M=256, BLOCK_N=64, num_warps=4, num_stages=2),
    128: TileConfig(BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=3),
    256: TileConfig(BLOCK_M=64,  BLOCK_N=64, num_warps=4, num_stages=2),
}

_HOPPER_TABLE_F16_ROPE: dict[int, TileConfig] = {
    32:  TileConfig(BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=3),
    64:  TileConfig(BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=3),
    128: TileConfig(BLOCK_M=64,  BLOCK_N=64, num_warps=4, num_stages=2),
    256: TileConfig(BLOCK_M=64,  BLOCK_N=32, num_warps=4, num_stages=2),
}

_HOPPER_TABLE_F32: dict[int, TileConfig] = {
    32:  TileConfig(BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=3),
    64:  TileConfig(BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=3),
    128: TileConfig(BLOCK_M=64,  BLOCK_N=64, num_warps=8, num_stages=2),
    256: TileConfig(BLOCK_M=64,  BLOCK_N=32, num_warps=4, num_stages=2),
}

_HOPPER_TABLE_F32_ROPE: dict[int, TileConfig] = {
    32:  TileConfig(BLOCK_M=64,  BLOCK_N=64, num_warps=4, num_stages=2),
    64:  TileConfig(BLOCK_M=64,  BLOCK_N=64, num_warps=4, num_stages=2),
    128: TileConfig(BLOCK_M=64,  BLOCK_N=32, num_warps=4, num_stages=2),
    256: TileConfig(BLOCK_M=32,  BLOCK_N=32, num_warps=4, num_stages=2),
}


def _graph_has_rope(graph: Graph) -> bool:
    score_nodes = [n for n, _ in graph.walk() if isinstance(n, ScoreOp)]
    return any(s.rope for s in score_nodes)


def _graph_is_sparse(graph: Graph) -> bool:
    """A graph is 'sparse' (in the workload sense) if any mask or bias
    reduces the effective FLOP count vs dense. Causal, sliding-window,
    and ALiBi (the latter adds a per-tile bias load that overlaps better
    with compute when tiles are smaller) all fall here.
    """
    masks = {m.kind for m in graph.collect_masks()}
    biases = {b.kind for b in graph.collect_biases()}
    has_sparse_mask = bool(masks - {MaskKind.FULL})
    has_alibi_bias = any(b.kind.value == "alibi" for b in graph.collect_biases())
    return has_sparse_mask or has_alibi_bias


def choose_tile_config(graph: Graph) -> TileConfig:
    """Return a tile config tuned for `graph` on RTX 3090 (sm_86).

    Selection order:
      1. RoPE-fused graphs use SMEM-constrained tables (extra cos/sin loads).
      2. Sparse graphs (causal, SW, ALiBi) use smaller tiles for better SM fill.
      3. Dense fp16/bf16 graphs use the default Ampere table.
      4. fp32 graphs use their own table (2 stages to fit SMEM).
    """
    head_dim = graph.q.head_dim
    has_rope = _graph_has_rope(graph)

    if _IS_HOPPER:
        if graph.q.dtype == "float32":
            table = _HOPPER_TABLE_F32_ROPE if has_rope else _HOPPER_TABLE_F32
        elif has_rope:
            table = _HOPPER_TABLE_F16_ROPE
        elif _graph_is_sparse(graph):
            table = _HOPPER_TABLE_F16_SPARSE
        else:
            table = _HOPPER_TABLE_F16
    else:
        if graph.q.dtype == "float32":
            table = _AMPERE_TABLE_F32_ROPE if has_rope else _AMPERE_TABLE_F32
        elif has_rope:
            table = _AMPERE_TABLE_F16_ROPE
        elif _graph_is_sparse(graph):
            table = _AMPERE_TABLE_F16_SPARSE
        else:
            table = _AMPERE_TABLE_F16

    cfg = table.get(head_dim)
    if cfg is None:
        cfg = TileConfig(BLOCK_M=64, BLOCK_N=32, num_warps=4, num_stages=2)

    masks = {m.kind for m in graph.collect_masks()}
    if masks <= {MaskKind.FULL}:
        cfg.skip_full_mask_blocks = False

    return cfg
