"""Runtime path for Flash Decoding (split-K attention).

Activates automatically when the input shape matches the autoregressive
decode pattern (Q.N = 1, large N_kv) and the graph uses only the supported
combinators (dense scaled-dot-product, optional ALiBi).
"""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import math
import sys
from pathlib import Path

import torch

from ..ir.high_level import Graph, BiasKind, MaskKind
from ..compiler.codegen_decode import get_decode_source

# Module-level cache for the compiled decode kernels (keyed by source hash).
_decode_module = None
_GENERATED_DIR = Path(__file__).parent.parent / "_generated"


def _load_decode_kernels():
    """Lazily compile the Flash Decoding split + combine kernels."""
    global _decode_module
    if _decode_module is not None:
        return _decode_module
    src = get_decode_source()
    _GENERATED_DIR.mkdir(exist_ok=True)
    h = hashlib.sha1(src.encode()).hexdigest()[:12]
    mod_name = f"_attnfuse_decode_{h}"
    fpath = _GENERATED_DIR / f"{mod_name}.py"
    if not fpath.exists():
        fpath.write_text(src)
    spec = importlib.util.spec_from_file_location(mod_name, fpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    _decode_module = mod
    return mod


def can_use_flash_decode(graph: Graph, Q: torch.Tensor, K: torch.Tensor) -> bool:
    """Heuristic: should we route this call through the Flash Decoding path?"""
    # Q.N must be 1 (the decode pattern); large enough N_kv that splitting helps
    if Q.shape[2] != 1:
        return False
    if K.shape[2] < 256:
        return False
    # Restrictions for this initial decode-kernel scope (matches docstring)
    masks = {m.kind for m in graph.collect_masks()}
    if masks - {MaskKind.FULL}:
        return False
    biases = {b.kind for b in graph.collect_biases()}
    if biases - {BiasKind.ALIBI}:
        return False
    # No fused RoPE in the decode kernel yet
    from ..ir.high_level import ScoreOp
    score_nodes = [n for n, _ in graph.walk() if isinstance(n, ScoreOp)]
    if any(s.rope for s in score_nodes):
        return False
    return True


@functools.lru_cache(maxsize=64)
def _pick_num_splits(B: int, H_q: int, N_kv: int) -> int:
    """Pick num_splits to saturate SMs without making each chunk so small
    that per-program overhead dominates.

    Total programs = B * H_q * NUM_SPLITS. We want this in the range
    [2*num_SMs, 8*num_SMs] -- enough for the scheduler to hide tail
    effects, not so many that each program is launch-bound. Each split
    should hold at least ~1024 keys (8 BLOCK_N=128 tiles) to make the
    matmul work amortise the per-split setup cost.
    """
    try:
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:
        num_sms = 80
    # Lower bound from SM saturation
    saturation_lo = max(1, (2 * num_sms + B * H_q - 1) // (B * H_q))
    # Upper bound from per-program work (keep ~1024 keys per split)
    work_cap = max(1, N_kv // 1024)
    # Upper bound from over-provisioning (don't exceed 8x SM count)
    sched_cap = max(1, (8 * num_sms + B * H_q - 1) // (B * H_q))

    target = max(saturation_lo, min(work_cap, sched_cap))
    # Round up to next power of 2
    p = 1
    while p < target:
        p *= 2
    return max(1, min(p, 32))


BLOCK_H_FOR_DECODE = 16   # Triton tl.dot needs M, N, K >= 16


def run_flash_decode(graph: Graph, Q: torch.Tensor, K: torch.Tensor,
                     V: torch.Tensor, sm_scale: float,
                     bias_kind: int, num_heads_kv: int,
                     out: torch.Tensor) -> torch.Tensor:
    """Two-phase attention for Q.N = 1, large N_kv."""
    mod = _load_decode_kernels()
    B, H_q, _, D = Q.shape
    _, H_kv, N_kv, _ = K.shape
    group_size = H_q // H_kv
    # chunk_size: how many distinct Q heads each split-program owns.
    # GQA case (group_size < 16): one program per group, BLOCK_H pads with
    # cyclic replicas. MQA / large-group case: program covers BLOCK_H of the
    # group at a time.
    chunk_size = min(group_size, BLOCK_H_FOR_DECODE)
    if H_q % chunk_size != 0:
        # Pathological shape: fall back to scalar single-head per program
        # by setting BLOCK_H=1 path. (Triton tl.dot won't work; we'd need
        # the previous scalar kernel for this. For now just bail.)
        raise RuntimeError(
            f"Flash Decoding requires H_q divisible by min(group_size, 16); "
            f"got H_q={H_q}, group_size={group_size}"
        )
    num_h_chunks = H_q // chunk_size
    num_splits = _pick_num_splits(B, num_h_chunks, N_kv)

    # BLOCK_N choice: smaller for split kernels (each split is shorter).
    block_n = 128 if D >= 128 else 64
    if N_kv // num_splits < block_n:
        block_n = 64
    if N_kv // num_splits < block_n:
        block_n = 32

    # Workspace for partials (kept in fp32 for numerical stability)
    workspace_m = torch.empty((num_splits, B, H_q), dtype=torch.float32,
                              device=Q.device)
    workspace_l = torch.empty((num_splits, B, H_q), dtype=torch.float32,
                              device=Q.device)
    workspace_acc = torch.empty((num_splits, B, H_q, D), dtype=torch.float32,
                                device=Q.device)

    if bias_kind == 1:
        from .dispatch import _alibi_slopes
        slopes = _alibi_slopes(H_q, str(Q.device), str(Q.dtype))
    else:
        from .dispatch import _placeholder_slopes
        slopes = _placeholder_slopes(str(Q.device), str(Q.dtype))

    # --- Phase 1: split kernel ---
    grid_split = (num_splits, B * num_h_chunks)
    mod.attnfuse_decode_split_kernel[grid_split](
        Q, K, V,
        workspace_m, workspace_l, workspace_acc,
        sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        workspace_m.stride(0), workspace_m.stride(1), workspace_m.stride(2),
        workspace_l.stride(0), workspace_l.stride(1), workspace_l.stride(2),
        workspace_acc.stride(0), workspace_acc.stride(1),
        workspace_acc.stride(2), workspace_acc.stride(3),
        B, H_q, N_kv,
        slopes,
        GROUP_SIZE=group_size,
        BIAS_KIND=bias_kind,
        HEAD_DIM=D,
        BLOCK_N=block_n,
        NUM_SPLITS=num_splits,
        BLOCK_H=BLOCK_H_FOR_DECODE,
        num_warps=4, num_stages=2,
    )

    # --- Phase 2: combine kernel ---
    grid_combine = (B * H_q,)
    mod.attnfuse_decode_combine_kernel[grid_combine](
        workspace_m, workspace_l, workspace_acc,
        out,
        workspace_m.stride(0), workspace_m.stride(1), workspace_m.stride(2),
        workspace_l.stride(0), workspace_l.stride(1), workspace_l.stride(2),
        workspace_acc.stride(0), workspace_acc.stride(1),
        workspace_acc.stride(2), workspace_acc.stride(3),
        out.stride(0), out.stride(1), out.stride(3),
        B, H_q,
        HEAD_DIM=D,
        NUM_SPLITS=num_splits,
        num_warps=4, num_stages=1,
    )
    return out
