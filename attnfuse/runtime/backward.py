"""Runtime path for the backward kernels."""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import torch

from ..ir.high_level import Graph, MaskKind, BiasKind, ScoreOp
from ..compiler.codegen_backward import get_backward_source

_bwd_module = None
_GENERATED_DIR = Path(__file__).parent.parent / "_generated"

# Tile config for the backward kernels. Overridable for sweeping via
# attnfuse.runtime.backward._BWD_TILE.update({...}).
#
# Defaults derived from benchmarks/backward_config_sweep.py on RTX 3090:
#   N <=  512 : BM=64  BN=64  warps=8   (best small-N parallelism)
#   N == 1024 : BM=128 BN=32  warps=4
#   N >= 2048 : BM=64  BN=64  warps=4   (best at long context)
# We pick a single mid-range default and the dispatch overrides it for
# short sequences below.
_BWD_TILE = {
    "BLOCK_M":    64,
    "BLOCK_N":    64,
    "num_warps":  4,
    "num_stages": 2,
    "auto":       True,         # True: let _pick_backward_tile choose per N_q
}


def _pick_backward_tile(N_q: int) -> dict:
    """Return the best tile config for this sequence length."""
    if N_q <= 512:
        return {"BLOCK_M": 64,  "BLOCK_N": 64, "num_warps": 8, "num_stages": 2}
    if N_q <= 1024:
        return {"BLOCK_M": 128, "BLOCK_N": 32, "num_warps": 4, "num_stages": 2}
    return     {"BLOCK_M": 64,  "BLOCK_N": 64, "num_warps": 4, "num_stages": 2}


def _load_backward_kernels():
    global _bwd_module
    if _bwd_module is not None:
        return _bwd_module
    src = get_backward_source()
    _GENERATED_DIR.mkdir(exist_ok=True)
    h = hashlib.sha1(src.encode()).hexdigest()[:12]
    mod_name = f"_attnfuse_bwd_{h}"
    fpath = _GENERATED_DIR / f"{mod_name}.py"
    if not fpath.exists():
        fpath.write_text(src)
    spec = importlib.util.spec_from_file_location(mod_name, fpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    _bwd_module = mod
    return mod


def can_backward(graph: Graph) -> bool:
    """True iff the graph is in the supported backward scope."""
    masks  = {m.kind for m in graph.collect_masks()}
    if masks - {MaskKind.FULL, MaskKind.CAUSAL}:
        return False
    biases = {b.kind for b in graph.collect_biases()}
    if biases - {BiasKind.ALIBI}:
        return False
    score_nodes = [n for n, _ in graph.walk() if isinstance(n, ScoreOp)]
    if any(s.rope for s in score_nodes):
        return False
    return True


def run_backward(graph: Graph,
                 Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                 O: torch.Tensor, L: torch.Tensor, dO: torch.Tensor,
                 sm_scale: float) -> tuple:
    """Return (dQ, dK, dV).

    L is the (B, H_q, N_q) fp32 log-sum-exp tensor saved by forward.
    dO is the upstream gradient (same shape and dtype as O).
    """
    mod = _load_backward_kernels()
    B, H_q, N_q, D = Q.shape
    _, H_kv, N_kv, _ = K.shape
    group_size = H_q // H_kv

    # Encode mask / bias kinds from the graph
    masks  = {m.kind for m in graph.collect_masks()}
    biases = {b.kind for b in graph.collect_biases()}
    if MaskKind.CAUSAL in masks:
        mask_kind = 1
    else:
        mask_kind = 0
    bias_kind = 1 if BiasKind.ALIBI in biases else 0

    # ALiBi slopes
    if bias_kind == 1:
        from .dispatch import _alibi_slopes
        slopes = _alibi_slopes(H_q, str(Q.device), str(Q.dtype))
    else:
        from .dispatch import _placeholder_slopes
        slopes = _placeholder_slopes(str(Q.device), str(Q.dtype))

    # Output buffers
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    # D = rowsum(dO * O), shape (B, H_q, N_q), fp32
    D_buf = torch.empty(B, H_q, N_q, dtype=torch.float32, device=Q.device)

    # Per-shape tile selection (sweep-derived). The sweep override
    # (_BWD_TILE) wins if it's been explicitly set by a caller.
    tile = _pick_backward_tile(N_q) if _BWD_TILE.get("auto", True) else _BWD_TILE
    BLOCK_M    = tile["BLOCK_M"]
    BLOCK_N    = tile["BLOCK_N"]
    NUM_WARPS  = tile["num_warps"]
    NUM_STAGES = tile["num_stages"]

    # -------- Preproc: compute D = rowsum(dO * O) ---------------------------
    grid_pre = ((N_q + BLOCK_M - 1) // BLOCK_M, B * H_q)
    mod.attnfuse_bwd_preproc_kernel[grid_pre](
        O, dO, D_buf,
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        D_buf.stride(0), D_buf.stride(1), D_buf.stride(2),
        B, H_q, N_q,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        num_warps=NUM_WARPS,
    )

    # -------- dK / dV --------------------------------------------------------
    grid_dkv = ((N_kv + BLOCK_N - 1) // BLOCK_N, B * H_kv)
    mod.attnfuse_bwd_dkv_kernel[grid_dkv](
        Q, K, V, dO, L, D_buf,
        dK, dV,
        sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        D_buf.stride(0), D_buf.stride(1), D_buf.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        B, H_q, H_kv, N_q, N_kv,
        slopes,
        GROUP_SIZE=group_size,
        MASK_KIND=mask_kind,
        BIAS_KIND=bias_kind,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )

    # -------- dQ -------------------------------------------------------------
    grid_dq = ((N_q + BLOCK_M - 1) // BLOCK_M, B * H_q)
    mod.attnfuse_bwd_dq_kernel[grid_dq](
        Q, K, V, dO, L, D_buf,
        dQ,
        sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        L.stride(0), L.stride(1), L.stride(2),
        D_buf.stride(0), D_buf.stride(1), D_buf.stride(2),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        B, H_q, H_kv, N_q, N_kv,
        slopes,
        GROUP_SIZE=group_size,
        MASK_KIND=mask_kind,
        BIAS_KIND=bias_kind,
        HEAD_DIM=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )

    return dQ, dK, dV
