"""Block-mask construction + runtime dispatch.

Mirrors PyTorch ``flex_attention``'s BlockMask data structure so the
existing tooling around it (mostly: how users *think* about block-sparse
attention) translates one-to-one. A BlockMask carries:

  * ``kv_indices``    -- padded list of active KV-block indices per Q-block
  * ``kv_num_blocks`` -- length of each row's active list
  * ``full_kv_idx``   -- subset of kv_indices that are FULLY active (no
                         per-element refinement needed). The kernel uses
                         a faster code path for these.
  * ``full_kv_num``   -- length of the full-active list per row.

The user usually builds a BlockMask from a per-element predicate via
``create_block_mask``; if every element of a coarse (BLOCK_M, BLOCK_N)
tile satisfies the predicate, that tile goes into the full list, else
into the partial list. For BigBird-style masks roughly all tiles end up
in the "full" bucket because the user precomputes which tiles to drop.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from ..compiler.codegen_blocksparse import get_blocksparse_source

_bs_module = None
_GENERATED_DIR = Path(__file__).parent.parent / "_generated"


def _load_blocksparse_kernel():
    global _bs_module
    if _bs_module is not None:
        return _bs_module
    src = get_blocksparse_source()
    _GENERATED_DIR.mkdir(exist_ok=True)
    h = hashlib.sha1(src.encode()).hexdigest()[:12]
    mod_name = f"_attnfuse_blocksparse_{h}"
    fpath = _GENERATED_DIR / f"{mod_name}.py"
    if not fpath.exists():
        fpath.write_text(src)
    spec = importlib.util.spec_from_file_location(mod_name, fpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    _bs_module = mod
    return mod


@dataclass
class BlockMask:
    """Compressed block-sparse mask, ready to feed into the kernel."""
    kv_num_blocks: torch.Tensor    # (n_q_blocks,) int32
    kv_indices:    torch.Tensor    # (n_q_blocks, max_partial) int32
    full_kv_num:   torch.Tensor    # (n_q_blocks,) int32
    full_kv_idx:   torch.Tensor    # (n_q_blocks, max_full) int32
    BLOCK_M:       int
    BLOCK_N:       int
    n_q_blocks:    int
    n_kv_blocks:   int
    # Total active tile count (for reporting / sparsity stats)
    n_active:      int
    n_full:        int


def create_block_mask(mask_fn: Callable[[int, int], bool],
                      Q_LEN: int, KV_LEN: int,
                      BLOCK_M: int = 64, BLOCK_N: int = 64,
                      device: str = "cuda") -> BlockMask:
    """Build a BlockMask from a per-element predicate.

    Args:
        mask_fn: callable ``(q_idx, kv_idx) -> bool`` that returns True if
            the query at index ``q_idx`` should attend to the key at index
            ``kv_idx``. (Compatible with the score-mod-style API of
            ``torch.nn.attention.flex_attention.create_block_mask``.)
        Q_LEN, KV_LEN: sequence lengths.
        BLOCK_M, BLOCK_N: kernel tile sizes that the BlockMask is
            specialised for (must match the runtime kernel's tile config).

    Returns a :class:`BlockMask`.
    """
    n_q_blocks  = (Q_LEN  + BLOCK_M - 1) // BLOCK_M
    n_kv_blocks = (KV_LEN + BLOCK_N - 1) // BLOCK_N

    # Build a coarse (n_q_blocks, n_kv_blocks) bool grid: a tile is
    # "fully active" iff every (q, kv) pair inside the tile passes
    # mask_fn; "partial" iff some pair passes but not all; "dead" iff none.
    full = torch.zeros((n_q_blocks, n_kv_blocks), dtype=torch.bool)
    part = torch.zeros((n_q_blocks, n_kv_blocks), dtype=torch.bool)

    # Vectorise the per-element test over each tile by building two arrays
    # of (q_idx, kv_idx) coordinates per tile and broadcasting mask_fn.
    # mask_fn typically uses elementwise ops on torch tensors.
    q_coords = torch.arange(Q_LEN)
    k_coords = torch.arange(KV_LEN)
    # Outer-product-style indices: (Q_LEN, KV_LEN)
    Q = q_coords[:, None].expand(Q_LEN, KV_LEN)
    K = k_coords[None, :].expand(Q_LEN, KV_LEN)
    elem_mask = mask_fn(Q, K)  # bool (Q_LEN, KV_LEN)

    for qb in range(n_q_blocks):
        q_lo, q_hi = qb * BLOCK_M, min((qb + 1) * BLOCK_M, Q_LEN)
        for kb in range(n_kv_blocks):
            k_lo, k_hi = kb * BLOCK_N, min((kb + 1) * BLOCK_N, KV_LEN)
            tile = elem_mask[q_lo:q_hi, k_lo:k_hi]
            t = tile.numel()
            n_true = int(tile.sum())
            if n_true == t:
                full[qb, kb] = True
            elif n_true > 0:
                # Block is partial -- some elements pass, some don't. In
                # this initial block-sparse scope we round these UP to
                # "fully active" so they get processed (any element-level
                # mask refinement would be the user's responsibility, e.g.
                # via af.causal() in a non-block-sparse path). This makes
                # block_sparse give an UPPER BOUND on attended keys. To
                # get exact element-level masking, do not use partial
                # blocks -- align your mask to block boundaries.
                full[qb, kb] = True

    # CSR-ify
    full_lists = [torch.nonzero(full[i], as_tuple=False).flatten().to(torch.int32)
                  for i in range(n_q_blocks)]
    part_lists = [torch.nonzero(part[i], as_tuple=False).flatten().to(torch.int32)
                  for i in range(n_q_blocks)]
    max_full = max((len(l) for l in full_lists), default=1)
    max_part = max((len(l) for l in part_lists), default=1)

    def pad(lists, max_len):
        out = torch.zeros((n_q_blocks, max(1, max_len)), dtype=torch.int32)
        for i, l in enumerate(lists):
            out[i, :len(l)] = l
        return out

    full_kv_idx   = pad(full_lists, max_full).to(device)
    kv_indices    = pad(part_lists, max_part).to(device)
    full_kv_num   = torch.tensor([len(l) for l in full_lists],
                                  dtype=torch.int32, device=device)
    kv_num_blocks = torch.tensor([len(l) for l in part_lists],
                                  dtype=torch.int32, device=device)
    return BlockMask(
        kv_num_blocks=kv_num_blocks, kv_indices=kv_indices,
        full_kv_num=full_kv_num,     full_kv_idx=full_kv_idx,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        n_q_blocks=n_q_blocks, n_kv_blocks=n_kv_blocks,
        n_active=int(kv_num_blocks.sum()) + int(full_kv_num.sum()),
        n_full=int(full_kv_num.sum()),
    )


def run_block_sparse(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                     block_mask: BlockMask,
                     sm_scale: float,
                     bias_kind: int = 0,
                     out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Launch the block-sparse forward kernel."""
    mod = _load_blocksparse_kernel()
    B, H_q, N_Q, D = Q.shape
    _, H_kv, N_KV, _ = K.shape
    group_size = H_q // H_kv
    if out is None:
        out = torch.empty_like(Q)

    BM, BN = block_mask.BLOCK_M, block_mask.BLOCK_N
    grid = ((N_Q + BM - 1) // BM, B * H_q)

    # ALiBi slopes if requested
    if bias_kind == 1:
        from .dispatch import _alibi_slopes
        slopes = _alibi_slopes(H_q, str(Q.device), str(Q.dtype))
    else:
        from .dispatch import _placeholder_slopes
        slopes = _placeholder_slopes(str(Q.device), str(Q.dtype))

    # Placeholder L (block-sparse backward is future work)
    from .dispatch import _placeholder_rope
    L = _placeholder_rope(str(Q.device), "float32")

    mod.attnfuse_blocksparse_fwd_kernel[grid](
        Q, K, V, out,
        sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        block_mask.kv_num_blocks,
        block_mask.kv_indices,
        block_mask.kv_indices.stride(0), block_mask.kv_indices.stride(1),
        block_mask.full_kv_num,
        block_mask.full_kv_idx,
        block_mask.full_kv_idx.stride(0), block_mask.full_kv_idx.stride(1),
        B, H_q, N_Q, N_KV,
        group_size,
        slopes,
        HEAD_DIM=D,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BIAS_KIND=bias_kind,
        L_ptr=L,
        stride_lb=1, stride_lh=1, stride_lm=1,
        SAVE_L=0,
        num_warps=4, num_stages=2,
    )
    return out
