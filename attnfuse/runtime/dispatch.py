"""Dispatch: take a Graph + real Q/K/V tensors and launch the compiled kernel.

This is the only place AttnFuse touches CUDA. We compute strides, lazily
build the ALiBi-slope table when needed, allocate O, and call into the
Triton kernel produced by the codegen pass.
"""
from __future__ import annotations

import functools
import math
from typing import Optional

import torch

from ..ir.high_level import Graph
from ..ir.tiled import TiledKernel
from ..compiler.codegen import kernel_constexprs, kernel_launch_meta
from .kernel_cache import get_or_compile


# ---------------------------------------------------------------------------
# ALiBi slopes (Press et al., 2021)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _alibi_slopes(n_heads: int, device_str: str, dtype_str: str) -> torch.Tensor:
    """Return the canonical ALiBi slope per head as a 1-D tensor.

    The standard recipe handles non-power-of-two head counts by interpolating
    between two power-of-two grids.
    """
    def power_of_two_slopes(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]

    if (n_heads & (n_heads - 1)) == 0:
        slopes = power_of_two_slopes(n_heads)
    else:
        closest = 1 << (n_heads - 1).bit_length() - 1  # largest power of two <= n_heads
        slopes = power_of_two_slopes(closest)
        extra = power_of_two_slopes(2 * closest)[0::2][: n_heads - closest]
        slopes = slopes + extra

    dtype = getattr(torch, dtype_str.replace("torch.", ""))
    return torch.tensor(slopes, dtype=dtype, device=device_str)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def run_attention(
    graph: Graph,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Launch the fused kernel and return the output tensor (B, H, N, D)."""
    if not (Q.is_cuda and K.is_cuda and V.is_cuda):
        raise RuntimeError("AttnFuse requires Q/K/V on a CUDA device.")
    if Q.shape != K.shape or K.shape != V.shape:
        # Cross-attention with mismatched K-seqlen would relax this; today we
        # only support self-attention with equal shapes.
        raise ValueError(
            f"Q/K/V shapes must match for self-attention; got {Q.shape}, {K.shape}, {V.shape}"
        )

    kernel, jit_fn = get_or_compile(graph)
    return _launch(jit_fn, kernel, Q, K, V, out)


def _launch(jit_fn, kernel: TiledKernel, Q, K, V, out):
    B, H, N, D = Q.shape

    if out is None:
        out = torch.empty_like(Q)

    cexprs = kernel_constexprs(kernel)
    meta   = kernel_launch_meta(kernel)

    BLOCK_M = cexprs["BLOCK_M"]
    grid = (triton_cdiv(N, BLOCK_M), B * H)

    # ALiBi slope table (or a 1-elt placeholder if unused -- Triton requires
    # a real pointer regardless)
    if cexprs["BIAS_KIND"] == 1:
        slopes = _alibi_slopes(H, str(Q.device), str(Q.dtype))
    else:
        slopes = torch.empty(1, device=Q.device, dtype=Q.dtype)

    sm_scale = float(kernel.score_scale)

    jit_fn[grid](
        Q, K, V, out,
        sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, N,
        slopes,
        **cexprs,
        **meta,
    )
    return out


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b
