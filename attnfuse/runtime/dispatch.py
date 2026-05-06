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
from .kernel_cache import get_or_compile


# ---------------------------------------------------------------------------
# ALiBi slopes (Press et al., 2021)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _alibi_slopes(n_heads: int, device_str: str, dtype_str: str) -> torch.Tensor:
    """Return the canonical ALiBi slope per head as a 1-D tensor."""
    def power_of_two_slopes(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]

    if (n_heads & (n_heads - 1)) == 0:
        slopes = power_of_two_slopes(n_heads)
    else:
        closest = 1 << (n_heads - 1).bit_length() - 1
        slopes = power_of_two_slopes(closest)
        extra = power_of_two_slopes(2 * closest)[0::2][: n_heads - closest]
        slopes = slopes + extra

    dtype = getattr(torch, dtype_str.replace("torch.", ""))
    return torch.tensor(slopes, dtype=dtype, device=device_str)


@functools.lru_cache(maxsize=8)
def _placeholder_slopes(device_str: str, dtype_str: str) -> torch.Tensor:
    """Single-element tensor used as a dummy ALiBi pointer when BIAS_KIND != 1."""
    dtype = getattr(torch, dtype_str.replace("torch.", ""))
    return torch.empty(1, dtype=dtype, device=device_str)


@functools.lru_cache(maxsize=8)
def _placeholder_bias(device_str: str, dtype_str: str) -> torch.Tensor:
    """4-D placeholder used as a dummy bias pointer when BIAS_KIND != 2."""
    dtype = getattr(torch, dtype_str.replace("torch.", ""))
    return torch.empty(1, 1, 1, 1, dtype=dtype, device=device_str)


@functools.lru_cache(maxsize=8)
def _placeholder_rope(device_str: str, dtype_str: str) -> torch.Tensor:
    """2-D (1, 1) placeholder used as a dummy cos/sin pointer when ROPE_KIND != 1."""
    dtype = getattr(torch, dtype_str.replace("torch.", ""))
    return torch.empty(1, 1, dtype=dtype, device=device_str)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def run_attention(
    graph: Graph,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Launch the fused kernel and return the output tensor (B, H, N, D).

    Args:
        graph:  compiled attention graph (from @af.attention tracing).
        Q, K, V: real GPU tensors of shape (B, H, N, D).
        bias:   optional (B, H, N, N) additive bias tensor; required when
                the graph was built with ``af.additive_bias()``.
        cos, sin: optional RoPE tables of shape (N, D) or (1, 1, N, D);
                required when the graph was built with ``af.rope()``.
        out:    optional pre-allocated output tensor; allocated if None.
    """
    if not (Q.is_cuda and K.is_cuda and V.is_cuda):
        raise RuntimeError("AttnFuse requires Q/K/V on a CUDA device.")
    if Q.shape != K.shape or K.shape != V.shape:
        raise ValueError(
            f"Q/K/V shapes must match for self-attention; got {Q.shape}, {K.shape}, {V.shape}"
        )

    bundle = get_or_compile(graph)
    B, H, N, D = Q.shape

    if out is None:
        out = torch.empty_like(Q)

    grid = (triton_cdiv(N, bundle.block_m), B * H)

    # ALiBi slopes (BIAS_KIND == 1)
    if bundle.bias_kind == 1:
        slopes = _alibi_slopes(H, str(Q.device), str(Q.dtype))
    else:
        slopes = _placeholder_slopes(str(Q.device), str(Q.dtype))

    # External additive bias tensor (BIAS_KIND == 2)
    if bundle.has_additive_bias:
        if bias is None:
            raise RuntimeError(
                "This attention graph was built with af.additive_bias(); "
                "you must pass bias=<tensor of shape (B,H,N,N)> at call time."
            )
        b = bias
    else:
        b = _placeholder_bias(str(Q.device), str(Q.dtype))

    # Fused RoPE cos/sin tables (ROPE_KIND == 1)
    if bundle.has_rope:
        if cos is None or sin is None:
            raise RuntimeError(
                "This attention graph was built with af.rope(); "
                "you must pass cos=<table> and sin=<table> at call time."
            )
        # Squeeze to 2-D (N, D) if caller passed (1, 1, N, D)
        c = cos.squeeze(0).squeeze(0) if cos.dim() == 4 else cos
        sn = sin.squeeze(0).squeeze(0) if sin.dim() == 4 else sin
    else:
        c = _placeholder_rope(str(Q.device), str(Q.dtype))
        sn = _placeholder_rope(str(Q.device), str(Q.dtype))

    bundle.jit_fn[grid](
        Q, K, V, out,
        bundle.sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, N,
        slopes,
        b,
        b.stride(0), b.stride(1), b.stride(2), b.stride(3),
        c, sn,
        c.stride(0), c.stride(1),
        **bundle.cexprs,
        **bundle.meta,
    )
    return out


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b
