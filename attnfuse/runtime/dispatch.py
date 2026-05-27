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
    save_lse: Optional[torch.Tensor] = None,   # if set, write L=m+log(l) into this (B,H_q,N_q) fp32 buffer
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
    # Allow GQA/MQA: Q has H_q heads, K and V have H_kv heads, H_q % H_kv == 0.
    # Also allow N_q != N_kv (cross-attention / KV-cache decoding) when the
    # graph has no mask (mask kinds depend on relative q-k positions).
    if K.shape != V.shape:
        raise ValueError(
            f"K and V must have the same shape; got {K.shape}, {V.shape}"
        )
    if Q.shape[0] != K.shape[0] or Q.shape[3] != K.shape[3]:
        raise ValueError(
            f"Q and K/V must agree on (batch, head_dim); got Q={Q.shape}, K={K.shape}"
        )
    H_q, H_kv = Q.shape[1], K.shape[1]
    if H_q % H_kv != 0:
        raise ValueError(
            f"Q heads ({H_q}) must be a multiple of KV heads ({H_kv}) for GQA"
        )
    group_size = H_q // H_kv
    N_q, N_kv = Q.shape[2], K.shape[2]

    bundle = get_or_compile(graph)
    B, H, N, D = Q.shape

    # Flash Decoding fast path: when Q.N = 1 and N_kv is large enough that
    # splitting the KV axis saturates the GPU's SMs better than launching
    # one program per (b, h_q). Restricted to the no-mask + (none / ALiBi
    # bias) + no-RoPE subset, which covers the common autoregressive
    # decode pattern. Skipped when the caller wants the L tensor (backward
    # path) because the decode kernels don't currently save it.
    from .flash_decode import can_use_flash_decode, run_flash_decode
    if save_lse is None and can_use_flash_decode(graph, Q, K):
        if out is None:
            out = torch.empty_like(Q)
        return run_flash_decode(graph, Q, K, V, bundle.sm_scale,
                                bundle.bias_kind, K.shape[1], out)

    # Cross-attention guard: causal and sliding-window only make sense when
    # N_q == N_kv (relative-position semantics require equal-length axes).
    if N_q != N_kv:
        from ..ir.high_level import MaskKind
        masks = {m.kind for m in graph.collect_masks()}
        if masks - {MaskKind.FULL}:
            raise ValueError(
                f"Cross-attention (N_q={N_q} != N_kv={N_kv}) is only supported "
                f"for full/dense masks; the graph uses {sorted(m.value for m in masks)}. "
                f"Use af.softmax(af.scaled_dot_product(Q, K)) @ V for KV-cache decoding."
            )

    if out is None:
        out = torch.empty_like(Q)

    grid = (triton_cdiv(N_q, bundle.block_m), B * H)

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

    if save_lse is not None:
        L = save_lse
        save_l_flag = 1
    else:
        L = _placeholder_rope(str(Q.device), "float32")  # tiny dummy buffer (fp32)
        save_l_flag = 0

    cexprs = dict(bundle.cexprs)
    cexprs["SAVE_L"] = save_l_flag

    bundle.jit_fn[grid](
        Q, K, V, out,
        bundle.sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, N_q, N_kv,
        group_size,
        slopes,
        b,
        b.stride(0), b.stride(1), b.stride(2), b.stride(3),
        c, sn,
        c.stride(0), c.stride(1),
        L,
        L.stride(0), L.stride(1) if L.dim() >= 2 else 1, L.stride(-1),
        **cexprs,
        **bundle.meta,
    )
    return out


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b
