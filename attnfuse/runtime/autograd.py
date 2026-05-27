"""PyTorch autograd integration for AttnFuse.

When the user calls an ``@af.attention`` function with any input that has
``requires_grad=True``, the decorator routes through ``AttnFuseFunction``
instead of the direct ``run_attention`` dispatch. ``AttnFuseFunction``:

  forward()
    - allocates an fp32 (B, H_q, N_q) log-sum-exp tensor L
    - calls the standard forward kernel with ``save_lse=L``
    - saves (Q, K, V, O, L, scale) and the graph for backward
    - returns O (with the same dtype as Q)

  backward(dO)
    - calls ``run_backward(graph, Q, K, V, O, L, dO, scale)``
    - returns (None_for_graph, dQ, dK, dV, None_for_bias_cos_sin)

This keeps the forward fast path (no autograd overhead) untouched for the
inference case; only training-time calls pay the L-allocation cost.
"""
from __future__ import annotations

from typing import Optional

import torch

from ..ir.high_level import Graph
from .dispatch import run_attention
from .backward import run_backward, can_backward


class AttnFuseFunction(torch.autograd.Function):
    """Autograd binding: forward saves L, backward calls run_backward."""

    @staticmethod
    def forward(ctx,
                graph: Graph,
                Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                bias: Optional[torch.Tensor],
                cos: Optional[torch.Tensor],
                sin: Optional[torch.Tensor]):
        # Verify the graph is in the supported backward scope; if not, fall
        # back to the inference path and remember that backward is unavailable.
        # (The wrapper checks can_backward first so we don't usually hit this.)
        if not can_backward(graph):
            raise RuntimeError(
                "AttnFuse backward currently supports only the subset "
                "{dense, causal} x {none, ALiBi} x softmax norm with no fused RoPE; "
                "got a graph outside that scope. Call .detach() on Q/K/V to use "
                "the inference path."
            )

        B, H_q, N_q, D = Q.shape
        L = torch.empty(B, H_q, N_q, dtype=torch.float32, device=Q.device)
        out = run_attention(graph, Q, K, V, bias=bias, cos=cos, sin=sin,
                            save_lse=L)

        ctx.graph = graph
        ctx.sm_scale = 1.0 / (D ** 0.5)
        ctx.save_for_backward(Q, K, V, out, L)
        return out

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        Q, K, V, O, L = ctx.saved_tensors
        dO = dO.contiguous()
        dQ, dK, dV = run_backward(ctx.graph, Q, K, V, O, L, dO, ctx.sm_scale)
        # Return values must match the forward inputs:
        # (graph, Q, K, V, bias, cos, sin)  -> 7 return slots
        return None, dQ, dK, dV, None, None, None
