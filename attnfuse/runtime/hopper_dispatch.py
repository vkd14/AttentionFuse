"""Runtime hook for the Hopper-targeted causal forward kernel.

This is the integration point that lets ``run_attention()`` route
qualifying calls through the spike kernel on H100/H200 (sm_90+). Mirrors
the structure of ``runtime/flash_decode.py`` so the dispatch site reads
identically.

Eligibility (all required):
  * GPU compute capability >= 9.0 (Hopper).
  * Graph has exactly one MaskOp with kind=CAUSAL.
  * Graph has no BiasOp (no ALiBi, no additive bias).
  * Fused RoPE is accepted (Session 6+). Other than RoPE, the graph
    must have no other ScoreOp transformation.
  * Norm is SOFTMAX.
  * Dtype is fp16 or bf16.
  * HEAD_DIM in {64, 128}.
  * N_q == N_kv (self-attention only -- causal requires it anyway).
  * N >= 2048: avoids the small-N regression where the spike loses to
    the production kernel by ~12%. At N=1024 there are only 8 m-blocks
    so the spike's larger setup cost is not amortised.
  * save_lse is None: the spike does not write the log-normaliser
    needed for backward. Training calls go through the production
    forward as before.
"""
from __future__ import annotations

from typing import Optional

import torch

from ..ir.high_level import BiasKind, Graph, MaskKind, NormKind, ScoreOp
from ..experimental.hopper_causal_fwd import hopper_causal_fwd


def _is_hopper() -> bool:
    try:
        if not torch.cuda.is_available():
            return False
        major, _ = torch.cuda.get_device_capability(0)
        return major >= 9
    except Exception:
        return False


_IS_HOPPER = _is_hopper()


def can_use_hopper_spike(graph: Graph, Q: torch.Tensor, K: torch.Tensor,
                          save_lse: Optional[torch.Tensor] = None) -> bool:
    """True if ``run_attention`` should dispatch this call to the spike."""
    if not _IS_HOPPER:
        return False
    if save_lse is not None:
        return False
    if Q.dtype not in (torch.float16, torch.bfloat16):
        return False

    B, H_q, N_q, D = Q.shape
    N_kv = K.shape[2]
    if D not in (64, 128):
        return False
    if N_q != N_kv:
        return False
    if N_q < 2048:
        return False

    # Mask must be exactly CAUSAL.
    masks = list(graph.collect_masks())
    if len(masks) != 1 or masks[0].kind is not MaskKind.CAUSAL:
        return False

    # No biases of any kind.
    if list(graph.collect_biases()):
        return False

    # RoPE is supported (Session 6). Nothing to reject here.
    _ = ScoreOp  # used for type clarity in the import above

    # Softmax norm only. graph.norm() returns the (single) NormOp.
    try:
        if graph.norm().kind is not NormKind.SOFTMAX:
            return False
    except Exception:
        return False

    return True


def run_hopper_spike(graph: Graph, Q: torch.Tensor, K: torch.Tensor,
                      V: torch.Tensor,
                      cos: Optional[torch.Tensor] = None,
                      sin: Optional[torch.Tensor] = None,
                      out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Launch the spike kernel. Caller guarantees ``can_use_hopper_spike``.

    If the graph uses fused RoPE, the caller must pass ``cos`` and ``sin``
    (same convention as ``run_attention``: 2-D (N, D) or 4-D (1, 1, N, D)).
    """
    # Detect whether this graph uses RoPE; if so, cos/sin must be provided.
    graph_has_rope = any(
        isinstance(node, ScoreOp) and node.rope for node, _ in graph.walk()
    )
    if graph_has_rope and (cos is None or sin is None):
        raise RuntimeError(
            "Hopper spike: graph uses af.rope() but cos/sin were not passed "
            "to run_attention(). The dispatch site should forward them."
        )

    if graph_has_rope:
        result = hopper_causal_fwd(Q, K, V, cos=cos, sin=sin)
    else:
        result = hopper_causal_fwd(Q, K, V)

    if out is None:
        return result
    out.copy_(result)
    return out
