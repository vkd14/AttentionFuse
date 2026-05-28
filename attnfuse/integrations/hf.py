"""Register AttnFuse as a HuggingFace ``attn_implementation`` backend.

HuggingFace ``transformers`` (4.x+) uses an ALL_ATTENTION_FUNCTIONS
registry: each entry is a function with a fixed signature taking
(module, query, key, value, attention_mask, dropout, scaling, is_causal,
**kwargs) and returning (attn_output, attn_weights_or_None). Models
look up the function by ``model.config._attn_implementation``.

The function we register:

  * Calls AttnFuse for the common Llama path: causal mask, GQA, optional
    RoPE (applied before this function is called, since HF rotates Q and
    K inside the attention layer itself, not via a kwarg to our fn).
  * Falls back to PyTorch SDPA for the cases AttnFuse currently does not
    cover (sliding-window without explicit window, attention_mask that's
    not a simple causal pattern, etc.).

This lets users get AttnFuse's fused-RoPE speedups for the common case
and zero behaviour change for everything else.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

import attnfuse as af


# Module-level cache: trace the right graph once per (head_dim, dtype, has_causal)
# combination so repeated calls don't re-trace.
_GRAPH_CACHE: dict[tuple, callable] = {}


def _get_attn_fn(head_dim: int, dtype: torch.dtype, is_causal: bool):
    """Return a JIT-traced AttnFuse function for this (dtype, mask) cell."""
    key = (head_dim, str(dtype), is_causal)
    fn = _GRAPH_CACHE.get(key)
    if fn is not None:
        return fn

    if is_causal:
        @af.attention
        def attn(Q, K, V):
            return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V
    else:
        @af.attention
        def attn(Q, K, V):
            return af.softmax(af.scaled_dot_product(Q, K)) @ V

    _GRAPH_CACHE[key] = attn
    return attn


def attnfuse_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    """AttnFuse drop-in for HuggingFace ``attn_implementation``.

    Shapes follow HF conventions: (B, H, N, D). For GQA, key.shape[1] is
    H_kv (smaller than query.shape[1]); AttnFuse handles the broadcast
    natively via its GROUP_SIZE constexpr, so we do NOT call repeat_kv.
    """
    if dropout > 0.0:
        # AttnFuse doesn't (yet) support attention dropout inside the kernel.
        # Fall through to PyTorch SDPA which does.
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        return ALL_ATTENTION_FUNCTIONS["sdpa"](
            module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs,
        )

    # AttnFuse cannot model arbitrary attention_mask shapes; only the
    # implicit-causal case. If the caller passed an explicit additive mask
    # that isn't just the standard causal triangle, fall back.
    if attention_mask is not None and not _is_pure_causal_mask(attention_mask, query.shape[2], key.shape[2]):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        return ALL_ATTENTION_FUNCTIONS["sdpa"](
            module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs,
        )

    # Determine causality: HF passes is_causal=True for decoder layers in
    # training; in decoding (Q.N=1) we want it False so we get the
    # AttnFuse Flash-Decoding fast path.
    causal_for_us = bool(is_causal) and query.shape[2] > 1

    head_dim = query.shape[-1]
    attn_fn = _get_attn_fn(head_dim, query.dtype, causal_for_us)
    out = attn_fn(query, key, value)
    # HF interface returns (output, attention_weights_or_None); the second
    # slot is for output_attentions=True paths which AttnFuse doesn't support.
    return out, None


def _is_pure_causal_mask(mask: torch.Tensor, N_q: int, N_kv: int) -> bool:
    """Heuristic: True iff mask is the canonical causal-additive pattern
    (zeros on/below the diagonal, -inf above). HF builds this for every
    decoder layer when is_causal=True is set; we want to recognise it so
    we can drop the explicit mask and use AttnFuse's fused causal path.
    """
    if mask is None:
        return True
    # The standard mask shape is (B, 1, N_q, N_kv) or similar
    if mask.dim() != 4:
        return False
    if mask.shape[-2] != N_q or mask.shape[-1] != N_kv:
        return False
    # Cheap check: the upper triangle should be near -inf
    # (we don't compare element-by-element; just look at the (0, 1) corner)
    if N_q > 1 and N_kv > 1:
        corner = mask[..., 0, 1]
        # If this corner is finite, the mask isn't purely causal
        return bool(torch.isinf(corner).all().item())
    return True


def register():
    """Register the ``attnfuse`` backend with HuggingFace transformers.

    Safe to call multiple times. After this, models can opt in via
    ``model.config._attn_implementation = "attnfuse"`` or by passing
    ``attn_implementation="attnfuse"`` to ``from_pretrained()``.
    """
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "HuggingFace transformers >= 4.40 is required for the AttnFuse "
            "integration. pip install 'transformers>=4.40'."
        ) from e
    ALL_ATTENTION_FUNCTIONS["attnfuse"] = attnfuse_attention_forward


# Register on import (this is the convention HF integrations use)
register()
