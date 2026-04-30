"""PyTorch SDPA wrapper. On Ampere+, this dispatches to FlashAttention-2.

Note: SDPA does NOT support sliding-window or ALiBi natively in any released
PyTorch as of 2024-25. For those variants this baseline falls back to the
naive path with a warning.
"""
from __future__ import annotations

import warnings
import torch
import torch.nn.functional as F


def sdpa_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    causal: bool = False,
    window: int | None = None,
    alibi_slopes: torch.Tensor | None = None,
) -> torch.Tensor:
    if window is not None or alibi_slopes is not None:
        # Fall through to naive path; SDPA can't express these directly.
        from .pytorch_naive import naive_attention
        warnings.warn(
            "SDPA does not natively support sliding-window/ALiBi; falling back "
            "to naive PyTorch for fair comparison.",
            stacklevel=2,
        )
        return naive_attention(
            Q, K, V, causal=causal, window=window, alibi_slopes=alibi_slopes,
        )

    return F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
