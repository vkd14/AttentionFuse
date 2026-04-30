"""Naive eager-mode attention: 5 separate kernel launches, materialises (B,H,N,N).

This is the slow-and-memory-hungry baseline AttnFuse aims to beat by ≥1.5×.
"""
from __future__ import annotations

import math
import torch


def naive_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    causal: bool = False,
    window: int | None = None,
    alibi_slopes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Q/K/V: (B, H, N, D)."""
    B, H, N, D = Q.shape
    scale = 1.0 / math.sqrt(D)

    # (B, H, N, N)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    if alibi_slopes is not None:
        i = torch.arange(N, device=Q.device)
        dist = (i[:, None] - i[None, :]).abs().to(Q.dtype)            # (N, N)
        scores = scores + (-alibi_slopes.view(1, H, 1, 1) * dist)

    if causal:
        mask = torch.ones(N, N, dtype=torch.bool, device=Q.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))

    if window is not None:
        i = torch.arange(N, device=Q.device)
        d = (i[:, None] - i[None, :])
        wmask = (d.abs() < window)
        scores = scores.masked_fill(~wmask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, V)
