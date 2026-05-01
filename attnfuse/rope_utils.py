"""Rotary Position Embedding (RoPE) utilities for use with AttnFuse.

RoPE (Su et al., 2021) rotates query and key vectors by position-dependent
angles before the dot-product.  Because the rotation is applied per-token
before attention, it can be fused as a pre-processing step: rotate Q and K
in Python, then pass the rotated tensors to an AttnFuse kernel.

Full kernel-side fusion (rotating Q/K tiles inside the Triton kernel) is a
planned extension; the current helper demonstrates DSL composability without
requiring a kernel change.

Usage::

    from attnfuse.rope_utils import build_rope_cache, apply_rope
    import attnfuse as af

    cos, sin = build_rope_cache(seqlen=N, head_dim=D, device=Q.device, dtype=Q.dtype)

    @af.attention
    def causal_rope_attn(Q, K, V):
        return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V

    Q_rot = apply_rope(Q, cos, sin)
    K_rot = apply_rope(K, cos, sin)
    out   = causal_rope_attn(Q_rot, K_rot, V)
"""
from __future__ import annotations

import math
import torch


def build_rope_cache(
    seqlen: int,
    head_dim: int,
    base: float = 10_000.0,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute (cos, sin) tables of shape (1, 1, seqlen, head_dim).

    The standard LLaMA/GPT-NeoX convention:
        θ_i = base^(-2i/d)   for i in 0..d/2-1
        cos[t, 2i]   = cos[t, 2i+1]   = cos(t * θ_i)
        sin[t, 2i]   = sin[t, 2i+1]   = sin(t * θ_i)
    """
    half = head_dim // 2
    theta = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=torch.float32) * 2 / head_dim))
    t     = torch.arange(seqlen, device=device, dtype=torch.float32)
    freqs = torch.outer(t, theta)                 # (seqlen, half)
    emb   = torch.cat([freqs, freqs], dim=-1)     # (seqlen, head_dim)
    cos   = emb.cos().to(dtype).unsqueeze(0).unsqueeze(0)  # (1,1,N,D)
    sin   = emb.sin().to(dtype).unsqueeze(0).unsqueeze(0)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by 90°: [-x2, x1] where x = [x1 | x2]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings to x (B, H, N, D).

    cos/sin can be (1, 1, N, D) or (N, D) — will be broadcast.
    """
    return x * cos + _rotate_half(x) * sin
