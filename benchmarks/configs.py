"""Static configuration: model shapes and the seqlen sweep."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelShape:
    name: str
    num_heads: int
    head_dim: int
    is_causal: bool


# BERT-base: 12 heads × 64 dim, bidirectional dense attention
BERT_BASE = ModelShape("bert-base", num_heads=12, head_dim=64, is_causal=False)

# GPT-2 small: 12 heads × 64 dim, causal autoregressive attention
GPT2_SMALL = ModelShape("gpt2-small", num_heads=12, head_dim=64, is_causal=True)


SEQLENS = (512, 1024, 2048, 4096)
BATCH_SIZE = 4              # fits 24GB at seqlen=4096, head_dim=64, fp16
WARMUP = 5                  # CUDA graph warmup launches
ITERS = 50                  # measured launches (median is reported)

# Variants we sweep in the full eval
VARIANTS = ("dense", "causal", "sliding_window", "causal_alibi")
SLIDING_WINDOW_SIZE = 256
