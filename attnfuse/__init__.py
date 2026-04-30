"""AttnFuse: an embedded Python DSL for attention that compiles to fused Triton kernels.

Public surface (re-exported from `attnfuse.dsl.api`):

    @attention                          decorator
    scaled_dot_product, additive_bias   score combinators
    causal, sliding_window, full        mask combinators
    alibi                               positional-bias combinator
    softmax, relu_attention             normalisation combinators

Usage:

    import attnfuse as af

    @af.attention
    def gpt2_attn(Q, K, V):
        s = af.scaled_dot_product(Q, K)
        s = af.causal(s)
        p = af.softmax(s)
        return p @ V

    out = gpt2_attn(q, k, v)   # JIT-compiles a fused Triton kernel on first call.
"""

from .dsl.api import (
    attention,
    scaled_dot_product,
    additive_bias,
    causal,
    sliding_window,
    full,
    alibi,
    softmax,
    relu_attention,
)

__version__ = "0.1.0"

__all__ = [
    "attention",
    "scaled_dot_product",
    "additive_bias",
    "causal",
    "sliding_window",
    "full",
    "alibi",
    "softmax",
    "relu_attention",
    "__version__",
]
