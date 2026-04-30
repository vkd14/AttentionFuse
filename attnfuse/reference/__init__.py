"""Reference attention implementations for the eval baselines.

`pytorch_naive`     -- eager-mode unfused attention (the AttnFuse target to beat)
`pytorch_sdpa`      -- torch.nn.functional.scaled_dot_product_attention (FA-2 backend)
`triton_flash`      -- manually written Triton FlashAttention reference
"""
from .pytorch_naive import naive_attention
from .pytorch_sdpa import sdpa_attention
from .triton_flash import flash_attention

__all__ = ["naive_attention", "sdpa_attention", "flash_attention"]
