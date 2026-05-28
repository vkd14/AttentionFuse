"""Integrations with downstream libraries.

The HuggingFace ``transformers`` integration registers AttnFuse as an
``attn_implementation`` so any Llama / Mistral / Falcon-class model can
use AttnFuse just by passing ``attn_implementation="attnfuse"`` to
``from_pretrained()`` or ``LlamaConfig``.

Example:

    import transformers
    import attnfuse.integrations.hf            # registers the backend

    cfg = transformers.LlamaConfig(...)
    model = transformers.LlamaForCausalLM(cfg)
    model.config._attn_implementation = "attnfuse"
"""
