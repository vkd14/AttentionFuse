"""Runtime: kernel cache + dispatch entry point."""
from .dispatch import run_attention
from .kernel_cache import get_or_compile, clear_cache

__all__ = ["run_attention", "get_or_compile", "clear_cache"]
