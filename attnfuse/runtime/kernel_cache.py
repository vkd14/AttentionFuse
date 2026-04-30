"""Process-local cache for compiled Triton kernels, keyed by graph signature."""
from __future__ import annotations

import os
import threading
from typing import Callable

from ..ir.high_level import Graph
from ..ir.tiled import TiledKernel
from ..compiler.lowering import lower_to_tiled
from ..compiler.codegen import generate_triton_source

_lock = threading.Lock()
_cache: dict[str, tuple[TiledKernel, Callable]] = {}


def _materialise(src: str) -> Callable:
    """Exec the kernel source in a fresh namespace and return the @triton.jit fn."""
    ns: dict = {}
    exec(src, ns)
    return ns["attnfuse_fwd_kernel"]


def get_or_compile(graph: Graph) -> tuple[TiledKernel, Callable]:
    """Return (TiledKernel, compiled_jit_fn) for `graph`, compiling on cache miss."""
    key = graph.signature()
    with _lock:
        if key in _cache:
            return _cache[key]
        kernel = lower_to_tiled(graph)
        src = generate_triton_source(kernel)
        if os.environ.get("ATTNFUSE_DEBUG"):
            from ..ir.printer import format_tiled
            print("[AttnFuse] tiled IR:")
            print(format_tiled(kernel))
            print("[AttnFuse] generated Triton source (truncated):")
            print(src[:1500] + ("...\n" if len(src) > 1500 else ""))
        jit_fn = _materialise(src)
        _cache[key] = (kernel, jit_fn)
        return _cache[key]


def clear_cache() -> None:
    with _lock:
        _cache.clear()
