"""Process-local cache for compiled Triton kernels, keyed by graph signature."""
from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..ir.high_level import Graph
from ..ir.tiled import TiledKernel
from ..compiler.lowering import lower_to_tiled
from ..compiler.codegen import generate_triton_source, kernel_constexprs, kernel_launch_meta

_lock = threading.Lock()
_cache: dict[str, "LaunchBundle"] = {}

_GENERATED_DIR = Path(__file__).parent.parent / "_generated"


@dataclass
class LaunchBundle:
    """Everything needed to launch the kernel — precomputed once at compile time."""
    kernel:   TiledKernel
    jit_fn:   Callable
    cexprs:   dict      # constexpr kwargs (BLOCK_M, MASK_KIND, …)
    meta:     dict      # launch hints (num_warps, num_stages)
    sm_scale: float
    block_m:  int
    bias_kind: int      # cached to avoid dict lookup in hot path


def _materialise(src: str) -> Callable:
    """Write kernel source to a file and import it so Triton can inspect it."""
    _GENERATED_DIR.mkdir(exist_ok=True)
    h = hashlib.sha1(src.encode()).hexdigest()[:12]
    mod_name = f"_attnfuse_kernel_{h}"
    fpath = _GENERATED_DIR / f"{mod_name}.py"
    if not fpath.exists():
        fpath.write_text(src)
    spec = importlib.util.spec_from_file_location(mod_name, fpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod.attnfuse_fwd_kernel


def get_or_compile(graph: Graph) -> LaunchBundle:
    """Return a LaunchBundle for `graph`, compiling on cache miss."""
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
        cexprs = kernel_constexprs(kernel)
        meta   = kernel_launch_meta(kernel)
        sm_scale = float(kernel.score_scale)
        bundle = LaunchBundle(
            kernel=kernel, jit_fn=jit_fn,
            cexprs=cexprs, meta=meta,
            sm_scale=sm_scale,
            block_m=cexprs["BLOCK_M"],
            bias_kind=cexprs["BIAS_KIND"],
        )
        _cache[key] = bundle
        return bundle


def clear_cache() -> None:
    with _lock:
        _cache.clear()
