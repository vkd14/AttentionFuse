"""Tile-configuration ablation study.

Sweeps BLOCK_M, BLOCK_N, num_warps, num_stages for dense and causal attention
at a fixed sequence length, measuring TFLOPS to justify the default Ampere
tile table in attnfuse/compiler/tiling.py.

Usage::

    python -m benchmarks.ablation [--seqlen 2048] [--dtype float16] [--output results/ablation.csv]

Supported dtypes: float16 (default), bfloat16, float32.
The default run takes ~3 minutes on an RTX 3090.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import statistics
import time
from pathlib import Path

import torch

import attnfuse as af
from attnfuse.ir.tiled import TileConfig, TiledKernel
from attnfuse.ir.high_level import MaskKind, NormKind
from attnfuse.compiler.codegen import generate_triton_source, kernel_constexprs, kernel_launch_meta
from attnfuse.runtime.kernel_cache import _materialise
from attnfuse.runtime.dispatch import (
    _alibi_slopes, _placeholder_slopes, _placeholder_bias, _placeholder_rope, triton_cdiv,
)


BLOCK_MS    = [64, 128]
BLOCK_NS    = [32, 64]
NUM_WARPS   = [4, 8]
NUM_STAGES  = [2, 3, 4]

SEQLEN      = 2048
HEAD_DIM    = 64
BATCH       = 4
NUM_HEADS   = 12
WARMUP      = 10
ITERS       = 30

_DTYPE_MAP = {
    "float16":  torch.float16,
    "bfloat16": torch.bfloat16,
    "float32":  torch.float32,
}


def _make_kernel(
    mask_kind: MaskKind, bm: int, bn: int, nw: int, ns: int, dtype_str: str,
) -> TiledKernel:
    cfg = TileConfig(
        BLOCK_M=bm, BLOCK_N=bn,
        num_warps=nw, num_stages=ns,
        skip_full_mask_blocks=True,
    )
    return TiledKernel(
        head_dim=HEAD_DIM,
        dtype=dtype_str,
        score_scale=1.0 / math.sqrt(HEAD_DIM),
        mask_kind=mask_kind,
        mask_window=None,
        bias_kind=None,
        bias_num_heads=None,
        norm_kind=NormKind.SOFTMAX,
        rope_kind=0,
        config=cfg,
        cache_key=f"ablation_{mask_kind.value}_{dtype_str}_{bm}_{bn}_{nw}_{ns}",
    )


def _bench_kernel(kernel: TiledKernel, Q, K, V) -> float | None:
    """Return median TFLOPS or None on OOM/compile error."""
    try:
        src    = generate_triton_source(kernel)
        jit_fn = _materialise(src)
        cexprs = kernel_constexprs(kernel)
        meta   = kernel_launch_meta(kernel)
    except Exception as e:
        print(f"    compile error: {e}")
        return None

    B, H, N, D = Q.shape
    out    = torch.empty_like(Q)
    grid   = (triton_cdiv(N, kernel.config.BLOCK_M), B * H)
    slopes = _placeholder_slopes(str(Q.device), str(Q.dtype))
    b_bias = _placeholder_bias(str(Q.device), str(Q.dtype))
    c_rope = _placeholder_rope(str(Q.device), str(Q.dtype))

    def run():
        jit_fn[grid](
            Q, K, V, out,
            kernel.score_scale,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            B, H, N,
            slopes,
            b_bias,
            b_bias.stride(0), b_bias.stride(1), b_bias.stride(2), b_bias.stride(3),
            c_rope, c_rope,
            c_rope.stride(0), c_rope.stride(1),
            **cexprs,
            **meta,
        )

    try:
        for _ in range(WARMUP):
            run()
        torch.cuda.synchronize()

        starters = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        stoppers  = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
        for s, e in zip(starters, stoppers):
            s.record(); run(); e.record()
        torch.cuda.synchronize()
        times_ms = [s.elapsed_time(e) for s, e in zip(starters, stoppers)]
        lat = statistics.median(times_ms)
        flops = 4.0 * B * H * N * N * D
        return flops / (lat * 1e-3) / 1e12
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        print(f"    runtime error: {e}")
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seqlen", type=int, default=SEQLEN)
    p.add_argument("--dtype",  default="float16",
                   choices=list(_DTYPE_MAP), help="Tensor dtype for the sweep")
    p.add_argument("--output", default="results/ablation.csv")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available"); return 1

    dtype     = _DTYPE_MAP[args.dtype]
    dtype_str = args.dtype
    N = args.seqlen
    g = torch.Generator(device="cuda").manual_seed(42)
    Q = torch.randn(BATCH, NUM_HEADS, N, HEAD_DIM, generator=g, device="cuda", dtype=dtype)
    K = torch.randn_like(Q); V = torch.randn_like(Q)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fout = open(args.output, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["variant", "dtype", "BLOCK_M", "BLOCK_N", "num_warps", "num_stages", "tflops"])

    print(f"# Ablation sweep  dtype={dtype_str}  N={N}  device={torch.cuda.get_device_name(0)}")
    print(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}")

    for variant, mask_kind in [("dense", MaskKind.FULL), ("causal", MaskKind.CAUSAL)]:
        for bm, bn, nw, ns in itertools.product(BLOCK_MS, BLOCK_NS, NUM_WARPS, NUM_STAGES):
            kernel  = _make_kernel(mask_kind, bm, bn, nw, ns, dtype_str)
            tfl     = _bench_kernel(kernel, Q, K, V)
            tfl_str = f"{tfl:.2f}" if tfl is not None else "oom"
            row = [variant, dtype_str, bm, bn, nw, ns, tfl_str]
            writer.writerow(row); fout.flush()
            print("  ".join(str(x) for x in row))

    fout.close()
    print(f"\n[ok] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
