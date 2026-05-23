"""Per-variant tile-config sweep.

Goal: find the (BLOCK_M, BLOCK_N, num_warps, num_stages) tuple that
minimises latency for each (variant, seqlen, dtype) combination on the
target hardware. Results are written as CSV; the winners are intended
to be promoted into ``attnfuse/compiler/tiling.py``.

Why this exists
---------------
The default Ampere tile table in ``tiling.py`` was tuned offline on a
single shape (gpt2-small dense, N=2048). Sliding-window and ALiBi have
different memory-access patterns and may want a different config.
Comparing against ``flex_attention`` (which uses TorchInductor's
autotuner) suggests our hand-picked configs are leaving 10-15% on the
table.

What this script does
---------------------
For each variant we monkeypatch ``choose_tile_config`` to return a
specific config, clear the LaunchBundle cache, JIT-compile, run a
warmup, then time. We sweep a small grid of plausible configs and
report the best.

Usage::

    python -m benchmarks.config_sweep [--output results/config_sweep.csv]
"""
from __future__ import annotations

import argparse
import csv
import gc
import math
import statistics
import time
from pathlib import Path

import torch
import attnfuse as af
from attnfuse.compiler import tiling as _tiling
from attnfuse.ir.tiled import TileConfig
from attnfuse.runtime import kernel_cache

BATCH      = 4
NUM_HEADS  = 12
HEAD_DIM   = 64
WINDOW     = 256
DTYPE      = torch.float16
WARMUP     = 10
ITERS      = 50

# Plausible config grid for Ampere fp16/bf16, HEAD_DIM=64.
# SMEM budget on RTX 3090 = ~101 KB usable per block.
# Per-stage SMEM ~= (BLOCK_M + 2*BLOCK_N) * HEAD_DIM * 2 bytes  (Q+K+V tiles)
# At BLOCK_M=128, BLOCK_N=128, HEAD_DIM=64: 49 KB / stage.  3 stages = 147 KB (OOM)
#                                             2 stages =  98 KB (fits, tight)
# At BLOCK_M=128, BLOCK_N=64                : 32 KB / stage.  3 stages =  96 KB (fits)
_CONFIG_GRID: list[TileConfig] = [
    # (BLOCK_M, BLOCK_N, num_warps, num_stages)
    # Small tiles (good for sparse variants)
    TileConfig(BLOCK_M=64,  BLOCK_N=32,  num_warps=4, num_stages=3),
    TileConfig(BLOCK_M=64,  BLOCK_N=32,  num_warps=4, num_stages=4),
    TileConfig(BLOCK_M=64,  BLOCK_N=64,  num_warps=4, num_stages=3),
    TileConfig(BLOCK_M=64,  BLOCK_N=64,  num_warps=8, num_stages=2),
    TileConfig(BLOCK_M=64,  BLOCK_N=128, num_warps=4, num_stages=2),
    TileConfig(BLOCK_M=64,  BLOCK_N=128, num_warps=8, num_stages=2),
    # Medium tiles (default class)
    TileConfig(BLOCK_M=128, BLOCK_N=32,  num_warps=4, num_stages=4),
    TileConfig(BLOCK_M=128, BLOCK_N=32,  num_warps=8, num_stages=3),
    TileConfig(BLOCK_M=128, BLOCK_N=64,  num_warps=4, num_stages=3),   # legacy default
    TileConfig(BLOCK_M=128, BLOCK_N=64,  num_warps=8, num_stages=2),   # new default
    TileConfig(BLOCK_M=128, BLOCK_N=128, num_warps=4, num_stages=2),
    TileConfig(BLOCK_M=128, BLOCK_N=128, num_warps=8, num_stages=2),
    # Large tiles (good for large-N dense to amortise loop overhead)
    TileConfig(BLOCK_M=256, BLOCK_N=32,  num_warps=8, num_stages=2),
    TileConfig(BLOCK_M=256, BLOCK_N=64,  num_warps=8, num_stages=2),
    TileConfig(BLOCK_M=256, BLOCK_N=64,  num_warps=4, num_stages=2),
]

# Variants under test.
@af.attention
def _af_dense(Q, K, V):
    return af.softmax(af.scaled_dot_product(Q, K)) @ V

@af.attention
def _af_causal(Q, K, V):
    return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V

@af.attention
def _af_sw(Q, K, V):
    return af.softmax(af.sliding_window(af.scaled_dot_product(Q, K), WINDOW)) @ V

@af.attention
def _af_causal_alibi(Q, K, V):
    s = af.scaled_dot_product(Q, K)
    s = af.alibi(s, num_heads=NUM_HEADS)
    s = af.causal(s)
    return af.softmax(s) @ V


VARIANTS = [
    ("dense",        _af_dense),
    ("causal",       _af_causal),
    ("sw_w256",      _af_sw),
    ("causal_alibi", _af_causal_alibi),
]
SEQLENS = [512, 1024, 2048, 4096]


def _bench_ms(fn, Q, K, V) -> float:
    """Median latency over ITERS calls, with WARMUP warmups."""
    for _ in range(WARMUP):
        fn(Q, K, V)
    torch.cuda.synchronize()
    es = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ee = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for s, e in zip(es, ee):
        s.record(); fn(Q, K, V); e.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(es, ee))


def _bench_with_config(fn, Q, K, V, cfg: TileConfig) -> float | None:
    """Force AttnFuse to use ``cfg`` for one call, then time normally.

    Bypasses ``choose_tile_config`` by monkeypatching the table; clears the
    LaunchBundle cache so the next call recompiles with the new config.
    """
    # Save originals
    orig_table = dict(_tiling._AMPERE_TABLE_F16)
    orig_cache = dict(kernel_cache._cache) if hasattr(kernel_cache, "_cache") else None

    try:
        _tiling._AMPERE_TABLE_F16[HEAD_DIM] = cfg
        # Wipe the LaunchBundle cache so we recompile with the new config
        if hasattr(kernel_cache, "_cache"):
            kernel_cache._cache.clear()
        # Warm + time
        return _bench_ms(fn, Q, K, V)
    except Exception as exc:
        # SMEM overflow, register pressure, etc.
        msg = str(exc).splitlines()[0][:80]
        print(f"    [skip] {cfg}: {msg}")
        return None
    finally:
        # Restore
        _tiling._AMPERE_TABLE_F16.clear()
        _tiling._AMPERE_TABLE_F16.update(orig_table)
        if orig_cache is not None and hasattr(kernel_cache, "_cache"):
            kernel_cache._cache.clear()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/config_sweep.csv")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available"); return 1

    print(f"# Config sweep  device={torch.cuda.get_device_name(0)}")
    print(f"# torch={torch.__version__}  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fout = open(args.output, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["variant", "seqlen", "BLOCK_M", "BLOCK_N",
                     "num_warps", "num_stages", "latency_ms"])

    best_per_variant_seqlen: dict[tuple[str, int], tuple[TileConfig, float]] = {}

    for vname, af_fn in VARIANTS:
        for N in SEQLENS:
            print(f"\n=== {vname:14s} N={N} ===")
            g = torch.Generator(device="cuda").manual_seed(0)
            Q = torch.randn(BATCH, NUM_HEADS, N, HEAD_DIM, generator=g,
                            device="cuda", dtype=DTYPE)
            K = torch.randn_like(Q); V = torch.randn_like(Q)

            best_ms, best_cfg = float("inf"), None
            for cfg in _CONFIG_GRID:
                ms = _bench_with_config(af_fn, Q, K, V, cfg)
                if ms is None:
                    continue
                writer.writerow([vname, N, cfg.BLOCK_M, cfg.BLOCK_N,
                                 cfg.num_warps, cfg.num_stages, f"{ms:.4f}"])
                fout.flush()
                marker = ""
                if ms < best_ms:
                    best_ms, best_cfg = ms, cfg
                    marker = " *"
                print(f"  BM={cfg.BLOCK_M:3d} BN={cfg.BLOCK_N:3d} "
                      f"warps={cfg.num_warps} stages={cfg.num_stages}  "
                      f"{ms:.4f} ms{marker}")
            if best_cfg is not None:
                best_per_variant_seqlen[(vname, N)] = (best_cfg, best_ms)
            del Q, K, V; torch.cuda.empty_cache(); gc.collect()

    fout.close()

    # Print summary
    print("\n" + "=" * 70)
    print(f"{'Variant':14s} {'N':>5s} {'BM':>4s} {'BN':>4s} {'warps':>6s} {'stages':>7s}  {'ms':>10s}")
    print("-" * 70)
    for (vname, N), (cfg, ms) in sorted(best_per_variant_seqlen.items()):
        print(f"{vname:14s} {N:>5d} {cfg.BLOCK_M:>4d} {cfg.BLOCK_N:>4d} "
              f"{cfg.num_warps:>6d} {cfg.num_stages:>7d}  {ms:>9.4f}")
    print(f"\n[ok] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
