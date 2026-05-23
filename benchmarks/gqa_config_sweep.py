"""Quick targeted sweep for GQA tile configs (HEAD_DIM=128).

GQA on HEAD_DIM=128 is slower than flex_attention on AttnFuse; the
default sparse table (BM=64, BN=32) appears suboptimal. This script
sweeps a small grid of plausible configs against the real GQA workload
to find a better setting.
"""
from __future__ import annotations

import csv
import gc
import statistics
import time
from pathlib import Path

import torch
import attnfuse as af
from attnfuse.compiler import tiling as _tiling
from attnfuse.ir.tiled import TileConfig
from attnfuse.runtime import kernel_cache


# Llama-3-8B-like geometry
B, H_Q, H_KV, D = 2, 32, 8, 128
DTYPE  = torch.float16
WARMUP = 8
ITERS  = 40
SEQLENS = [1024, 2048, 4096]

# HEAD_DIM=128 candidates. SMEM budget ~101 KB.
# Per Q tile: BM * 128 * 2 = 256*BM bytes; per K/V stage: BN * 128 * 2 = 256*BN
# 2-stage K+V pipeline = 4 * 256 * BN bytes
# Total: Q + pipeline + ALiBi/cos/sin (~few KB)
_GRID: list[TileConfig] = [
    TileConfig(BLOCK_M=64,  BLOCK_N=32,  num_warps=4, num_stages=2),   # current default
    TileConfig(BLOCK_M=64,  BLOCK_N=64,  num_warps=4, num_stages=2),
    TileConfig(BLOCK_M=64,  BLOCK_N=64,  num_warps=8, num_stages=2),
    TileConfig(BLOCK_M=128, BLOCK_N=32,  num_warps=4, num_stages=2),
    TileConfig(BLOCK_M=128, BLOCK_N=32,  num_warps=8, num_stages=2),
    TileConfig(BLOCK_M=128, BLOCK_N=64,  num_warps=4, num_stages=2),
    TileConfig(BLOCK_M=128, BLOCK_N=64,  num_warps=8, num_stages=2),
    TileConfig(BLOCK_M=128, BLOCK_N=128, num_warps=8, num_stages=2),
]


@af.attention
def causal(Q, K, V):
    return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V


def _bench_ms(Q, K, V) -> float:
    for _ in range(WARMUP):
        causal(Q, K, V)
    torch.cuda.synchronize()
    es = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ee = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for s, e in zip(es, ee):
        s.record(); causal(Q, K, V); e.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(es, ee))


def _with_config(cfg: TileConfig, Q, K, V) -> float | None:
    orig = dict(_tiling._AMPERE_TABLE_F16_SPARSE)
    try:
        _tiling._AMPERE_TABLE_F16_SPARSE[D] = cfg
        kernel_cache._cache.clear()
        return _bench_ms(Q, K, V)
    except Exception as e:
        print(f"    [skip] {cfg}: {str(e).splitlines()[0][:80]}")
        return None
    finally:
        _tiling._AMPERE_TABLE_F16_SPARSE.clear()
        _tiling._AMPERE_TABLE_F16_SPARSE.update(orig)
        kernel_cache._cache.clear()


def main() -> int:
    print(f"# GQA tile sweep  device={torch.cuda.get_device_name(0)}")
    print(f"# Geometry: B={B} H_q={H_Q} H_kv={H_KV} D={D} (group={H_Q//H_KV})")
    print(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}")

    out_path = Path("results/e2e_2026-05-22/gqa_config_sweep.csv")
    fout = open(out_path, "w", newline="")
    writer = csv.writer(fout)
    writer.writerow(["N", "BLOCK_M", "BLOCK_N", "num_warps", "num_stages", "latency_ms"])

    best_by_n = {}
    for N in SEQLENS:
        print(f"\n=== N={N} ===")
        g = torch.Generator(device="cuda").manual_seed(0)
        Q = torch.randn(B, H_Q,  N, D, generator=g, device="cuda", dtype=DTYPE)
        K = torch.randn(B, H_KV, N, D, generator=g, device="cuda", dtype=DTYPE)
        V = torch.randn(B, H_KV, N, D, generator=g, device="cuda", dtype=DTYPE)

        best_ms, best_cfg = float("inf"), None
        for cfg in _GRID:
            ms = _with_config(cfg, Q, K, V)
            if ms is None:
                continue
            writer.writerow([N, cfg.BLOCK_M, cfg.BLOCK_N,
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
            best_by_n[N] = (best_cfg, best_ms)
        del Q, K, V; torch.cuda.empty_cache(); gc.collect()

    fout.close()
    print("\n" + "=" * 60)
    for N, (cfg, ms) in sorted(best_by_n.items()):
        print(f"N={N:5d}  BM={cfg.BLOCK_M:3d} BN={cfg.BLOCK_N:3d} "
              f"warps={cfg.num_warps} stages={cfg.num_stages}  "
              f"{ms:.4f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
