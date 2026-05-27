"""Tile-config sweep for the backward kernels.

The forward kernels were tuned via benchmarks/config_sweep.py; this script
does the equivalent for the dK/dV and dQ kernels. It overrides the
``_BWD_TILE`` dict in ``attnfuse.runtime.backward`` and benchmarks
forward+backward for causal+fp16 across a small config grid.

Output (CSV + console table) lets us pick a sweet-spot config per
sequence-length tier.
"""
from __future__ import annotations

import csv
import gc
import statistics
import time
from pathlib import Path

import torch
import attnfuse as af
from attnfuse.runtime import backward as _bwd

BATCH    = 4
H        = 12
HEAD_DIM = 64
DTYPE    = torch.float16
WARMUP   = 6
ITERS    = 25
SEQLENS  = [512, 1024, 2048, 4096]

# Candidate configs for the backward kernels. Note these are shared between
# the dK/dV and dQ kernels in the current implementation; future work could
# tune them separately.
_GRID = [
    {"BLOCK_M":  32, "BLOCK_N":  64, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M":  64, "BLOCK_N":  32, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M":  64, "BLOCK_N":  64, "num_warps": 4, "num_stages": 2},   # baseline
    {"BLOCK_M":  64, "BLOCK_N":  64, "num_warps": 8, "num_stages": 2},
    {"BLOCK_M":  64, "BLOCK_N": 128, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M":  64, "BLOCK_N": 128, "num_warps": 8, "num_stages": 2},
    {"BLOCK_M": 128, "BLOCK_N":  32, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 128, "BLOCK_N":  64, "num_warps": 4, "num_stages": 2},
    {"BLOCK_M": 128, "BLOCK_N":  64, "num_warps": 8, "num_stages": 2},
    {"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 8, "num_stages": 2},
]


@af.attention
def causal(Q, K, V):
    return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V


def _make(N):
    g = torch.Generator(device="cuda").manual_seed(0)
    Q = torch.randn(BATCH, H, N, HEAD_DIM, generator=g, device="cuda",
                    dtype=DTYPE, requires_grad=True)
    K = torch.randn_like(Q, requires_grad=True)
    V = torch.randn_like(Q, requires_grad=True)
    return Q, K, V


def _bench(Q, K, V) -> float:
    def call():
        Q.grad = K.grad = V.grad = None
        O = causal(Q, K, V)
        O.sum().backward()
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    es = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ee = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for s, e in zip(es, ee):
        s.record(); call(); e.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(es, ee))


def main():
    from attnfuse.runtime import kernel_cache
    # Reset state every sweep cell so the new config takes effect
    out_path = Path("results/e2e_2026-05-22/backward_config_sweep.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = open(out_path, "w", newline="")
    w = csv.writer(fout)
    w.writerow(["N", "BLOCK_M", "BLOCK_N", "num_warps", "num_stages",
                "fwd_bwd_ms"])
    print(f"# Backward config sweep  device={torch.cuda.get_device_name(0)}")
    print(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}")

    best_by_N: dict[int, tuple[dict, float]] = {}
    for N in SEQLENS:
        print(f"\n=== N={N} ===")
        Q, K, V = _make(N)
        best_ms, best_cfg = float("inf"), None
        for cfg in _GRID:
            try:
                _bwd._BWD_TILE.update(cfg)
                _bwd._BWD_TILE["auto"] = False       # use this exact config
                _bwd._bwd_module = None              # force recompile
                kernel_cache._cache.clear()
                ms = _bench(Q, K, V)
            except Exception as e:
                msg = str(e).splitlines()[0][:80]
                print(f"  BM={cfg['BLOCK_M']:>3} BN={cfg['BLOCK_N']:>3} "
                      f"w={cfg['num_warps']} s={cfg['num_stages']}  SKIP: {msg}")
                continue
            w.writerow([N, cfg["BLOCK_M"], cfg["BLOCK_N"],
                        cfg["num_warps"], cfg["num_stages"], f"{ms:.3f}"])
            fout.flush()
            marker = ""
            if ms < best_ms:
                best_ms, best_cfg = ms, cfg
                marker = " *"
            print(f"  BM={cfg['BLOCK_M']:>3} BN={cfg['BLOCK_N']:>3} "
                  f"w={cfg['num_warps']} s={cfg['num_stages']}  "
                  f"{ms:.3f} ms{marker}")
        if best_cfg is not None:
            best_by_N[N] = (best_cfg, best_ms)
        del Q, K, V
        torch.cuda.empty_cache()
        gc.collect()

    fout.close()
    print("\n" + "=" * 60)
    print(f"{'N':>5s}  {'BM':>4s}  {'BN':>4s}  {'warps':>5s}  {'stages':>6s}  {'ms':>8s}")
    for N, (cfg, ms) in sorted(best_by_N.items()):
        print(f"{N:>5d}  {cfg['BLOCK_M']:>4d}  {cfg['BLOCK_N']:>4d}  "
              f"{cfg['num_warps']:>5d}  {cfg['num_stages']:>6d}  {ms:>7.3f}")


if __name__ == "__main__":
    main()
