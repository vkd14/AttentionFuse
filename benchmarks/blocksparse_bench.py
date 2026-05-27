"""Block-sparse forward benchmark vs PyTorch flex_attention.

Both systems accept a coarse block-mask specification. We compare a
BigBird-style mask (global rows + local window + a few scattered random
blocks) and a strided pattern across sequence lengths.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
import warnings
from pathlib import Path

import torch
import attnfuse as af

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 128
    HAS_FLEX = True
except ImportError:
    HAS_FLEX = False

BATCH    = 4
H        = 12
HEAD_DIM = 64
DTYPE    = torch.float16
WARMUP   = 8
ITERS    = 40
SEQLENS  = [1024, 2048, 4096]
BLOCK    = 64


@af.attention
def bs_attn(Q, K, V):
    s = af.scaled_dot_product(Q, K)
    s = af.block_sparse(s)
    return af.softmax(s) @ V


def _bigbird(q, kv):
    qb = q // BLOCK
    kb = kv // BLOCK
    return (qb == 0) | (kb == 0) | ((qb - kb).abs() <= 1)


def _strided(q, kv):
    qb = q // BLOCK
    kb = kv // BLOCK
    return (qb + kb) % 2 == 0


def _bench_ms(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    es = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    ee = [torch.cuda.Event(enable_timing=True) for _ in range(ITERS)]
    for s, e in zip(es, ee):
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) for s, e in zip(es, ee))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/blocksparse_bench.csv")
    args = p.parse_args()
    if not torch.cuda.is_available() or not HAS_FLEX:
        print("Need CUDA + PyTorch >= 2.5"); return 1

    print(f"# Block-sparse bench  device={torch.cuda.get_device_name(0)}")
    print(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fout = open(args.output, "w", newline="")
    w = csv.writer(fout)
    w.writerow(["pattern", "seqlen", "n_active_blocks", "n_total_blocks",
                "backend", "latency_ms", "speedup_vs_flex"])

    print(f"{'pattern':>10s} {'N':>5s} {'active':>10s} {'backend':>10s} "
          f"{'ms':>8s}  vs_flex")

    for pattern_name, pred in [("bigbird", _bigbird), ("strided", _strided)]:
        for N in SEQLENS:
            g = torch.Generator(device="cuda").manual_seed(0)
            Q = torch.randn(BATCH, H, N, HEAD_DIM, generator=g,
                            device="cuda", dtype=DTYPE)
            K = torch.randn_like(Q); V = torch.randn_like(Q)

            af_mask = af.create_block_mask(pred, N, N, BLOCK, BLOCK)
            n_blocks_total = (N // BLOCK) ** 2

            # AttnFuse
            af_call = lambda: bs_attn(Q, K, V, block_mask=af_mask)
            af_ms = _bench_ms(af_call)

            # flex_attention
            try:
                flex_bm = create_block_mask(pred, B=None, H=None,
                                             Q_LEN=N, KV_LEN=N,
                                             BLOCK_SIZE=BLOCK)
                fn = torch.compile(flex_attention, dynamic=False)
                flex_call = lambda: fn(Q, K, V, block_mask=flex_bm)
                flex_ms = _bench_ms(flex_call)
            except Exception as e:
                flex_ms = float("nan")
                print(f"  flex failed for {pattern_name} N={N}: {str(e)[:80]}")

            speedup = (flex_ms / af_ms) if not math.isnan(flex_ms) else float("nan")
            for backend, ms in [("attnfuse", af_ms), ("flex", flex_ms)]:
                sp = 1.0 if backend == "attnfuse" else speedup
                w.writerow([pattern_name, N,
                            af_mask.n_active, n_blocks_total,
                            backend,
                            f"{ms:.3f}" if not math.isnan(ms) else "nan",
                            f"{sp:.2f}" if not math.isnan(sp) else "nan"])
                fout.flush()
                print(f"{pattern_name:>10s} {N:>5d} "
                      f"{af_mask.n_active:>5d}/{n_blocks_total:<4d} "
                      f"{backend:>10s} {ms:>7.3f}  "
                      f"{'(ref)' if backend == 'attnfuse' else f'{sp:.2f}x'}")
            del Q, K, V; torch.cuda.empty_cache()

    fout.close()
    print(f"\n[ok] wrote {args.output}")
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    raise SystemExit(main())
