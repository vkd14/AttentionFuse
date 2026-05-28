"""HuggingFace Llama-3 training step benchmark.

Measures forward+backward+optimizer.step() on a Llama-3-8B-class
transformer block with three attention backends:

  * sdpa             - PyTorch's stock SDPA
  * flex_attention   - PyTorch's compile-based attention
  * attnfuse         - our registered backend

The model uses real Llama-3 dimensions but RANDOMLY-INITIALISED weights
(we are measuring kernel speed, not loss curves; downloading the actual
HF checkpoint is not required for a perf bench).

Geometry (Llama-3-8B):
  hidden_size      = 4096
  num_heads        = 32
  num_kv_heads     = 8     (GQA group_size = 4)
  head_dim         = 128
  intermediate     = 14336
  one transformer block ~ 218 M params with GQA

Default seq lengths: 512, 1024, 2048. (4096 OOMs at batch=1 with the
big intermediate-size FFN on a 24 GB 3090; users with H100 can raise.)
"""
from __future__ import annotations

import argparse
import csv
import gc
import math
import statistics
import time
import warnings
from pathlib import Path

import torch

# Register AttnFuse with HF before importing transformers' Llama
import attnfuse.integrations.hf  # noqa: F401

from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer, LlamaRotaryEmbedding,
)

BATCH    = 1
DTYPE    = torch.float16
WARMUP   = 4
ITERS    = 20
SEQLENS  = [512, 1024, 2048]
BACKENDS = ["sdpa", "flex_attention", "attnfuse"]


def _make_block(impl: str) -> tuple[LlamaDecoderLayer, LlamaConfig, LlamaRotaryEmbedding]:
    cfg = LlamaConfig(
        hidden_size=4096,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=8,            # GQA group_size = 4
        intermediate_size=14336,
        max_position_embeddings=2048,
        rope_scaling=None,
        attn_implementation=impl,
        torch_dtype=DTYPE,
    )
    layer = LlamaDecoderLayer(cfg, layer_idx=0)
    layer = layer.to(device="cuda", dtype=DTYPE)
    rotary = LlamaRotaryEmbedding(cfg).to(device="cuda", dtype=DTYPE)
    return layer, cfg, rotary


def _make_inputs(N: int, cfg: LlamaConfig, rotary: LlamaRotaryEmbedding):
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(BATCH, N, cfg.hidden_size, device="cuda",
                     dtype=DTYPE, generator=g, requires_grad=True)
    pos = torch.arange(N, device="cuda").unsqueeze(0)
    cos, sin = rotary(x, pos)
    return x, pos, (cos, sin)


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


def _train_step(layer, opt, x, pos, pos_emb):
    def step():
        opt.zero_grad(set_to_none=True)
        out = layer(x, position_ids=pos, position_embeddings=pos_emb)[0]
        out.sum().backward()
        opt.step()
    return step


def _fwd_only(layer, x, pos, pos_emb):
    def step():
        with torch.no_grad():
            layer(x, position_ids=pos, position_embeddings=pos_emb)
    return step


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/hf_llama_bench.csv")
    args = p.parse_args()
    if not torch.cuda.is_available():
        print("CUDA not available"); return 1

    print(f"# HuggingFace Llama-3-8B (random init) bench")
    print(f"# device={torch.cuda.get_device_name(0)}")
    print(f"# {time.strftime('%Y-%m-%d %H:%M:%S')}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fout = open(args.output, "w", newline="")
    w = csv.writer(fout)
    w.writerow(["seqlen", "stage", "backend", "latency_ms", "speedup_vs_sdpa"])

    print(f"{'N':>5s}  {'stage':>10s}  {'backend':>14s}  {'ms':>10s}  vs_sdpa")
    for N in SEQLENS:
        per_backend_ms = {"fwd": {}, "train": {}}
        for impl in BACKENDS:
            try:
                layer, cfg, rotary = _make_block(impl)
                x, pos, pos_emb = _make_inputs(N, cfg, rotary)

                # Forward only
                fwd_ms = _bench_ms(_fwd_only(layer, x, pos, pos_emb))
                per_backend_ms["fwd"][impl] = fwd_ms

                # Full training step
                opt = torch.optim.SGD(layer.parameters(), lr=1e-3)
                train_ms = _bench_ms(_train_step(layer, opt, x, pos, pos_emb))
                per_backend_ms["train"][impl] = train_ms

                del layer, opt, x, pos, pos_emb, rotary
                torch.cuda.empty_cache(); gc.collect()
            except Exception as e:
                msg = str(e).splitlines()[0][:120]
                print(f"  [skip {impl} N={N}] {msg}")
                per_backend_ms["fwd"][impl] = float("nan")
                per_backend_ms["train"][impl] = float("nan")

        sdpa_fwd   = per_backend_ms["fwd"].get("sdpa", float("nan"))
        sdpa_train = per_backend_ms["train"].get("sdpa", float("nan"))
        for stage, label in [("fwd", "fwd"), ("train", "fwd+bwd+step")]:
            for impl in BACKENDS:
                ms = per_backend_ms[stage].get(impl, float("nan"))
                ref = sdpa_fwd if stage == "fwd" else sdpa_train
                ratio = (ms / ref) if ref > 0 and not math.isnan(ref) and not math.isnan(ms) else float("nan")
                w.writerow([N, label, impl,
                            f"{ms:.3f}" if not math.isnan(ms) else "nan",
                            f"{ratio:.2f}" if not math.isnan(ratio) else "nan"])
                fout.flush()
                print(f"{N:>5d}  {label:>10s}  {impl:>14s}  "
                      f"{ms:>9.3f}  {'(ref)' if impl == 'sdpa' else f'{ratio:.2f}x'}")

    fout.close()
    print(f"\n[ok] wrote {args.output}")
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    raise SystemExit(main())
