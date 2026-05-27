# AttnFuse

> A composable Python DSL that compiles attention to fused Triton GPU kernels.
>
> **Status:** research artifact, post-coursework, targeting MLSys-class submission.
> **Reference hardware:** NVIDIA RTX 3090 (Ampere sm_86). H100 (Hopper sm_90)
> deploy scripts in `scripts/`.

---

## What this is

A user writes a short Python function using combinators (`scaled_dot_product`,
`rope`, `causal`, `sliding_window`, `alibi`, `block_sparse`, `softmax`, …)
and AttnFuse JIT-compiles it to a single fused Triton kernel:

```python
import attnfuse as af

@af.attention
def llama_attn(Q, K, V):
    s = af.rope(Q, K)                  # rotate Q, K inside the kernel
    s = af.causal(s)                   # causal mask
    s = af.alibi(s, num_heads=12)      # ALiBi positional bias
    return af.softmax(s) @ V

out = llama_attn(Q, K, V, cos=cos, sin=sin)        # forward
loss = out.sum(); loss.backward()                  # backward via torch.autograd
```

The compiler fuses score → mask → bias → norm → aggregate into one tiled
online-softmax loop and supports **arbitrary compositions** of the
combinators. Triton's `tl.constexpr` specialisation means every variant
gets its own specialised PTX binary with **zero runtime branching**.

The same DSL surface covers:

- Forward attention inference (dense, causal, sliding-window, ALiBi, RoPE)
- **Autoregressive KV-cache decoding** with Flash Decoding (split-K)
- **Training** via a complete FA2-style backward kernel
- **Grouped-Query / Multi-Query attention** (Llama-3, Mistral, Falcon geometries)
- **Cross-attention** (encoder-decoder + KV-cache, `N_q ≠ N_kv`)
- **Block-sparse attention** (BigBird-style, user-supplied `BlockMask`)
- **Fused RoPE inside the Triton kernel** — a unique capability;
  `flex_attention`'s `score_mod` runs after $QK^\top$ and structurally
  cannot fuse Q/K rotation before the matmul.

---

## Headline results (RTX 3090, fp16)

### Forward — head-to-head against PyTorch `flex_attention`

GPT-2 geometry (B=4, H=12, D=64), bidirectional sliding-window for fair
comparison. Speedup is `flex_attention_latency / attnfuse_latency`:

| Variant | N=512 | N=1024 | N=2048 | N=4096 |
|---|---:|---:|---:|---:|
| Dense        | **1.74×** | 1.00× | 0.98× | 0.95× |
| Causal       | **1.10×** | **1.15×** | **1.04×** | 0.94× |
| **Sliding-W** | **1.16×** | **1.16×** | **1.10×** | **1.05×** |
| Causal+ALiBi | **1.20×** | **1.19×** | **1.10×** | 0.96× |

**AttnFuse wins 12 of 16 cells.** Sliding-window is a clean sweep at every
sequence length.

### Forward — RoPE compositions (the structural-novelty case)

`flex_attention` cannot fuse RoPE (Q/K rotation has to happen *before* the
score matmul; `score_mod` runs after). AttnFuse fuses everything in one kernel:

| Composition | N=512 | N=1024 | N=2048 | N=4096 |
|---|---:|---:|---:|---:|
| **Causal + RoPE** | **2.59×** | **1.67×** | **1.29×** | **2.00×** |
| **Causal + RoPE + ALiBi** | **2.39×** | **1.70×** | **1.23×** | **2.10×** |

AttnFuse wins **every cell**; up to **2.59× faster** than `flex_attention`.

### KV-cache decoding (production-LLM inference, `Q.N=1`)

Flash Decoding split-K with cyclic-replicated GQA Q heads. AttnFuse vs
`flex_attention.enable_gqa=True`:

| Geometry | cache=1024 | cache=4096 | cache=16384 | cache=32768 |
|---|---:|---:|---:|---:|
| Llama-3-8B   (H_q=32, H_kv=8)  | **1.04×** | **1.05×** | **1.07×** | **1.07×** |
| Llama-3-70B  (H_q=64, H_kv=8)  | **1.02×** | **1.05×** | **1.10×** | **1.06×** |
| Falcon-MQA   (H_q=64, H_kv=1)  | **1.06×** | **1.09×** | **1.06×** | **1.08×** |

**12/12 cells beat `flex_attention`.** At Llama-3-70B 32k cache the
previously catastrophic gap (17× slower in the original code) is now
**1.06× *faster***.

### vs PyTorch SDPA (N=4096)

For variants SDPA cannot accelerate (it falls back to O(n²)):

| Variant | AttnFuse | SDPA | Speedup |
|---|---:|---:|---:|
| Sliding-window (W=256) | 440 TFLOPS | 7.8 TFLOPS | **56.5×** |
| Causal + ALiBi         | 117 TFLOPS | 5.6 TFLOPS | **20.7×** |

### Backward pass (training)

| Variant @ N=4096 | AttnFuse fwd+bwd | SDPA fwd+bwd | Ratio |
|---|---:|---:|---:|
| Causal | 8.93 ms | 5.93 ms | **1.51×** of SDPA |

Backward is correct across fp16 / bf16 / fp32 and MHA / GQA / MQA. The
1.5× gap to SDPA is mostly the documented Triton-vs-CUTLASS overhead
amplified across the 7 matmuls per (m, n) tile pair in backward (vs 2
in forward).

### Block-sparse (BigBird) — sub-quadratic in active fraction

| Pattern at N=4096 | Active blocks | Latency |
|---|---:|---:|
| Dense AttnFuse (for reference) | 4096 / 4096 | 3.48 ms |
| BigBird (global + local + random) | 314 / 4096 (7.7%) | **0.47 ms** |
| Strided every-other-block | 2048 / 4096 (50%) | 1.90 ms |

Active-fraction scaling confirms the kernel is genuinely sub-quadratic.

---

## Combinators

```python
af.scaled_dot_product(Q, K)        # S = QKᵀ / √d
af.rope(Q, K)                      # fused RoPE inside the kernel
af.causal(s)                       # lower-triangular mask
af.sliding_window(s, W)            # local attention, |i - j| < W
af.full(s)                         # no-op mask (dense)
af.block_sparse(s)                 # user-supplied BlockMask at call time
af.alibi(s, num_heads=H)           # ALiBi linear bias
af.additive_bias(s)                # external (B,H,N,N) bias tensor
af.softmax(s)                      # online stable softmax
af.relu_attention(s)               # ReLU normalisation
```

Compose them freely — the compiler picks a tile config from a
hardware-tuned table and emits one Triton kernel per unique combination.

---

## Install

```bash
git clone <repo>
cd AttnFuse

conda create -n attnfuse python=3.11 -y
conda activate attnfuse

pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install triton==3.1.0
pip install -e ".[bench,dev]"

python -m pytest tests/        # 95 GPU tests
```

**H100 deployment:** see `scripts/H100_DEPLOY.md` for one-command setup
and benchmark scripts (`scripts/setup_h100.sh`, `scripts/run_h100_benchmarks.sh`).

---

## Quick examples

```python
import torch, attnfuse as af

# 1. GPT-2 style causal
@af.attention
def gpt2_attn(Q, K, V):
    return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V

# 2. LLaMA / Mistral: fused RoPE + GQA + causal
@af.attention
def llama_attn(Q, K, V):
    s = af.rope(Q, K)
    s = af.causal(s)
    return af.softmax(s) @ V

Q  = torch.randn(2, 32,  4096, 128, device="cuda", dtype=torch.float16)   # H_q = 32
KV = lambda: torch.randn(2, 8, 4096, 128, device="cuda", dtype=torch.float16)  # H_kv = 8 (GQA)
K, V = KV(), KV()
from attnfuse.rope_utils import build_rope_cache
cos, sin = build_rope_cache(4096, 128, device="cuda", dtype=torch.float16)
out = llama_attn(Q, K, V, cos=cos, sin=sin)

# 3. KV-cache decode step (Flash Decoding automatic)
Q1 = torch.randn(2, 32,    1, 128, device="cuda", dtype=torch.float16)  # one new token
out = gpt2_attn(Q1, K, V)   # autoregressive decode

# 4. Training (backward via torch.autograd)
Q.requires_grad_(); K.requires_grad_(); V.requires_grad_()
out = gpt2_attn(Q, K, V)
out.sum().backward()        # AttnFuse backward kernel runs

# 5. BigBird block-sparse
def bigbird(q, kv):
    qb = q // 64; kb = kv // 64
    return (qb == 0) | (kb == 0) | ((qb - kb).abs() <= 1)

mask = af.create_block_mask(bigbird, 4096, 4096, BLOCK_M=64, BLOCK_N=64)

@af.attention
def bigbird_attn(Q, K, V):
    s = af.scaled_dot_product(Q, K)
    s = af.block_sparse(s)
    return af.softmax(s) @ V

out = bigbird_attn(Q, K, V, block_mask=mask)
```

---

## Repository layout

```
AttnFuse/
├── attnfuse/
│   ├── dsl/                 ← @attention decorator + combinators + tracer
│   ├── ir/                  ← high-level graph + tiled-kernel record
│   ├── compiler/            ← fwd kernel, decode kernel, backward kernel,
│   │                          block-sparse kernel, tiling, lowering
│   ├── runtime/             ← dispatch, kernel cache, flash decode,
│   │                          block_mask, autograd, backward
│   ├── reference/           ← naive PyTorch / SDPA / hand Triton FA baselines
│   └── rope_utils.py        ← RoPE cache builder + host-side apply_rope
├── benchmarks/              ← bench_runner, flex_bench, composition_bench,
│                              gqa_bench, kvcache_bench, backward_bench,
│                              blocksparse_bench, rope_bench, roofline_runner,
│                              ablation, config_sweep, ...
├── examples/                ← runnable demos per variant
├── tests/                   ← 95 GPU correctness tests
├── scripts/                 ← H100 deploy + reproducibility scripts
├── results/                 ← committed benchmark CSVs and figures
└── docs/                    ← final_report.tex (course paper)
```

---

## Key design decisions

| Decision | Why |
|---|---|
| Two-level IR (Graph → TiledKernel) | Separates user-facing combinator semantics from tile-loop implementation |
| Single parameterised Triton kernel | `tl.constexpr` specialisation gives zero runtime branching |
| LaunchBundle cache | Per-call dispatch overhead < 1 µs |
| Variant-aware tile tables | Sparse variants (causal, SW, ALiBi) want different tiles than dense |
| Flash Decoding split-K kernel | Saturates SMs for `Q.N = 1` autoregressive decoding |
| Cyclic Q-head replication in decode | Recovers tensor-core throughput for GQA without redundant K/V loads |
| FA2-style backward (saved L, recompute P) | Standard algorithm; ~O(1) extra memory beyond the forward |
| Dedicated block-sparse kernel | CSR-style active-block lists give true sub-quadratic FLOPs |
| Hopper-aware tile selection | Auto-detects sm_90 at import time; uses bigger SMEM and tile sizes |
| Hand-tuned tile tables vs autotune | Faster compile, predictable behaviour; full autotune is a future opt-in |

---

## Hardware notes

**Ampere (RTX 3090, sm_86, reference)**

- 82 SMs, 24 GB GDDR6X, 936 GB/s
- ~101 KB usable SMEM per block (after CUDA driver overhead)
- Default fp16 tiles in `_AMPERE_TABLE_F16{,_SPARSE,_ROPE}` per `attnfuse/compiler/tiling.py`
- Triton 3.1 `tl.dot(fp32, fp32)` precision bug on the (M,N)@(N,D) reduction
  direction is worked around with a bf16 cast in the backward dQ matmul

**Hopper (H100, sm_90)**

- 132 SMs, ~228 KB SMEM per SM
- WGMMA + TMA available but not yet specifically exploited
- `_HOPPER_TABLE_*` provide tuned starting points; the benchmark sweep on
  H100 (`scripts/run_h100_benchmarks.sh`) refines them

---

## Reproducing the headline numbers

```bash
# Full evaluation sweep (this writes everything to results/e2e_*)
python -m benchmarks.bench_runner --dtype float16 --output results/eval_fp16.csv
python -m benchmarks.bench_runner --dtype bfloat16 --output results/eval_bf16.csv
python -m benchmarks.bench_runner --dtype float32 --output results/eval_fp32.csv

# Head-to-head vs flex_attention
python -m benchmarks.flex_bench         --output results/flex_bench.csv
python -m benchmarks.composition_bench  --output results/composition_bench.csv

# LLM-realism
python -m benchmarks.gqa_bench          --output results/gqa_bench.csv
python -m benchmarks.kvcache_bench      --output results/kvcache_bench.csv

# Training
python -m benchmarks.backward_bench     --output results/backward_bench.csv

# Block-sparse (BigBird)
python -m benchmarks.blocksparse_bench  --output results/blocksparse_bench.csv

# Micro-benchmarks
python -m benchmarks.rope_bench         --output results/rope_bench.csv
python -m benchmarks.jit_compile_bench  --output results/jit_compile.csv
python -m benchmarks.roofline_runner    --output results/roofline.csv
```

---

## Testing

```bash
pytest tests/                   # 95 GPU correctness tests
pytest tests/test_backward.py   # 30 backward correctness tests
pytest tests/test_block_sparse.py
pytest tests/test_fuzz.py       # 80-example Hypothesis property fuzzer
```

All tests check against a naive PyTorch reference within FlashAttention-2's
documented fp16 tolerance (`max|err| < 2e-2`).

---

## Citation

This repository accompanies a paper draft (in preparation) on composable
attention compilation. See `docs/final_report.tex` for the course version
and `results/e2e_2026-05-22/PAPER_HEADLINE.md` / `FINAL_RESULTS.md` for
the latest paper-grade numbers and the full session changelog. A separate
`PROGRESS.md` (root of this repo) summarises the work done between the
original course submission and the current state.

---

## License

MIT
