# AttnFuse

> An embedded Python DSL that compiles attention variants to fused Triton GPU kernels.
>
> **Course:** CS 790/657 — Domain-Specific Programming for AI  
> **Hardware:** NVIDIA RTX 3090 (Ampere sm_86, 24 GB GDDR6X, 936 GB/s)

---

## Overview

AttnFuse lets you *declare* an attention mechanism in a few lines of Python and
get a fused, IO-optimal GPU kernel automatically — no Triton or CUDA required.

```python
import attnfuse as af

@af.attention
def my_attn(Q, K, V):
    s = af.scaled_dot_product(Q, K)    # score
    s = af.causal(s)                   # causal mask
    s = af.alibi(s, num_heads=12)      # ALiBi positional bias
    return af.softmax(s) @ V

out = my_attn(Q, K, V)   # JIT-compiles a fused Triton kernel on first call
```

The compiler fuses score → mask → bias → norm → aggregate into one tiled
online-softmax loop, achieving the same IO complexity as FlashAttention-2
while supporting arbitrary mask/bias/norm compositions.

---

## Supported combinators

| Combinator | Description |
|---|---|
| `af.scaled_dot_product(Q, K)` | $S = QK^T / \sqrt{d}$ |
| `af.rope(Q, K)` | Fused RoPE rotation inside the Triton kernel |
| `af.causal(s)` | Lower-triangular causal mask |
| `af.sliding_window(s, window_size)` | Local attention window |
| `af.full(s)` | No-op mask (dense) |
| `af.alibi(s, num_heads)` | ALiBi linear positional bias |
| `af.additive_bias(s)` | External (B,H,N,N) bias tensor |
| `af.softmax(s)` | Online stable softmax |
| `af.relu_attention(s)` | ReLU normalisation |

---

## Variants and performance (RTX 3090, fp16, N=4096)

| Variant | AttnFuse | SDPA | Speedup |
|---|---|---|---|
| Dense | 59 TFLOPS | 70 TFLOPS | 0.85× |
| Causal | 105 TFLOPS | 122 TFLOPS | 0.86× |
| **Sliding-window W=256** | **298 TFLOPS** | 7.8 TFLOPS | **38×** |
| **Causal + ALiBi** | **98 TFLOPS** | 5.6 TFLOPS | **17×** |

SDPA falls back to O(n²) for sliding-window and ALiBi; AttnFuse stays
sub-quadratic for all variants.

---

## Install

```bash
git clone <repo>
cd AttnFuse

# Create conda environment (recommended)
conda create -n attnfuse python=3.11 -y
conda activate attnfuse

# Install with CUDA 12.1 packages
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install triton==3.1.0
pip install -e ".[bench,dev]"

# Verify
python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.is_available())"
python -m pytest tests/test_correctness.py -v   # 25 tests, all GPU
```

---

## Quick start

```python
import torch, attnfuse as af

# Dense bidirectional (BERT style)
@af.attention
def bert_attn(Q, K, V):
    return af.softmax(af.scaled_dot_product(Q, K)) @ V

# Causal (GPT style)
@af.attention
def gpt_attn(Q, K, V):
    return af.softmax(af.causal(af.scaled_dot_product(Q, K))) @ V

# Sliding-window local attention (Mistral style)
@af.attention
def mistral_attn(Q, K, V):
    s = af.sliding_window(af.scaled_dot_product(Q, K), window_size=256)
    return af.softmax(s) @ V

# RoPE fused inside the kernel (LLaMA style)
@af.attention
def llama_attn(Q, K, V):
    s = af.rope(Q, K)
    s = af.causal(s)
    return af.softmax(s) @ V

from attnfuse.rope_utils import build_rope_cache
B, H, N, D = 2, 12, 2048, 64
Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
K, V = torch.randn_like(Q), torch.randn_like(Q)
cos, sin = build_rope_cache(N, D, device="cuda", dtype=torch.float16)

out = llama_attn(Q, K, V, cos=cos, sin=sin)
```

---

## Repository layout

```
AttnFuse/
├── attnfuse/
│   ├── dsl/          ← user-facing API: @attention, combinators, tracer
│   ├── ir/           ← high-level dataflow IR + tiled-loop IR
│   ├── compiler/     ← lowering, tiling, fusion, Triton codegen passes
│   ├── runtime/      ← kernel cache (LaunchBundle) + dispatch
│   ├── reference/    ← naive / SDPA / manual-Triton-flash baselines
│   └── rope_utils.py ← RoPE table builder and host-side apply_rope
├── benchmarks/       ← BERT-base + GPT-2 timing harness, ablation, roofline
├── examples/         ← runnable demos for each variant
├── tests/            ← 25 GPU correctness tests (pytest)
├── results/          ← benchmark CSVs and figures
└── docs/             ← final_report.tex (academic paper), hw6_report.tex
```

---

## Reproducing results

```bash
# Full evaluation sweep (all variants, all dtypes)
bash benchmarks/run_dtype_sweep.sh

# Or individual runs
python -m benchmarks.bench_runner --dtype float16  --output results/eval.csv
python -m benchmarks.bench_runner --dtype bfloat16 --output results/eval_bf16.csv
python -m benchmarks.bench_runner --dtype float32  --output results/eval_fp32.csv

# Tile ablation
python -m benchmarks.ablation --dtype float16 --output results/ablation.csv
python   benchmarks/ablation_plot.py --csv results/ablation.csv

# Roofline
python -m benchmarks.bench_runner --output results/roofline.csv  # produces hbm_gbs column
python   benchmarks/roofline_plot.py

# RoPE fused vs pre-processing
python -m benchmarks.rope_bench --output results/rope_bench.csv

# JIT compile time
python -m benchmarks.jit_compile_bench --output results/jit_compile.csv
```

---

## Key design decisions

| Decision | Rationale |
|---|---|
| Two-level IR (Graph → TiledKernel) | High-level semantics separate from tile-loop implementation |
| Single parameterised kernel | `tl.constexpr` specialisation: zero runtime branching |
| LaunchBundle cache | Eliminates ~7 µs per-call Python overhead |
| Triton vs CUDA | Portability + rapid iteration; 14--17% TFLOPS gap vs SDPA for supported variants |
| Online softmax (Milakov & Gimelshein) | O(1) extra memory; numerically identical to standard softmax |

---

## Hardware notes (RTX 3090, Ampere sm_86)

- Usable SMEM per block: ~101 KB (after CUDA driver overhead)
- Default fp16 tile: `BLOCK_M=128, BLOCK_N=64, num_warps=4, num_stages=3`
- RoPE tiles: `num_stages=2` to fit extra cos/sin tiles
- fp32 tiles: `num_stages=2` (fp32 doubles tile bytes)
- Peak: 142 TFLOPS fp16, 936 GB/s HBM, ridge point ~152 FLOP/byte

---

## Citation / Report

See `docs/final_report.tex` for the complete project paper.

---

## License

MIT
