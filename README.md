# AttnFuse

> An embedded Python DSL for specifying attention mechanisms and automatically
> compiling them into fused Triton GPU kernels.
>
> **Course:** CompSci 790/657 — Domain Specific Programming for AI
> **Target hardware:** NVIDIA RTX 3090 Ti (Ampere, sm_86, 24 GB GDDR6X, 84 SMs, 128 KB L1/SMEM per SM)

---

## What this project is

Modern Transformers spend most of their compute and memory budget inside multi-head
self-attention. PyTorch eager-mode attention launches several kernels (Q·Kᵀ, scale,
mask, softmax, attn·V) and materialises the full *n × n* score matrix in HBM —
quadratic in sequence length. FlashAttention solves this by hand, but writing
new attention variants still means writing a new CUDA/Triton kernel by hand.

**AttnFuse** is a small DSL where a user *declares* an attention variant
— mask pattern, score function, normalisation — and the compiler emits a
fused, tiled Triton kernel that never materialises the full attention matrix.

```python
import attnfuse as af

@af.attention
def my_attn(Q, K, V):
    scores = af.scaled_dot_product(Q, K)          # score combinator
    scores = af.causal(scores)                    # mask combinator
    scores = af.alibi(scores, num_heads=Q.num_heads)  # bias combinator
    probs  = af.softmax(scores)                   # norm combinator
    return probs @ V

out = my_attn(Q, K, V)   # JIT-compiles a fused Triton kernel on first call
```

Four built-in variants ship: `dense`, `causal`, `sliding_window`, `causal_alibi`.

---

## Repository layout

```
AttnFuse/
├── README.md                ← you are here
├── pyproject.toml
├── requirements.txt
├── attnfuse/                ← the DSL + compiler + runtime
│   ├── dsl/                 ← user-facing surface (decorator, combinators, tracer)
│   ├── ir/                  ← high-level dataflow IR + tiled-loop IR
│   ├── compiler/            ← lowering, tiling, fusion, codegen passes
│   ├── runtime/             ← kernel cache + dispatch
│   └── reference/           ← PyTorch naive, SDPA, manual-Triton-flash baselines
├── benchmarks/              ← BERT-base + GPT-2 timing harness
├── tests/                   ← pytest suite (correctness vs PyTorch reference)
├── examples/                ← short, runnable .py demos for each variant
├── scripts/                 ← shell helpers (env setup, full eval)
├── docs/design.md           ← IR + pass design notes
└── results/                 ← CSVs and plots produced by the benchmark runner
```

---

## Step-by-step tasks (what *you* do)

These steps follow the 6-week timeline in the proposal. Run them in order.
Each step ends with a `git tag` so you can resume from a known state.

### Step 0 — Hardware & driver sanity check (do this once)

Confirm the box is an RTX 3090 Ti with a recent driver:

```bash
nvidia-smi                        # expect: Driver ≥ 535, CUDA ≥ 12.1
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
# expect: NVIDIA GeForce RTX 3090 Ti, 8.6, 24576 MiB
```

If `compute_cap` is not `8.6`, you are not on the target GPU and the Triton
block sizes in `attnfuse/compiler/tiling.py` will not be optimal — re-tune
`BLOCK_M`, `BLOCK_N`, `num_warps`, `num_stages`.

### Step 1 — Environment setup

```bash
cd AttnFuse
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[bench,dev]"
python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.is_available())"
```

You should see CUDA `True` and Triton ≥ 2.2. Then:

```bash
pytest tests/test_smoke.py -v   # 30-second sanity check
```

### Step 2 — Run the baselines (Week 1 deliverable)

Before optimising anything, capture how slow naive PyTorch is and how fast
SDPA's built-in FlashAttention is. These two numbers bracket what AttnFuse
must beat / approach.

```bash
bash scripts/run_baselines.sh
# → results/baselines.csv  (latency_ms, peak_mem_mb, tflops)
```

Inspect the CSV. Expect ~2–4× gap between naive and SDPA at seqlen=2048.

### Step 3 — DSL frontend (Week 2)

Read `attnfuse/dsl/api.py`. Try the four canned examples:

```bash
python examples/bert_dense.py      # dense bidirectional
python examples/gpt2_causal.py     # causal LM
python examples/sliding_window.py  # Mistral-style local attention
python examples/causal_alibi.py    # GPT-2 style + ALiBi positional bias
```

Each script prints the captured high-level IR before running. Verify the
graph matches what you expect.

### Step 4 — Compiler & codegen (Week 3)

The compiler lowers the high-level IR through tiling → fusion → Triton source.
Re-run the examples with `ATTNFUSE_DEBUG=1`:

```bash
ATTNFUSE_DEBUG=1 python examples/gpt2_causal.py
# Dumps: high-level IR, tiled IR, generated Triton source.
```

Generated kernels are cached under `attnfuse/_generated/`. Delete that
directory to force re-compilation.

### Step 5 — Correctness (Week 4)

```bash
pytest tests/test_correctness.py -v
# Compares AttnFuse output against torch.nn.functional.scaled_dot_product_attention
# with atol=1e-2 (fp16) / 1e-3 (fp32) for every variant.
```

If a test fails, run with `ATTNFUSE_DEBUG=1` to see which pass produced wrong IR.

### Step 6 — Profile with Nsight Compute (Week 4)

```bash
ncu --set full --target-processes all -o results/ncu_dense \
    python -m benchmarks.bench_runner --variant dense --seqlen 2048 --warmup 3 --iters 1
```

Open `results/ncu_dense.ncu-rep` in `ncu-ui`. Look for SM occupancy ≥ 50 %
and HBM read traffic ≪ `4 * n^2 * 2` bytes (fp16). If not, revisit
`attnfuse/compiler/tiling.py`.

### Step 7 — Full evaluation (Week 5)

```bash
bash scripts/run_full_eval.sh
# Sweeps: variants × seqlens {512, 1024, 2048, 4096} × baselines.
# → results/eval.csv, results/eval_latency.png, results/eval_memory.png
```

Targets from the proposal:

| Metric                | Target vs. naive PyTorch |
|-----------------------|--------------------------|
| Inference latency     | 1.5 – 2× faster          |
| Peak GPU memory       | ≥ 40 % reduction         |
| FlashAttention-2 gap  | within 10–20 % is good   |

### Step 8 — Report & packaging (Week 6)

```bash
python benchmarks/make_figures.py results/eval.csv  # generates report-ready PDFs
```

Drop the figures into `docs/report.tex`. Tag the final state:

```bash
git tag v1.0-final
```

---

## Suggested commit cadence

The repo is committed in logical chunks; `git log --oneline` should read like a
build journal. When you extend it, keep the cadence:

1. scaffold / configs
2. DSL frontend
3. IR + printers
4. compiler passes (one commit per pass)
5. runtime + dispatch
6. reference baselines
7. benchmark harness
8. tests
9. examples + docs
10. eval results

---

## Hardware-specific notes (RTX 3090 Ti)

- **Tensor cores:** Ampere fp16/bf16 with HMMA.16168.F16 — generated kernels
  use `tl.dot` with bf16 accumulation when inputs are bf16.
- **No fp8.** Do not enable Triton's `fp8_fast_accum`; that path requires Hopper.
- **SMEM budget:** 100 KB usable per block (after CUDA driver overhead) →
  default `BLOCK_M=128, BLOCK_N=64, head_dim ≤ 128` keeps Q/K/V tiles in SMEM.
- **Register pressure:** `num_warps=4, num_stages=3` is the safe default;
  `num_warps=8` only helps for `head_dim ≥ 128` and `BLOCK_M ≥ 128`.
- **PCIe Gen 4 x16:** keep tensors resident on GPU between benchmark iterations;
  the warm-up phase moves and pins them.

---

## License

MIT — see `LICENSE`.
