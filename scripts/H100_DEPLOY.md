# H100 deployment instructions

Run these on the H100 host via SSH. The scripts are designed to be
idempotent — safe to re-run.

## One-time setup

```bash
git clone <your-attnfuse-fork-url>
cd AttnFuse
bash scripts/setup_h100.sh
```

This will:

1. Verify the GPU is sm_90 (Hopper).
2. Install miniconda if missing.
3. Create the `attnfuse` conda env (Python 3.11).
4. Install PyTorch 2.5.1 + cu121, Triton 3.1.0, and dev deps.
5. Pip-install AttnFuse in editable mode.
6. Run the test suite (~90 tests, ~30s).

If the test suite passes, the install is good. If it fails on the
hypothesis fuzzer, that's fine for paper-grade benchmarks — those
tests are sensitive to GPU determinism and can be flaky across runs.

**Customisation knobs (env vars):**

- `ATTNFUSE_ENV=<name>` — use a different conda env name (default `attnfuse`)
- `CONDA_HOME=<path>` — use an existing conda install instead of installing miniconda

## Full benchmark sweep

```bash
bash scripts/run_h100_benchmarks.sh
```

Writes everything to `results/h100_YYYY-MM-DD/`:

| File | Content |
|---|---|
| `env.txt` | GPU + torch + triton version snapshot |
| `eval_{fp16,bf16,fp32}.csv` | Headline throughput across dtypes |
| `flex_bench.csv` | AttnFuse vs PyTorch `flex_attention` |
| `composition_bench.csv` | RoPE-composition structural-novelty bench |
| `gqa_bench.csv` | Llama-3 / Falcon GQA forward |
| `kvcache_bench.csv` | KV-cache autoregressive decoding |
| `backward_bench.csv` | Training-loop forward + backward |
| `rope_bench.csv` | Fused vs pre-process RoPE |
| `jit_compile.csv` | Cold/warm JIT compile cost |
| `roofline.csv` | Variant-correct TFLOPS + analytical HBM |
| `config_sweep.csv` | Forward tile-config sweep (slow; ~30 min) |
| `backward_config_sweep.txt` | Backward tile-config sweep |

Total wall-clock for the full sweep: roughly 1 hour on a warm Triton
cache. First run after install is longer (~90 minutes) because each
kernel is JIT-compiled on first call.

## Bringing results back

After the sweep finishes, copy `results/h100_YYYY-MM-DD/` back to your
local machine:

```bash
# from your local machine
scp -r <user>@<h100-host>:/path/to/AttnFuse/results/h100_YYYY-MM-DD ./results/
```

The committed numbers should not be overwritten — the H100 results live
in their own dated directory.

## Expected differences vs RTX 3090 (sm_86)

Hopper (sm_90) has:

- **132 SMs** (vs 82 on 3090) → more programs needed to saturate; the
  sparse-variant tile configs adapt automatically via the new
  `_HOPPER_TABLE_*` tables in `attnfuse/compiler/tiling.py`.
- **228 KB SMEM per SM** (vs ~101 KB usable on 3090) → allows
  `num_stages=3` even with BLOCK_M=128 BLOCK_N=128 in fp16, which
  is over-budget on Ampere.
- **WGMMA + TMA** → potentially big speedups but require
  Triton-specific opt-ins not in the current kernel template. Future
  work to exploit; the current ports run on the standard MMA path.
- **fp32 TF32 tensor cores** at higher throughput → the dQ bf16-cast
  workaround for the Triton 3.1 fp32-dot bug may not be needed if
  Triton has fixed it on sm_90 by then. The benchmark will reveal.

## After the run

The numbers in `results/h100_YYYY-MM-DD/` are paper-grade. To update the
written report or summary docs, plug them in via the same workflow used
for the 3090 numbers.

If a benchmark shows a worse number on H100 than on 3090, it's almost
always a tile-config issue — run `python -m benchmarks.config_sweep`
and `python -m benchmarks.backward_config_sweep` on the H100 to find a
better config, then update `_HOPPER_TABLE_*` in `tiling.py` and re-bench.
