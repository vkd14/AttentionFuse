#!/usr/bin/env bash
# AttnFuse — full benchmark sweep on H100.
#
# Run AFTER scripts/setup_h100.sh has completed.
# All outputs land under results/h100_$(date +%Y-%m-%d)/.
set -euo pipefail

ENV_NAME="${ATTNFUSE_ENV:-attnfuse}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"

# shellcheck disable=SC1091
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

OUT="results/h100_$(date +%Y-%m-%d)"
LOGS="$OUT/logs"
mkdir -p "$OUT" "$LOGS"

run_step () {
    local name="$1"; shift
    echo "==[start  $name  @ $(date +%H:%M:%S)]=="
    /usr/bin/time -f "==[done   $name  @ %E elapsed]==" \
        bash -c "$*" >> "$LOGS/${name}.log" 2>&1
    echo "==[finish $name  exit=$?]=="
}

PY=python

# --- 0. Environment snapshot ---
{
    echo "# AttnFuse H100 benchmark run"
    date
    echo
    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
               --format=csv
    echo
    $PY -c "import torch, triton; print('torch', torch.__version__, '| triton', triton.__version__, '| cuda', torch.version.cuda)"
} > "$OUT/env.txt"

# --- 1. Headline forward sweep ---
run_step bench_fp16 "$PY -m benchmarks.bench_runner --dtype float16  --output $OUT/eval_fp16.csv"
run_step bench_bf16 "$PY -m benchmarks.bench_runner --dtype bfloat16 --output $OUT/eval_bf16.csv"
run_step bench_fp32 "$PY -m benchmarks.bench_runner --dtype float32  --output $OUT/eval_fp32.csv"

# --- 2. flex_attention head-to-head ---
run_step flex_bench "$PY -m benchmarks.flex_bench --output $OUT/flex_bench.csv"
run_step composition_bench \
    "$PY -m benchmarks.composition_bench --output $OUT/composition_bench.csv"

# --- 3. LLM-realism: GQA + KV-cache decoding ---
run_step gqa_bench     "$PY -m benchmarks.gqa_bench     --output $OUT/gqa_bench.csv"
run_step kvcache_bench "$PY -m benchmarks.kvcache_bench --output $OUT/kvcache_bench.csv"

# --- 4. Backward (training) ---
run_step backward_bench "$PY -m benchmarks.backward_bench --output $OUT/backward_bench.csv"

# --- 5. RoPE micro + JIT + roofline ---
run_step rope_bench   "$PY -m benchmarks.rope_bench         --output $OUT/rope_bench.csv"
run_step jit_compile  "$PY -m benchmarks.jit_compile_bench  --output $OUT/jit_compile.csv"
run_step roofline     "$PY -m benchmarks.roofline_runner    --output $OUT/roofline.csv"

# --- 6. Tile-config sweep (slow; let it cook) ---
run_step config_sweep "$PY -m benchmarks.config_sweep \
                          --output $OUT/config_sweep.csv"
run_step backward_sweep \
    "$PY -m benchmarks.backward_config_sweep > $OUT/backward_config_sweep.txt"

echo
echo "==[all H100 benchmarks done  @ $(date +%H:%M:%S)]=="
echo "Results: $OUT"
echo
ls -la "$OUT"/*.csv 2>/dev/null
