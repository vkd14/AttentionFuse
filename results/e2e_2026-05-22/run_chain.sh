#!/usr/bin/env bash
# Chained run of every remaining benchmark for the 3090 e2e snapshot.
# Each step writes its own CSV + log; on failure, the next step still runs
# so we maximise the data we get back.
set -u
PY=$HOME/miniconda3/envs/attnfuse/bin/python
OUT=results/e2e_2026-05-22
LOGS=$OUT/logs
cd /home/ashiqur/Documents/DSP_Project/AttnFuse

run_step () {
  local name=$1; shift
  local cmd="$*"
  echo "==[start $name @ $(date +%H:%M:%S)]=="
  /usr/bin/time -f "==[done  $name @ %E elapsed]==" bash -c "$cmd" \
      >> "$LOGS/chain_$name.log" 2>&1
  echo "==[finished $name, exit=$? rows=$(wc -l <"$OUT/${name}.csv" 2>/dev/null || echo n/a)]=="
}

# --- bf16 sweep ---
run_step bench_bf16 "$PY -m benchmarks.bench_runner --dtype bfloat16 --output $OUT/eval_bf16.csv"

# --- fp32 sweep ---
run_step bench_fp32 "$PY -m benchmarks.bench_runner --dtype float32 --output $OUT/eval_fp32.csv"

# --- rope fused vs preprocess ---
run_step rope_bench "$PY -m benchmarks.rope_bench --output $OUT/rope_bench.csv"

# --- jit compile / cached call ---
run_step jit_compile "$PY -m benchmarks.jit_compile_bench --output $OUT/jit_compile.csv"

# --- corrected-FLOP roofline ---
run_step roofline "$PY -m benchmarks.roofline_runner --output $OUT/roofline.csv"

# --- tile-config ablation at N=2048 (longest) ---
run_step ablation "$PY -m benchmarks.ablation --dtype float16 --output $OUT/ablation.csv"

echo "==[all done @ $(date +%H:%M:%S)]=="
ls -la $OUT/*.csv
