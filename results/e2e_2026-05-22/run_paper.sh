#!/usr/bin/env bash
# Final paper-grade sweep on the 3090 after the SW 3-loop split and
# variant-aware tile config. Each step writes to results/e2e_2026-05-22/.
set -u
PY=$HOME/miniconda3/envs/attnfuse/bin/python
OUT=results/e2e_2026-05-22
LOGS=$OUT/logs
cd /home/ashiqur/Documents/DSP_Project/AttnFuse

run_step () {
  local name=$1; shift
  local cmd="$*"
  echo "==[start $name @ $(date +%H:%M:%S)]=="
  /usr/bin/time -f "==[done $name @ %E elapsed]==" bash -c "$cmd" \
      >> "$LOGS/paper_$name.log" 2>&1
  local rc=$?
  echo "==[finished $name, exit=$rc]=="
}

run_step bench_fp16   "$PY -m benchmarks.bench_runner --dtype float16  --output $OUT/eval_fp16_paper.csv"
run_step bench_bf16   "$PY -m benchmarks.bench_runner --dtype bfloat16 --output $OUT/eval_bf16_paper.csv"
run_step bench_fp32   "$PY -m benchmarks.bench_runner --dtype float32  --output $OUT/eval_fp32_paper.csv"
run_step rope_bench   "$PY -m benchmarks.rope_bench   --output $OUT/rope_bench_paper.csv"
run_step jit_compile  "$PY -m benchmarks.jit_compile_bench --output $OUT/jit_compile_paper.csv"
run_step roofline     "$PY -m benchmarks.roofline_runner   --output $OUT/roofline_paper.csv"

echo "==[all paper benches done @ $(date +%H:%M:%S)]=="
ls -la $OUT/*paper*.csv
