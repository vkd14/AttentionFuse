#!/usr/bin/env bash
# Run the full evaluation sweep for float16, bfloat16, and float32.
# Outputs: results/eval.csv (fp16), results/eval_bf16.csv, results/eval_fp32.csv
# Estimated runtime: ~15 minutes on RTX 3090.
set -e

echo "=== float16 eval ==="
python -m benchmarks.bench_runner --dtype float16  --output results/eval.csv

echo "=== bfloat16 eval ==="
python -m benchmarks.bench_runner --dtype bfloat16 --output results/eval_bf16.csv

echo "=== float32 eval (N<=2048 only) ==="
python -m benchmarks.bench_runner --dtype float32  --output results/eval_fp32.csv \
    --seqlen 256 --seqlen 512 --seqlen 1024 --seqlen 2048 2>/dev/null || \
python -m benchmarks.bench_runner --dtype float32  --output results/eval_fp32.csv

echo "=== dtype comparison plot ==="
python benchmarks/dtype_comparison_plot.py \
    --csvs results/eval.csv results/eval_bf16.csv results/eval_fp32.csv \
    --out  results/dtype_comparison.png

echo "=== ablation sweeps ==="
python -m benchmarks.ablation --dtype float16  --output results/ablation.csv
python -m benchmarks.ablation --dtype bfloat16 --output results/ablation_bf16.csv
python benchmarks/ablation_plot.py --csv results/ablation.csv
python benchmarks/ablation_plot.py --csv results/ablation_bf16.csv

echo "Done."
