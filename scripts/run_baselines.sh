#!/usr/bin/env bash
# Week-1 deliverable: capture naive PyTorch + SDPA baselines on BERT and GPT-2.
set -euo pipefail
mkdir -p results
python -m benchmarks.bench_runner \
    --variant dense \
    --model bert \
    --output results/baselines_bert.csv

python -m benchmarks.bench_runner \
    --variant causal \
    --model gpt2 \
    --output results/baselines_gpt2.csv

# Combine for convenience
python - <<'PY'
import pandas as pd
b = pd.read_csv("results/baselines_bert.csv")
g = pd.read_csv("results/baselines_gpt2.csv")
pd.concat([b, g], ignore_index=True).to_csv("results/baselines.csv", index=False)
print("[ok] results/baselines.csv")
PY
