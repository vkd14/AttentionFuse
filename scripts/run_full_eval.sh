#!/usr/bin/env bash
# Week-5 deliverable: full eval sweep + figures.
set -euo pipefail
mkdir -p results
python -m benchmarks.bench_runner --output results/eval.csv "$@"
python benchmarks/make_figures.py results/eval.csv
echo "[ok] results in results/"
