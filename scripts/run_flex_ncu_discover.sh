#!/usr/bin/env bash
# Discovery mode: profile EVERY kernel ncu sees during the flex run,
# no filter. Fallback when the regex in run_flex_ncu.sh misses.
#
# The output CSV will be large. Find the attention kernel by cycle
# count: it should be the longest-running by far.
#
# Usage:
#   bash scripts/run_flex_ncu_discover.sh [SEQLEN]
set -euo pipefail

SEQLEN="${1:-4096}"

ENV_NAME="${ATTNFUSE_ENV:-attnfuse}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"
# shellcheck disable=SC1091
source "$CONDA_HOME/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

NCU="${NCU:-ncu}"
if ! command -v "$NCU" >/dev/null 2>&1; then
    NCU="/usr/local/cuda/bin/ncu"
fi

OUT_DIR="results/ncu"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/ncu_flex_${SEQLEN}_fp16_discover.csv"

# Minimal metric set so the CSV stays readable when we capture many kernels.
METRICS=(
    sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed
    sm__throughput.avg.pct_of_peak_sustained_elapsed
    sm__warps_active.avg.pct_of_peak_sustained_active
    dram__throughput.avg.pct_of_peak_sustained_elapsed
    smsp__warp_issue_stalled_wait_per_warp_active.pct
    launch__registers_per_thread
    launch__block_size
    gpu__time_duration.sum
)
METRICS_CSV=$(IFS=, ; echo "${METRICS[*]}")

echo "=== ncu DISCOVERY profile: flex_attention causal N=$SEQLEN fp16 ==="
echo "    writing $OUT"

# Skip the first 6 launches (warmup eager ops), then grab the next 50.
"$NCU" \
    --target-processes all \
    --launch-skip 6 \
    --launch-count 50 \
    --kernel-name-base demangled \
    --csv \
    --metrics "$METRICS_CSV" \
    python -m benchmarks.flex_ncu \
        --seqlen "$SEQLEN" --warmup 12 \
    > "$OUT" 2> "${OUT}.log"

echo ""
echo "Top kernels by duration:"
awk -F'","' '
    NR==1                                                  { next }
    /gpu__time_duration/ {
        gsub(/"/, "", $5);
        gsub(/"/, "", $15);
        kn=$5; tn=$15+0; tot[kn]+=tn
    }
    END {
        for (k in tot) printf "  %12.2f us   %s\n", tot[k], substr(k,1,80)
    }
' "$OUT" | sort -rn | head -15
