#!/usr/bin/env bash
# Profile the Hopper-spike causal kernel with the same metric set as
# scripts/run_ncu_profile.sh. Run after the wall-clock spike confirms
# numerics + measurable improvement.
#
# Usage:
#   bash scripts/run_hopper_spike_ncu.sh [SEQLEN]
#
# Writes results/ncu/ncu_hopper_causal_<SEQLEN>_fp16.csv
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
if ! command -v "$NCU" >/dev/null 2>&1; then
    echo "ERROR: ncu not found. Install Nsight Compute or set NCU=<path>." >&2
    exit 1
fi

OUT_DIR="results/ncu"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/ncu_hopper_causal_${SEQLEN}_fp16.csv"

# Same metric set as the production profile so the two are directly comparable.
METRICS=(
    sm__throughput.avg.pct_of_peak_sustained_elapsed
    sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed
    dram__bytes_read.sum.per_second
    dram__bytes_write.sum.per_second
    dram__throughput.avg.pct_of_peak_sustained_elapsed
    l1tex__throughput.avg.pct_of_peak_sustained_elapsed
    lts__throughput.avg.pct_of_peak_sustained_elapsed
    sm__warps_active.avg.pct_of_peak_sustained_active
    smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
    smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct
    smsp__warp_issue_stalled_membar_per_warp_active.pct
    smsp__warp_issue_stalled_wait_per_warp_active.pct
    smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct
    launch__registers_per_thread
    launch__shared_mem_per_block_allocated
    launch__block_size
    launch__waves_per_multiprocessor
)
METRICS_CSV=$(IFS=, ; echo "${METRICS[*]}")

echo "=== ncu profiling: hopper_spike causal N=$SEQLEN fp16 ==="
echo "    writing $OUT"

"$NCU" \
    --target-processes all \
    --launch-skip 3 \
    --launch-count 1 \
    --kernel-name '_hopper_causal_fwd_kernel' \
    --csv \
    --metrics "$METRICS_CSV" \
    python -m benchmarks.hopper_spike_ncu \
        --seqlen "$SEQLEN" --warmup 4 \
    > "$OUT" 2> "${OUT}.log"

echo "[ok] $OUT"
echo ""
echo "Quick read:"
grep -E "sm__throughput|hmma|dram__throughput|warps_active|long_scoreboard|short_scoreboard|stalled_wait|registers_per_thread|shared_mem|block_size" "$OUT" \
    | head -15
