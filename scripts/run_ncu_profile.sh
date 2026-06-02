#!/usr/bin/env bash
# Run Nsight Compute on one AttnFuse forward kernel launch.
#
# Usage:
#   bash scripts/run_ncu_profile.sh <variant> <seqlen> <dtype>
#
# Examples:
#   bash scripts/run_ncu_profile.sh causal         4096 fp16
#   bash scripts/run_ncu_profile.sh sliding_window 4096 fp16
#
# Writes results/ncu_<variant>_<seqlen>_<dtype>.csv next to the other
# benchmark outputs. Requires Nsight Compute 2023.2+ (CUDA 12.x).
#
# Requires root or `--allow-host-syncing` permissions on most clusters
# (the kernel sets `cudaSetDevice` which ncu needs to interpose).
set -euo pipefail

VARIANT="${1:-causal}"
SEQLEN="${2:-4096}"
DTYPE="${3:-fp16}"

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

OUT_DIR="results"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/ncu_${VARIANT}_${SEQLEN}_${DTYPE}.csv"

# Metrics map. These match Nsight Compute's section sets but listed
# explicitly so the report doesn't depend on which section-set version
# the local install has.
METRICS=(
    # Overall SM throughput (compute pipe + memory pipe combined)
    sm__throughput.avg.pct_of_peak_sustained_elapsed

    # Tensor core utilisation (fp16 HMMA pipe). On Hopper, prefer
    # sm__pipe_tensor_op_hgemm_cycles_active when measuring WGMMA.
    sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed

    # HBM read / write bandwidth and percentage of peak
    dram__bytes_read.sum.per_second
    dram__bytes_write.sum.per_second
    dram__throughput.avg.pct_of_peak_sustained_elapsed

    # L1 / L2 throughput (interesting for the bandwidth-bound regime)
    l1tex__throughput.avg.pct_of_peak_sustained_elapsed
    lts__throughput.avg.pct_of_peak_sustained_elapsed

    # Occupancy: how many warps are actually resident
    sm__warps_active.avg.pct_of_peak_sustained_active

    # Warp stall breakdown -- this is what the analytical roofline misses
    smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct
    smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct
    smsp__warp_issue_stalled_membar_per_warp_active.pct
    smsp__warp_issue_stalled_wait_per_warp_active.pct
    smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct

    # Launch attributes
    launch__registers_per_thread
    launch__shared_mem_per_block_allocated
    launch__block_size
    launch__waves_per_multiprocessor
)
METRICS_CSV=$(IFS=, ; echo "${METRICS[*]}")

echo "=== ncu profiling: variant=$VARIANT N=$SEQLEN dtype=$DTYPE ==="
echo "    writing $OUT"

# --launch-skip 3 -- skip the warmup compile launches; profile the 4th.
# --csv          -- machine-readable output.
# --target-processes all -- in case PyTorch spawns helpers.
"$NCU" \
    --target-processes all \
    --launch-skip 3 \
    --launch-count 1 \
    --kernel-name 'attnfuse_fwd_kernel' \
    --csv \
    --metrics "$METRICS_CSV" \
    python -m benchmarks.ncu_profile \
        --variant "$VARIANT" --seqlen "$SEQLEN" --dtype "$DTYPE" \
        --warmup 4 \
    > "$OUT" 2> "${OUT}.log"

echo "[ok] $OUT"
echo ""
echo "Quick read:"
grep -E "sm__throughput|hmma|dram__throughput|warps_active|long_scoreboard|registers_per_thread|shared_mem" "$OUT" \
    | head -10
