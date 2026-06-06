#!/usr/bin/env bash
# Profile the Hopper-spike RoPE+causal kernel with the same metric set
# as the plain-causal profile. The kernel name is the same
# (_hopper_causal_fwd_kernel) but the inner loop runs the rotation
# block, so register pressure / wait stalls / SMEM may all differ.
#
# Diagnostic question (Session 7 prong B1): WHY does the RoPE+causal
# kernel scale worse than plain causal at large N? Three hypotheses to
# distinguish:
#
#   H1. Register spill from holding cos/sin/K_rot_half in fp32 at
#       once. Look for high stalled_wait + lower-than-expected
#       active warps relative to the plain-causal baseline.
#   H2. SMEM pressure forcing fewer blocks per SM. Compare
#       launch__shared_mem_per_block_allocated to the plain case
#       and check launch__waves_per_multiprocessor.
#   H3. The K_rot_half second HBM load. Look at DRAM throughput;
#       if it's noticeably higher than plain causal, this is real.
#       (Back-of-envelope says <1% but counter data is authoritative.)
#
# Usage:
#   bash scripts/run_hopper_spike_rope_ncu.sh [SEQLEN]
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
    echo "ERROR: ncu not found." >&2
    exit 1
fi

OUT_DIR="results/ncu"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/ncu_hopper_rope_causal_${SEQLEN}_fp16.csv"

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

echo "=== ncu profiling: hopper_spike rope_causal N=$SEQLEN fp16 ==="
echo "    writing $OUT"

"$NCU" \
    --target-processes all \
    --launch-skip 3 \
    --launch-count 1 \
    --kernel-name '_hopper_causal_fwd_kernel' \
    --csv \
    --metrics "$METRICS_CSV" \
    python -m benchmarks.hopper_spike_ncu \
        --seqlen "$SEQLEN" --warmup 4 --rope \
    > "$OUT" 2> "${OUT}.log"

echo "[ok] $OUT"
echo ""
echo "Quick read:"
grep -E "sm__throughput|hmma|dram__throughput|warps_active|long_scoreboard|short_scoreboard|stalled_wait|registers_per_thread|shared_mem|block_size|waves" "$OUT" \
    | head -15
