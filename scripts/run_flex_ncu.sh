#!/usr/bin/env bash
# Profile the torch.compile'd flex_attention causal forward with the
# same metric set as the AttnFuse spike. Setting an HMMA-pipe upper
# bound at this shape.
#
# Inductor names its compiled flex kernel something like
# `triton_per_fused__flex_attention_*` -- the regex below catches it.
# If multiple Triton kernels match (sub-kernels for the block-mask
# bookkeeping, etc.), the CSV will contain all of them; the relevant
# one is the highest-cycle entry.
#
# Usage:
#   bash scripts/run_flex_ncu.sh [SEQLEN]
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
OUT="$OUT_DIR/ncu_flex_${SEQLEN}_fp16.csv"

# Same metric set as run_ncu_profile.sh so the numbers are directly
# comparable to the AttnFuse production and Hopper-spike profiles.
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

echo "=== ncu profiling: flex_attention causal N=$SEQLEN fp16 ==="
echo "    writing $OUT"

# --launch-skip 8 -- skip torch.compile autotune + warmup launches.
# Increase if the chosen kernel is still the wrong one (you'll see
# multiple distinct Kernel Name values in the CSV).
# --kernel-name-base mangled -- use the demangled name for filtering.
# --kernel-name regex:... -- pick any compiled Triton kernel.
"$NCU" \
    --target-processes all \
    --launch-skip 8 \
    --launch-count 4 \
    --kernel-name-base mangled \
    --kernel-name 'regex:triton.*' \
    --csv \
    --metrics "$METRICS_CSV" \
    python -m benchmarks.flex_ncu \
        --seqlen "$SEQLEN" --warmup 6 \
    > "$OUT" 2> "${OUT}.log"

echo "[ok] $OUT"
echo ""
echo "Distinct kernel names seen (the attention kernel should dominate):"
awk -F',' 'NR>1 {print $5}' "$OUT" | sort -u | head -10
echo ""
echo "Quick read for each kernel:"
grep -E "Kernel Name|sm__throughput|hmma|dram__throughput|warps_active|stalled_wait|registers_per_thread" "$OUT" \
    | head -40
