#!/usr/bin/env bash
# AttnFuse — first-run setup on an H100 host via SSH.
#
# Usage on the H100:
#   git clone <this repo>            # if not already cloned
#   cd AttnFuse
#   bash scripts/setup_h100.sh
#
# The script:
#   1. Verifies the GPU is sm_90 (Hopper).
#   2. Creates a conda env (or uses an existing one).
#   3. Installs PyTorch with CUDA 12.1 wheels + Triton + dev deps.
#   4. Pip-installs AttnFuse in editable mode.
#   5. Runs the test suite to confirm everything compiles and matches
#      the reference within FA2 tolerance.
#
# After this, run scripts/run_h100_benchmarks.sh to capture the
# paper-grade numbers.
set -euo pipefail

ENV_NAME="${ATTNFUSE_ENV:-attnfuse}"
CONDA_HOME="${CONDA_HOME:-$HOME/miniconda3}"

echo "=== AttnFuse H100 setup ==="
echo

# --- 1. GPU check ---
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found on PATH. Is this an H100 host?" >&2
    exit 1
fi
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version \
           --format=csv,noheader,nounits
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1)
if [[ "$CC" != "9.0" && "$CC" != "9.0a" ]]; then
    echo
    echo "WARNING: compute capability is $CC, not 9.0 (Hopper)."
    echo "AttnFuse will still run; the Hopper-specific tile tables won't be used."
    echo "Continue anyway? (Ctrl-C to abort, Enter to continue)"
    read -r _
fi

# --- 2. Conda env ---
if [[ ! -d "$CONDA_HOME" ]]; then
    echo
    echo "Miniconda not found at $CONDA_HOME; installing it..."
    curl -L -o /tmp/miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_HOME"
fi

# shellcheck disable=SC1091
source "$CONDA_HOME/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo
    echo "Creating conda env '$ENV_NAME' (python 3.11)..."
    conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"

# --- 3. Dependencies ---
echo
echo "Installing PyTorch (cu121) + Triton + dev deps..."
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install triton==3.1.0
pip install pytest hypothesis pandas matplotlib

# --- 4. Editable install ---
echo
echo "Installing AttnFuse..."
pip install -e .

# --- 5. Smoke test ---
echo
echo "=== Sanity: import + GPU detection ==="
python -c "
import torch, triton, attnfuse as af
print(f'  AttnFuse  version : {af.__version__}')
print(f'  PyTorch         : {torch.__version__}')
print(f'  Triton          : {triton.__version__}')
print(f'  CUDA            : {torch.version.cuda}')
print(f'  GPU             : {torch.cuda.get_device_name(0)}')
print(f'  Compute cap     : sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}')
print(f'  SM count        : {torch.cuda.get_device_properties(0).multi_processor_count}')
"

echo
echo "=== Running test suite (this triggers the first-time JIT compile) ==="
python -m pytest tests/ -q
echo
echo "[ok] H100 setup complete."
echo "Next: bash scripts/run_h100_benchmarks.sh"
