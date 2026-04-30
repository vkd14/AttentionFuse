#!/usr/bin/env bash
# One-shot environment setup for AttnFuse on RTX 3090 Ti / Linux.
set -euo pipefail

PYTHON=${PYTHON:-python3.11}
$PYTHON -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install -e ".[bench,dev]"

python - <<'PY'
import torch, triton
print(f"torch    : {torch.__version__}")
print(f"triton   : {triton.__version__}")
print(f"cuda     : {torch.cuda.is_available()} -- {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU'}")
PY
