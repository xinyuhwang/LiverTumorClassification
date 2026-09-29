#!/bin/bash
# cluster/setup_env.sh — create the Python environment on AICR.
# Run inside a short GPU session so the CUDA build of PyTorch is verified:
#   salloc --partition=rtx-devel --gpus=1 --cpus-per-task=4 --mem=16G --time=00:45:00
#   bash cluster/setup_env.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
HERALD_LOAD_MODULES=1 source "$HERE/env.sh"

if [ ! -d "$HERALD_VENV" ]; then
    python3 -m venv "$HERALD_VENV"
fi
source "$HERALD_VENV/bin/activate"
python -m pip install --upgrade pip
# PyTorch wheels bundle their own CUDA runtime; cu128 supports RTX PRO 6000
# (Blackwell) and B200. Change the index URL if `nvidia-smi` reports an older driver.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$HERE/../requirements.txt"

python - <<'EOF'
import torch, timm, nibabel, skimage
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
    x = torch.randn(1024, 1024, device="cuda"); print("matmul ok:", float((x @ x).sum()) != 0)
EOF
echo "Environment ready: $HERALD_VENV"
