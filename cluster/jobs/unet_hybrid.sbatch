#!/bin/bash
#SBATCH --job-name=herald-unet
#SBATCH --partition=rtx-batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
# Usage: bash cluster/submit.sh unet_hybrid --stage 1 [--resume]
set -euo pipefail
source "$HERALD_CODE/cluster/env.sh"
cd "$HERALD_CODE/unet_hybrid"
echo "git commit: $(git -C "$HERALD_CODE" rev-parse --short HEAD)"
python train.py --data_dir "$HERALD_DATA" \
    --ckpt_dir "$HERALD_WORK/unet_hybrid/checkpoints" \
    --log_dir "$HERALD_WORK/unet_hybrid/logs" "$@"
rsync -a --exclude '*_latest.pth' "$HERALD_WORK/unet_hybrid/" "$HERALD_RESULTS/unet_hybrid/"
