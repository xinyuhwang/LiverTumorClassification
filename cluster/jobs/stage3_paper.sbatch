#!/bin/bash
#SBATCH --job-name=herald-stage3
#SBATCH --partition=rtx-batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# Usage: bash cluster/submit.sh stage3_paper --backbone efficientnet_b3
#        bash cluster/submit.sh stage3_paper --mode ensemble
set -euo pipefail
source "$HERALD_CODE/cluster/env.sh"
cd "$HERALD_CODE/unet_hybrid"
echo "git commit: $(git -C "$HERALD_CODE" rev-parse --short HEAD)"
python stage3_paper.py --data_dir "$HERALD_DATA" \
    --ckpt_dir "$HERALD_WORK/unet_hybrid/checkpoints" \
    --log_dir "$HERALD_WORK/unet_hybrid/logs" --num_workers 8 "$@"
rsync -a --exclude '*_latest.pth' "$HERALD_WORK/unet_hybrid/" "$HERALD_RESULTS/unet_hybrid/"
