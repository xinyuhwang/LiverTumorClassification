#!/bin/bash
#SBATCH --job-name=herald-nnunet-train
#SBATCH --partition=rtx-batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
# E06: train nnU-Net 3d_fullres on fold 0 (= HERALD's 360 train / 78 val split).
# Usage: bash cluster/submit.sh nnunet_train            # first job
#        bash cluster/submit.sh nnunet_train --c        # continue after the 24 h limit
# At the end nnU-Net writes validation predictions to .../fold_0/validation/.
set -euo pipefail
source "$HERALD_CODE/cluster/env.sh"
ID="${NNUNET_DATASET_ID:-501}"
export nnUNet_n_proc_DA="${SLURM_CPUS_PER_TASK}"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "git commit: $(git -C "$HERALD_CODE" rev-parse --short HEAD)"
nnUNetv2_train "$ID" 3d_fullres 0 "$@"
