#!/bin/bash
#SBATCH --job-name=herald-nnunet-prep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
# E06: build the nnU-Net raw dataset from $HERALD_DATA, then plan + preprocess
# (with nnU-Net's dataset integrity check). Usage: bash cluster/submit.sh nnunet_prep
set -euo pipefail
source "$HERALD_CODE/cluster/env.sh"
ID="${NNUNET_DATASET_ID:-501}"
echo "git commit: $(git -C "$HERALD_CODE" rev-parse --short HEAD)"
python "$HERALD_CODE/nnunet/prepare_dataset.py" --dataset_id "$ID" --workers "$SLURM_CPUS_PER_TASK"
nnUNetv2_plan_and_preprocess -d "$ID" -c 3d_fullres --verify_dataset_integrity \
    -np "$SLURM_CPUS_PER_TASK"
