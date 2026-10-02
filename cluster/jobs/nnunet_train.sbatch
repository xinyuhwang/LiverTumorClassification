#!/bin/bash
#SBATCH --job-name=herald-nnunet-train
#SBATCH --partition=rtx-batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
# E06: train nnU-Net 3d_fullres on one fold of splits_final.json.
#   fold 0     = HERALD's 360 train / 78 val split (v1)
#   folds 1..5 = inner CV over the 360 train cases (v2; prepare_dataset.py --cv_folds 5)
# Usage: bash cluster/submit.sh nnunet_train                  # fold 0
#        NNUNET_FOLD=3 bash cluster/submit.sh nnunet_train    # another fold
#        bash cluster/submit.sh nnunet_train --c              # continue after the 24 h limit
# At the end nnU-Net writes validation predictions to .../fold_<N>/validation/.
set -euo pipefail
source "$HERALD_CODE/cluster/env.sh"
ID="${NNUNET_DATASET_ID:-501}"
FOLD="${NNUNET_FOLD:-0}"
SPLITS="$nnUNet_preprocessed/Dataset$(printf %03d "$ID")_MCTLTDiag/splits_final.json"
# nnU-Net silently trains on a random 80:20 split if the fold isn't in the file
python -c "import json, sys; n = len(json.load(open('$SPLITS')))
sys.exit(f'fold $FOLD not in $SPLITS ({n} folds)') if $FOLD >= n else print(f'fold $FOLD of {n} in splits_final.json')"
export nnUNet_n_proc_DA="${SLURM_CPUS_PER_TASK}"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "git commit: $(git -C "$HERALD_CODE" rev-parse --short HEAD)"
nnUNetv2_train "$ID" 3d_fullres "$FOLD" "$@"
