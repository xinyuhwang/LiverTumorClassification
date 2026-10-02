#!/bin/bash
#SBATCH --job-name=herald-nnunet-ensemble
#SBATCH --partition=rtx-batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
# E06 v2: predict HERALD's val and test cases with an ensemble of folds
# (softmax average + nnU-Net's mirroring), then score both with HERALD's metric code.
# Val cases are in no inner fold, so the ensemble is compared with v1 on val.
# Usage: bash cluster/submit.sh nnunet_predict_ensemble
#        NNUNET_FOLDS="1 2 3" NNUNET_RUN=name bash cluster/submit.sh nnunet_predict_ensemble
set -euo pipefail
source "$HERALD_CODE/cluster/env.sh"
ID="${NNUNET_DATASET_ID:-501}"
FOLDS="${NNUNET_FOLDS:-1 2 3 4 5}"
NAME="Dataset$(printf %03d "$ID")_MCTLTDiag"
OUT="$HERALD_RESULTS/runs/nnunet/${NNUNET_RUN:-e06_v2_5fold_ensemble}"
mkdir -p "$OUT/pred_val" "$OUT/pred_test"
echo "git commit: $(git -C "$HERALD_CODE" rev-parse --short HEAD) | folds: $FOLDS"
# shellcheck disable=SC2086
for split in val test; do
    src="$nnUNet_raw/$NAME/images$([ $split = val ] && echo Val || echo Ts)"
    nnUNetv2_predict -i "$src" -o "$OUT/pred_$split" -d "$ID" -c 3d_fullres -f $FOLDS
    python "$HERALD_CODE/nnunet/evaluate_predictions.py" --pred "$OUT/pred_$split" \
        --split "$split" --out "$OUT"
done
