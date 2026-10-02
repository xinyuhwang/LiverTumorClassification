#!/bin/bash
#SBATCH --job-name=herald-nnunet-predict
#SBATCH --partition=rtx-batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
# E06: predict the held-out test cases with the fold-0 model, then score
# validation and test with HERALD's metric code.
# Usage: bash cluster/submit.sh nnunet_predict
set -euo pipefail
source "$HERALD_CODE/cluster/env.sh"
ID="${NNUNET_DATASET_ID:-501}"
NAME="Dataset$(printf %03d "$ID")_MCTLTDiag"
MODEL="$nnUNet_results/$NAME/nnUNetTrainer__nnUNetPlans__3d_fullres"
OUT="$HERALD_RESULTS/runs/nnunet/e06_v1_3d_fullres_fold0"
mkdir -p "$OUT/pred_test"
echo "git commit: $(git -C "$HERALD_CODE" rev-parse --short HEAD)"
nnUNetv2_predict -i "$nnUNet_raw/$NAME/imagesTs" -o "$OUT/pred_test" \
    -d "$ID" -c 3d_fullres -f 0
python "$HERALD_CODE/nnunet/evaluate_predictions.py" --pred "$MODEL/fold_0/validation" \
    --split val --out "$OUT"
python "$HERALD_CODE/nnunet/evaluate_predictions.py" --pred "$OUT/pred_test" \
    --split test --out "$OUT"
