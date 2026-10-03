#!/bin/bash
# E02 · v6 — re-score nnU-Net E06 v1 and v3 predictions with lesion metrics (CPU). Run on AICR from the repository root.
set -euo pipefail
source cluster/env.sh
M=$nnUNet_results/Dataset501_MCTLTDiag
R=$HERALD_RESULTS/runs/nnunet
srun --account="$HERALD_ACCOUNT" --partition=cpu --cpus-per-task=16 --mem=64G --time=02:00:00 bash -c "
  python nnunet/evaluate_predictions.py --workers 16 --split val  --pred $M/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation --out $R/e06_v1_3d_fullres_fold0
  python nnunet/evaluate_predictions.py --workers 16 --split test --pred $R/e06_v1_3d_fullres_fold0/pred_test --out $R/e06_v1_3d_fullres_fold0
  python nnunet/evaluate_predictions.py --workers 16 --split val  --pred $M/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0/validation --out $R/e06_v3_resenc_m_fold0
  python nnunet/evaluate_predictions.py --workers 16 --split test --pred $R/e06_v3_resenc_m_fold0/pred_test --out $R/e06_v3_resenc_m_fold0
"
# On your laptop: pull tumor/ and liver/ of both runs into the E06 version folders' results/
