#!/bin/bash
# E06 · v2 — nnU-Net 5-fold ensemble (inner CV over the 360 train cases).
# Run on AICR from the repository root, after v1 (same dataset, plans and preprocessing).
set -euo pipefail
source cluster/env.sh

# 1. Add folds 1-5 to splits_final.json (fold 0 = v1 unchanged) and link the val images
python nnunet/prepare_dataset.py --cv_folds 5 --splits_only

# 2. Train the five folds (~5.5 h each; parallel if the queue allows)
for f in 1 2 3 4 5; do
    NNUNET_FOLD=$f SBATCH_EXTRA="--job-name=herald-nnunet-f$f" bash cluster/submit.sh nnunet_train
done

# 3. After all five finish: ensemble prediction + scoring of val and test
bash cluster/submit.sh nnunet_predict_ensemble

# On your laptop:
#   bash cluster/pull_results.sh /work/neu/p2026_0093_neu/herald/results/runs/nnunet/e06_v2_5fold_ensemble \
#        experiments/E06_nnunet_baseline/v2_5fold_ensemble/results
