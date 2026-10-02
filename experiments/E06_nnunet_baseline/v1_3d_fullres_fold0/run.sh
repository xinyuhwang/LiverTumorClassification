#!/bin/bash
# E06 · v1 — nnU-Net 3d_fullres fold 0. Run on AICR from the repository root.
set -euo pipefail
source cluster/env.sh
python -m pip install "nnunetv2>=2.5"            # once, into ~/envs/herald

bash cluster/submit.sh nnunet_prep               # CPU: raw dataset + plan/preprocess
bash cluster/submit.sh nnunet_train              # GPU, after prep finishes
#   bash cluster/submit.sh nnunet_train --c     # continue after each 24 h limit
bash cluster/submit.sh nnunet_predict            # after training: test prediction + scoring

# On your laptop:
#   bash cluster/pull_results.sh /work/neu/p2026_0093_neu/herald/results/runs/nnunet/e06_v1_3d_fullres_fold0 \
#        experiments/E06_nnunet_baseline/v1_3d_fullres_fold0/results
