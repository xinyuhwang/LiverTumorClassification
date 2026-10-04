#!/bin/bash
# E03 · v4 — 5-fold cross-validated evaluation of v0, v1, v2. Run on AICR from the repository root.
set -euo pipefail
# smoke test first (2 folds on 70 cases, 2 epochs; devel partition)
SBATCH_EXTRA="--partition=rtx-devel --time=01:00:00 --job-name=herald-s3cv-smoke" \
    bash cluster/submit.sh stage3_paper --backbone resnet50 --cv_folds 5 --tag cvsmoke --smoke_test

for bb in efficientnet_b3 vit_b16 swin_tiny swin_base resnet50 unet; do
    SBATCH_EXTRA="--job-name=herald-s3cv0-$bb" bash cluster/submit.sh stage3_paper --backbone "$bb" --cv_folds 5 --tag cv_v0
    SBATCH_EXTRA="--job-name=herald-s3cv1-$bb" bash cluster/submit.sh stage3_paper --backbone "$bb" --cv_folds 5 --pooling mask --tag cv_v1
    SBATCH_EXTRA="--job-name=herald-s3cv2-$bb" bash cluster/submit.sh stage3_paper --backbone "$bb" --cv_folds 5 --mil abmil --tag cv_v2
done
# after all 18 finish:
for t in cv_v0 cv_v1 cv_v2; do
    bash cluster/submit.sh stage3_paper --mode ensemble \
        --members efficientnet_b3_$t,vit_b16_$t,swin_tiny_$t,swin_base_$t,resnet50_$t,unet_$t
done
