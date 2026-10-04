#!/bin/bash
# E03 · v5 — 4-phase input, mask pooling, 5-fold OOF. Run on AICR from the repository root.
set -euo pipefail
SBATCH_EXTRA="--partition=rtx-devel --time=01:00:00 --job-name=herald-s3v5-smoke" \
    bash cluster/submit.sh stage3_paper --backbone resnet50 --input 4phase --pooling mask --cv_folds 5 --tag v5smoke --smoke_test
for bb in efficientnet_b3 vit_b16 swin_tiny swin_base resnet50 unet; do
    SBATCH_EXTRA="--job-name=herald-s3cv5-$bb" bash cluster/submit.sh stage3_paper --backbone "$bb" \
        --input 4phase --pooling mask --cv_folds 5 --tag cv_v5
done
# after all six finish:
bash cluster/submit.sh stage3_paper --mode ensemble \
    --members efficientnet_b3_cv_v5,vit_b16_cv_v5,swin_tiny_cv_v5,swin_base_cv_v5,resnet50_cv_v5,unet_cv_v5
