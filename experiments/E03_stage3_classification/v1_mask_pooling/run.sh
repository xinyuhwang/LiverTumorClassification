#!/bin/bash
# E03 · v1 — mask-weighted pooling + tumor-area slice weighting. Run on AICR from the repository root.
set -euo pipefail
for bb in efficientnet_b3 vit_b16 swin_tiny swin_base resnet50 unet; do
    SBATCH_EXTRA="--job-name=herald-s3v1-$bb" bash cluster/submit.sh stage3_paper --backbone "$bb" --pooling mask --tag v1
done
# after all six finish:
bash cluster/submit.sh stage3_paper --mode ensemble \
    --members efficientnet_b3_v1,vit_b16_v1,swin_tiny_v1,swin_base_v1,resnet50_v1,unet_v1
