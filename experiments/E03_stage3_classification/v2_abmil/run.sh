#!/bin/bash
# E03 · v2 — ABMIL per-case aggregation. Run on AICR from the repository root.
set -euo pipefail
for bb in efficientnet_b3 vit_b16 swin_tiny swin_base resnet50 unet; do
    SBATCH_EXTRA="--job-name=herald-s3v2-$bb" bash cluster/submit.sh stage3_paper --backbone "$bb" --mil abmil --tag v2
done
# after all six finish:
bash cluster/submit.sh stage3_paper --mode ensemble \
    --members efficientnet_b3_v2,vit_b16_v2,swin_tiny_v2,swin_base_v2,resnet50_v2,unet_v2
