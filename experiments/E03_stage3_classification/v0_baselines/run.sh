#!/bin/bash
# E03 · v0 — Stage 3 classification baselines on the shared split. Run on AICR from the repository root.
set -euo pipefail

# 1. Paper Stage 3: one job per backbone (unet uses the E05 v2 Stage 2 checkpoint
#    $HERALD_WORK/unet_hybrid/checkpoints/unet_v2_tumor_best.pth)
for bb in efficientnet_b3 vit_b16 swin_tiny swin_base resnet50 unet; do
    SBATCH_EXTRA="--job-name=herald-s3-$bb" bash cluster/submit.sh stage3_paper --backbone "$bb"
done

# 2. DS²Net Stage 3 (4 phases)
bash cluster/submit.sh ds2net --stage 3 --run_name e03_v0_ds2net_cls

# 3. After all six backbones finish: the paper's ensemble, and all six
bash cluster/submit.sh stage3_paper --mode ensemble --members efficientnet_b3,vit_b16,unet
bash cluster/submit.sh stage3_paper --mode ensemble --members efficientnet_b3,vit_b16,swin_tiny,swin_base,resnet50,unet

# On your laptop:
#   rsync -a aicr:/work/neu/p2026_0093_neu/herald/results/unet_hybrid/logs/ experiments/E03_stage3_classification/v0_baselines/results/paper/ \
#         --include 'stage3_paper_*/***' --include '*.json' --exclude '*'
#   bash cluster/pull_results.sh /work/neu/p2026_0093_neu/herald/results/runs/ds2net/e03_v0_ds2net_cls \
#        experiments/E03_stage3_classification/v0_baselines/results/ds2net
