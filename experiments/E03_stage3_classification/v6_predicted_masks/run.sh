#!/bin/bash
# E03 · v6 — score the E03 v1 models on nnU-Net (E06 v3) predicted tumor masks. Run on AICR from the repository root.
set -euo pipefail
source cluster/env.sh
VAL=$nnUNet_results/Dataset501_MCTLTDiag/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0/validation
TEST=$HERALD_RESULTS/runs/nnunet/e06_v3_resenc_m_fold0/pred_test
for bb in efficientnet_b3 vit_b16 swin_tiny swin_base resnet50 unet; do
    SBATCH_EXTRA="--job-name=herald-s3v6-$bb" bash cluster/submit.sh stage3_paper --backbone "$bb" --pooling mask \
        --eval_only --ckpt_name "${bb}_v1" --mask_source pred --pred_val_dir "$VAL" --pred_test_dir "$TEST" --tag v6pred
done
# after all six finish:
bash cluster/submit.sh stage3_paper --mode ensemble \
    --members efficientnet_b3_v6pred,vit_b16_v6pred,swin_tiny_v6pred,swin_base_v6pred,resnet50_v6pred,unet_v6pred
