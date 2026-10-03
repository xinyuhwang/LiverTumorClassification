#!/bin/bash
# E06 · v3 — nnU-Net ResEnc M plans, fold 0. Run on AICR from the repository root.
set -euo pipefail
source cluster/env.sh

# 1. Plans only (reuses the 3d_fullres preprocessed data of v1)
srun --account="$HERALD_ACCOUNT" --partition=cpu --cpus-per-task=4 --mem=16G --time=00:30:00 \
    nnUNetv2_plan_experiment -d 501 -pl nnUNetPlannerResEncM

# 2. Train fold 0 (~6-8 h)
NNUNET_PLANS=nnUNetResEncUNetMPlans SBATCH_EXTRA="--job-name=herald-nnunet-resencm" \
    bash cluster/submit.sh nnunet_train

# 3. After training: test prediction + val/test scoring
NNUNET_PLANS=nnUNetResEncUNetMPlans NNUNET_RUN=e06_v3_resenc_m_fold0 bash cluster/submit.sh nnunet_predict

# On your laptop:
#   bash cluster/pull_results.sh /work/neu/p2026_0093_neu/herald/results/runs/nnunet/e06_v3_resenc_m_fold0 \
#        experiments/E06_nnunet_baseline/v3_resenc_m_fold0/results
