#!/bin/bash
# E02 v6 follow-up: contact sheets of GT tumor components < 0.1 ml (val), with E06 v3 predictions. CPU, ~2 min.
set -euo pipefail
source cluster/env.sh
PRED=$nnUNet_results/Dataset501_MCTLTDiag/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0/validation
srun --account="$HERALD_ACCOUNT" --partition=cpu --cpus-per-task=4 --mem=32G --time=00:30:00 \
    python data_prep/review_lesion_fragments.py --pred_dir "$PRED" \
    --out "$HERALD_RESULTS/fragment_review" \
    --cases 230525a4 231025c16 240722e100 240620a22 231025c02 240112d26
# On your laptop: rsync -a aicr:/work/neu/p2026_0093_neu/herald/results/fragment_review/ experiments/E02_evaluation/v6_failures_lesions/fragment_review/
