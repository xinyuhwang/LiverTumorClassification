#!/bin/bash
# E01 · v1 — official liver masks. Run on AICR from the repository root,
# after the one-time setup in docs/AICR.md (env, dataset download).
set -euo pipefail
source cluster/env.sh

# 1. Prepare data + QA (CPU job), then summarise the manifest
bash cluster/submit.sh prepare_data
#    when it has finished:
python data_prep/summarize_manifest.py "$HERALD_DATA/manifest.csv" \
    --out experiments/E01_liver_labels/v1_official_liver_masks/results/data_qa.json

# 2. Stage 1 on real liver labels (GPU job; resubmit with --resume if it hits 24 h)
bash cluster/submit.sh ds2net --stage 1 --run_name e01_v1_liver --save_liver_masks

# 3. On your laptop, copy the metrics into this folder:
#    bash cluster/pull_results.sh $HERALD_STORE/results/runs/ds2net/e01_v1_liver \
#         experiments/E01_liver_labels/v1_official_liver_masks/results
