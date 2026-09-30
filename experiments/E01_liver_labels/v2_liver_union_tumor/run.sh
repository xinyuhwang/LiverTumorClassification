#!/bin/bash
# E01 · v2 — liver label = official liver mask ∪ tumor mask.
# Run on AICR from the repository root. Data is already prepared (see v1).
set -euo pipefail
bash cluster/submit.sh ds2net --stage 1 --run_name e01_v2_liver_union \
    --save_liver_masks --set liver_includes_tumor=True

# On your laptop, once the job has finished:
#   bash cluster/pull_results.sh /work/neu/p2026_0093_neu/herald/results/runs/ds2net/e01_v2_liver_union \
#        experiments/E01_liver_labels/v2_liver_union_tumor/results
