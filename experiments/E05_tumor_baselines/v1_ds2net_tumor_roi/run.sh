#!/bin/bash
# E05 · v1 — DS²Net Stage 2 on the liver region. Run on AICR from the repository root.
# Needs E01 v2's saved liver masks (run e01_v2_liver_union) under $HERALD_WORK/runs/ds2net/.
set -euo pipefail
bash cluster/submit.sh ds2net --stage 2 --run_name e05_v1_ds2net_tumor_roi \
    --set roi=liver slices=liver liver_includes_tumor=True roi_liver_run=e01_v2_liver_union

# On your laptop, once finished:
#   bash cluster/pull_results.sh /work/neu/p2026_0093_neu/herald/results/runs/ds2net/e05_v1_ds2net_tumor_roi \
#        experiments/E05_tumor_baselines/v1_ds2net_tumor_roi/results
