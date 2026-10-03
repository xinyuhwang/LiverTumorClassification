#!/bin/bash
# E12 · v1 — DS²Net Stage 2 at native 5 mm slice spacing. Run on AICR from the repository root.
# Needs E01 v2's saved liver masks (run e01_v2_liver_union) under $HERALD_WORK/runs/ds2net/.
set -euo pipefail
SET="roi=liver slices=liver liver_includes_tumor=True roi_liver_run=e01_v2_liver_union
     z_spacing=5.0 samples_per_epoch=59314"

# Smoke test first (~10 min, devel partition): 4 train / 2 val / 2 test cases, 2 epochs
# shellcheck disable=SC2086
SBATCH_EXTRA="--partition=rtx-devel --time=01:00:00" \
    bash cluster/submit.sh ds2net --stage 2 --run_name e12_v1_smoke --smoke_test --set $SET

# Full run (~6 h)
# shellcheck disable=SC2086
bash cluster/submit.sh ds2net --stage 2 --run_name e12_v1_native_z --set $SET

# On your laptop, once finished:
#   bash cluster/pull_results.sh /work/neu/p2026_0093_neu/herald/results/runs/ds2net/e12_v1_native_z \
#        experiments/E12_ds2net_nnunet_recipe/v1_native_z/results
