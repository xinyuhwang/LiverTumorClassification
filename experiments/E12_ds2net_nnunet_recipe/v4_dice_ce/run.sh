#!/bin/bash
# E12 · v4 — DS²Net Stage 2 with Dice + CE loss (on top of v2). Run on AICR from the repository root.
set -euo pipefail
SET="roi=liver slices=liver liver_includes_tumor=True roi_liver_run=e01_v2_liver_union
     z_spacing=5.0 samples_per_epoch=59314 phase_norm=fixed loss=dice_ce"

# shellcheck disable=SC2086
SBATCH_EXTRA="--partition=rtx-devel --time=01:00:00" \
    bash cluster/submit.sh ds2net --stage 2 --run_name e12_v4_smoke --smoke_test --set $SET

# shellcheck disable=SC2086
bash cluster/submit.sh ds2net --stage 2 --run_name e12_v4_dice_ce --set $SET

# On your laptop:
#   bash cluster/pull_results.sh /work/neu/p2026_0093_neu/herald/results/runs/ds2net/e12_v4_dice_ce \
#        experiments/E12_ds2net_nnunet_recipe/v4_dice_ce/results
