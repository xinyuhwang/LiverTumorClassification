#!/bin/bash
# E03 · v3 — DS²Net Stage 3 with phase_norm=fixed. Run on AICR from the repository root.
set -euo pipefail
bash cluster/submit.sh ds2net --stage 3 --run_name e03_v3_ds2net_fixed_norm --set phase_norm=fixed
# On your laptop:
#   bash cluster/pull_results.sh /work/neu/p2026_0093_neu/herald/results/runs/ds2net/e03_v3_ds2net_fixed_norm \
#        experiments/E03_stage3_classification/v3_ds2net_fixed_norm/results
