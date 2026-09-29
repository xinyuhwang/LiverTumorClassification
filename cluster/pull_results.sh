#!/bin/bash
# cluster/pull_results.sh — run on your LAPTOP. Copies a finished run's small
# files (config, metrics, history, per-case CSV; no checkpoints) into an
# experiment version folder so they are tracked in git.
#   bash cluster/pull_results.sh <remote run path> <experiment version folder>
# e.g.
#   bash cluster/pull_results.sh /work/neu/mygroup/herald/results/runs/ds2net/e01_v1_liver \
#        experiments/E01_liver_labels/v1_official_liver_masks/results
set -euo pipefail
remote="$1"; dest="$2"
mkdir -p "$dest"
rsync -av --include='*.json' --include='*.csv' --include='*.txt' --exclude='*' \
      "aicr-dtn:$remote/" "$dest/"
