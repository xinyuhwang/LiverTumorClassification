#!/bin/bash
#SBATCH --job-name=herald-prepare
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
# Raw .tar archives ($HERALD_RAW) → prepared per-case NIfTI ($HERALD_DATA) + manifest.csv.
# Download the archives first, on the data-transfer node (see docs/AICR.md).
set -euo pipefail
source "$HERALD_CODE/cluster/env.sh"
python "$HERALD_CODE/data_prep/prepare_mct_ltdiag.py" \
    --raw "$HERALD_RAW" --out "$HERALD_DATA" --workers "$SLURM_CPUS_PER_TASK" "$@"
