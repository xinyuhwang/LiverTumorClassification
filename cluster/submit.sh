#!/bin/bash
# cluster/submit.sh — submit a HERALD job with the account from env.sh.
#   bash cluster/submit.sh <job> [args passed to the script]
# Jobs (cluster/jobs/<job>.sbatch):
#   prepare_data                         raw archives → prepared per-case NIfTI
#   ds2net       --stage N [...]         ds2net/train.py
#   unet_hybrid  --stage N [...]         unet_hybrid/train.py
#   stage3_paper --backbone X [...]      unet_hybrid/stage3_paper.py
# Extra sbatch options go in SBATCH_EXTRA, e.g.
#   SBATCH_EXTRA="--partition=b200-batch --time=12:00:00" bash cluster/submit.sh ds2net --stage 1
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
job="$1"; shift
[ "$HERALD_ACCOUNT" = "CHANGE_ME" ] && { echo "Set HERALD_ACCOUNT in cluster/env.sh"; exit 1; }
mkdir -p "$HERALD_SCRATCH/logs"
# shellcheck disable=SC2086
sbatch --account="$HERALD_ACCOUNT" --output="$HERALD_SCRATCH/logs/%x-%j.out" \
       ${SBATCH_EXTRA:-} "$HERE/jobs/$job.sbatch" "$@"
