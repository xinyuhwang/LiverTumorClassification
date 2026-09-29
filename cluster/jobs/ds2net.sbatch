#!/bin/bash
#SBATCH --job-name=herald-ds2net
#SBATCH --partition=rtx-batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
# Usage: bash cluster/submit.sh ds2net --stage 1 --run_name e01_v1_liver [--resume]
# Jobs longer than 24 h: resubmit the same command with --resume.
set -euo pipefail
source "$HERALD_CODE/cluster/env.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv
cd "$HERALD_CODE/ds2net"
echo "git commit: $(git -C "$HERALD_CODE" rev-parse --short HEAD)"
python train.py --data_dir "$HERALD_DATA" --work_dir "$HERALD_WORK" \
    --workers "$SLURM_CPUS_PER_TASK" --num_workers 8 "$@"
# keep results off purgeable scratch (skip the large resume checkpoint)
rsync -a --exclude last.pth "$HERALD_WORK/runs/" "$HERALD_RESULTS/runs/"
