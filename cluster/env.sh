# cluster/env.sh — shared settings for every AICR job. Source it:
#   source cluster/env.sh
# Project allocation p2026_0093_neu (Northeastern). Override any value by
# exporting it before sourcing this file.

# Slurm account (list yours with: sacctmgr show user $USER withassoc format=user,account -p)
export HERALD_ACCOUNT="${HERALD_ACCOUNT:-p2026_0093_neu}"

# Long-lived group storage (1 TB, 7-day snapshots): raw data, prepared data, results
export HERALD_STORE="${HERALD_STORE:-/work/neu/p2026_0093_neu/herald}"

# Fast scratch (10 TiB, files older than 30 days are PURGED). Caches and checkpoints.
export HERALD_SCRATCH="${HERALD_SCRATCH:-/scratch/$USER/herald}"

# Code checkout on AICR (home is 100 GiB and snapshotted)
export HERALD_CODE="${HERALD_CODE:-$HOME/LiverTumorClassification}"

# Python environment created by cluster/setup_env.sh
export HERALD_VENV="${HERALD_VENV:-$HOME/envs/herald}"

# Derived paths used by the training scripts
export HERALD_RAW="$HERALD_STORE/raw"                 # 180 GB of .tar archives
export HERALD_DATA="$HERALD_STORE/mct_ltdiag"         # prepared per-case NIfTI
export HERALD_WORK="$HERALD_SCRATCH/work"             # caches, runs, checkpoints
export HERALD_RESULTS="$HERALD_STORE/results"         # copies of finished runs

# nnU-Net v2 (E06): raw = symlinks + labels (small), preprocessed = large and
# rebuildable (scratch), results = checkpoints and predictions (keep)
export nnUNet_raw="$HERALD_STORE/nnunet/raw"
export nnUNet_preprocessed="$HERALD_SCRATCH/nnunet/preprocessed"
export nnUNet_results="$HERALD_STORE/nnunet/results"

if [ -n "${SLURM_JOB_ID:-}" ] ||[ "${HERALD_LOAD_MODULES:-0}" = 1 ]; then
    module load miniforge3 2>/dev/null || module load conda 2>/dev/null || true
    module load cuda 2>/dev/null || true
fi
[ -f "$HERALD_VENV/bin/activate" ] && source "$HERALD_VENV/bin/activate"
export PYTHONUNBUFFERED=1
mkdir -p "$HERALD_WORK" 2>/dev/null || true
