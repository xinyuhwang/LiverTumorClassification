#!/usr/bin/env bash
set -euo pipefail

# The Conda env can live on a shared disk via a prefix path (recommended when
# home space is small):
#   export LIVER_TUMOR_CONDA_PREFIX=/shared/space/envs/liver-tumor-seg
# Otherwise a named env is created in the default Conda location.
ENV_PREFIX="${LIVER_TUMOR_CONDA_PREFIX:-}"
ENV_NAME="${LIVER_TUMOR_CONDA_ENV:-liver-tumor-seg}"
CONFIG="${1:-configs/phase1_server_audit.yaml}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH" >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if [ -n "${ENV_PREFIX}" ]; then
  if [ -d "${ENV_PREFIX}" ]; then
    conda env update -p "${ENV_PREFIX}" -f environment.yml --prune
  else
    conda env create -p "${ENV_PREFIX}" -f environment.yml
  fi
  conda activate "${ENV_PREFIX}"
else
  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda env update -n "${ENV_NAME}" -f environment.yml --prune
  else
    conda env create -n "${ENV_NAME}" -f environment.yml
  fi
  conda activate "${ENV_NAME}"
fi

python smoke_test.py --config "${CONFIG}" --require-cuda --skip-data
