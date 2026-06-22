#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${LIVER_TUMOR_CONDA_ENV:-liver-tumor-seg}"
CONFIG="${1:-configs/phase1_server_audit.yaml}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH" >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update -n "${ENV_NAME}" -f environment.yml --prune
else
  conda env create -n "${ENV_NAME}" -f environment.yml
fi

conda activate "${ENV_NAME}"
python smoke_test.py --config "${CONFIG}" --require-cuda --skip-data
