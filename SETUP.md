# Setup: Environment, Dataset Download, and Exploration

This branch sets up the project foundation and **records** it: a reproducible
Conda environment, the full MCT-LTDiag download, and a dataset audit/exploration
report. Only compact reports/logs are committed back through Git.

Workflow: run on the server → commit `runs/<run_id>/reports` and logs → push →
pull locally → review (and let the coding agent read the reports).

## 0. Everything lives on the professor's shared disk

Personal/home space on the server is small, so keep the repo, the Conda env, the
dataset, and all caches under the shared disk. Set these once per shell session
(adjust the base path to the real shared location):

```bash
export PROJECT_SPACE=/shared/professor/liver-tumor       # big shared disk
export LIVER_TUMOR_CONDA_PREFIX="$PROJECT_SPACE/envs/liver-tumor-seg"
export MCT_DATA_ROOT="$PROJECT_SPACE/MCT_LTDiag"          # ~270 GiB workspace
export CONDA_PKGS_DIRS="$PROJECT_SPACE/conda-pkgs"        # keep pkg cache off home
export PIP_CACHE_DIR="$PROJECT_SPACE/pip-cache"
export TMPDIR="$PROJECT_SPACE/tmp"; mkdir -p "$TMPDIR"
```

Clone the repo into `$PROJECT_SPACE` and `cd` into it before running anything.
The GPU is attached directly to this environment — there is no separate GPU node
or job scheduler.

## 1. Conda environment

```bash
# Creates the env at $LIVER_TUMOR_CONDA_PREFIX (shared space) and runs a smoke
# check: imports, CUDA, NIfTI I/O, and a tiny tensor op.
bash scripts/server_phase1_smoke.sh
conda activate "$LIVER_TUMOR_CONDA_PREFIX"
```

## 2. Dataset download (long-running, resumable)

```bash
# Run inside tmux/screen; it resumes if interrupted.
python download_dataset.py --config configs/phase1_server_download.yaml
```

Downloads 517 case archives (~168 GiB) into `$MCT_DATA_ROOT` and extracts, per
case, the four phase NIfTIs (`nc/art/pvp/delay`) plus the tumor and liver masks.
DICOM is skipped (add `--extract-dicom` only if a later study needs it).

## 3. Dataset exploration / audit

```bash
python explore_dataset.py --config configs/phase1_server_audit.yaml
```

Writes `runs/<run_id>/` containing:

- `reports/dataset_audit/` — `cases.csv`, `summary.json`, `issues.csv`,
  `report.md`, and `qc/*.png` (four-phase overlays);
- `logs/run.log`, `status.json`, `environment.json`, `git_info.json`.

## 4. Record the results (commit back)

```bash
git add runs/<run_id>          # per-run source/ snapshots are gitignored
git commit -m "Record dataset audit <run_id>"
git push
```

Pull locally and review:

```bash
python analyze_run.py runs/<run_id>
```

## Local (Mac) quick check — no GPU or dataset needed

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python smoke_test.py --config configs/phase1_local_sample.yaml --skip-data
```

## What is and isn't tracked in Git

- Tracked: code, configs, and compact run records (logs, metrics, reports, QC
  PNGs, manifests).
- Ignored: datasets, archives, caches, checkpoints/weights, full NIfTI
  predictions, and per-run source snapshots (see `.gitignore`).
