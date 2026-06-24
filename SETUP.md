# Setup: Environment, Dataset Download, and Exploration

This branch sets up the project foundation and **records** it: a reproducible
Conda environment, the full MCT-LTDiag download, and a dataset audit/exploration
report. Heavy steps run on the GPU server; only compact reports/logs are
committed back through Git.

Workflow: run on the server → commit `runs/<run_id>/reports` and logs → push →
pull locally → review (and let the coding agent read the reports).

## 1. Server environment

```bash
# Creates/updates the `liver-tumor-seg` Conda env and runs a smoke check
# (imports, CUDA, NIfTI I/O, a tiny tensor op). The CUDA check must run on a
# GPU node — login nodes usually expose no GPU.
bash scripts/server_phase1_smoke.sh
```

## 2. Dataset download (long-running, resumable)

```bash
export MCT_DATA_ROOT=/path/with/space/MCT_LTDiag   # ~270 GiB workspace needed

# Run inside tmux/screen or a batch job; it resumes if interrupted.
conda run -n liver-tumor-seg python download_dataset.py \
  --config configs/phase1_server_download.yaml
```

Downloads 517 case archives (~168 GiB) and extracts, per case, the four phase
NIfTIs (`nc/art/pvp/delay`) plus the tumor and liver masks. DICOM is skipped
(add `--extract-dicom` only if a later study needs it).

## 3. Dataset exploration / audit

```bash
conda run -n liver-tumor-seg python explore_dataset.py \
  --config configs/phase1_server_audit.yaml
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
