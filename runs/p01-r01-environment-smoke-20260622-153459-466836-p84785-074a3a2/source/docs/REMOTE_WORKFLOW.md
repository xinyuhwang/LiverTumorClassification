# Remote GPU Experiment Workflow

The Mac is the development and analysis machine. The school server owns the
Conda environment, full dataset, GPU execution, caches, checkpoints, and full
prediction volumes.

## One-Time Server Setup

```bash
git clone <repository-url>
cd LiverTumorClassification
git switch codex/script-experiment-workflow

export MCT_DATA_ROOT=/path/with/enough/space/MCT_LTDiag
export DATAVERSE_API_TOKEN='set-this-in-the-shell-or-secret-store'

bash scripts/server_phase1_smoke.sh configs/phase1_server_audit.yaml
```

Do not place tokens in YAML, shell scripts, Git config, or run logs.

## Phase 1 Commands

Inspect Dataverse metadata without downloading files:

```bash
conda run -n liver-tumor-seg \
  python download_dataset.py \
  --config configs/phase1_server_download.yaml \
  --metadata-only
```

Download and safely extract the full dataset:

```bash
conda run -n liver-tumor-seg \
  python download_dataset.py \
  --config configs/phase1_server_download.yaml
```

Run the full audit after download:

```bash
conda run -n liver-tumor-seg \
  python explore_dataset.py \
  --config configs/phase1_server_audit.yaml
```

Every command prints `RUN_ID` and `RUN_DIR`. The returned `runs/<run_id>`
directory contains compact logs and reports suitable for Git.

## Per-Run Branch Protocol

Local development:

```bash
git add <source-and-config-files>
git commit -m "Implement phase 1 round N"
git push origin codex/script-experiment-workflow
```

Server execution from the exact source commit:

```bash
git fetch origin
git switch codex/script-experiment-workflow
git pull --ff-only

RUN_BRANCH="run/phase1-<short-description>"
git switch -c "${RUN_BRANCH}"

# Execute one command and note the printed RUN_ID.

git add runs/<run-id>
git commit -m "Record <run-id> results"
git push -u origin "${RUN_BRANCH}"
```

Local result integration:

```bash
git fetch origin
git switch codex/script-experiment-workflow
git cherry-pick <result-commit-sha>
python analyze_run.py runs/<run-id>
```

After integration, the temporary run branch can be deleted locally and
remotely. This avoids conflicts if local code changes while a GPU job runs.

## Artifact Policy

Track in normal Git:

- logs and status JSON;
- resolved configs and source snapshots;
- CSV/JSON metrics and dataset audit reports;
- compact PNG plots and qualitative examples;
- artifact manifests.

Keep server-side only:

- raw and derived datasets;
- preprocessing caches;
- checkpoints and model weights;
- full NIfTI predictions;
- TensorBoard/W&B caches.

The `.gitignore` blocks common model, medical-volume, archive, and cache file
extensions. Large server-only files should still be registered in the run's
artifact manifest with path, size, and SHA-256.
