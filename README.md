# Liver Tumor Segmentation and Classification

Reproducible research code for multi-phase CT liver tumor segmentation and
downstream tumor subtype classification using the MCT-LTDiag dataset.

The project is being migrated from notebooks to Python scripts. Local macOS is
used for development and result analysis; full data processing and GPU training
run in a Conda environment on a school server.

## Dataset

MCT-LTDiag contains 517 cases with four registered CT phases:

- non-contrast (`nc.nii.gz`);
- arterial (`art.nii.gz`);
- portal venous (`pvp.nii.gz`);
- delayed (`delay.nii.gz`).

Diagnostic classes are HCC, ICC, CRLM, BCLM, and HH. The expert-validated
binary tumor annotation is `mask_pvp.nii.gz`. The separate
`liver_mask_pvp.nii.gz` was generated automatically and is treated as a
pseudo-label rather than an equivalent expert ground truth.

Data descriptor: <https://doi.org/10.1038/s41597-025-06343-4>

Dataverse: <https://doi.org/10.7910/DVN/S3RW15>

## Phase 1 Tools

Lightweight local setup (no PyTorch required):

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e '.[dev]'

.venv/bin/python smoke_test.py \
  --config configs/phase1_local_sample.yaml --skip-data
```

When the ignored sample dataset exists at `data/mct-ltdiag_sample`:

```bash
.venv/bin/python smoke_test.py --config configs/phase1_local_sample.yaml
.venv/bin/python explore_dataset.py --config configs/phase1_local_sample.yaml
```

The server uses `environment.yml` so CUDA/PyTorch can be validated there.
Server commands and the Git result-return protocol are documented in
[`docs/REMOTE_WORKFLOW.md`](docs/REMOTE_WORKFLOW.md).

## Experiment Records

Every command creates a unique `runs/<run_id>` directory containing:

- resolved configuration and command;
- Git commit, dirty state, and diff;
- Python, Conda, PyTorch, CUDA, and GPU information;
- source snapshots;
- terminal/file logs and final status;
- compact reports and artifact manifests.

Datasets, model weights, checkpoints, caches, and full prediction volumes are
ignored by Git. Git LFS is intentionally not used at this stage.

## Current Research Order

1. Reproducible infrastructure and dataset validation.
2. PVP-only and four-phase nnU-Net baselines.
3. Corrected scripted DS2Net and controlled phase ablations.
4. Multi-token attention Swin segmentation.
5. End-to-end predicted-mask classification.
