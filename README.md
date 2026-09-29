# LiverTumorClassification

HERALD (Hepatic Evaluation and Reporting with AI-Driven Diagnostics) covers liver segmentation, tumor segmentation, and tumor subtype classification (BCLM, CRLM, HCC, HH, ICC). It uses the MCT-LTDiag multi-phase CT dataset ([doi:10.7910/DVN/S3RW15](https://doi.org/10.7910/DVN/S3RW15), CC0).

## Layout

```text
common/                    shared by all pipelines
  splits.py                load_splits(): the one train/val/test split
  mct_ltdiag_split.csv     517 cases, stratified by type and age bin
data_prep/
  download_mct_ltdiag.py   download from Harvard Dataverse (resumable, MD5-checked)
  prepare_mct_ltdiag.py    archives → per-case NIfTI layout + QA manifest
  summarize_manifest.py    QA summary of the manifest
ds2net/                    DS²Net pipeline (Swin-Tiny + DEM/SEM)
  config.py datasets.py models.py train.py
  PORTING.md               notebooks → scripts: what changed and why
  notebooks/               original Colab notebooks (reference)
unet_hybrid/               UNet-Hybrid pipeline (UNetTransformer)
  datasets.py models.py train.py
  stage3_paper.py          Stage 3 as described in the paper
  STAGE3_DESIGN.md         the two Stage 3 implementations
  results/                 original training logs
cluster/                   AICR setup: env, job scripts, submit / pull helpers
docs/
  AICR.md                  connecting to and running on AICR
  ROADMAP.md               the 11 proposals, prioritised
experiments/               one folder per experiment and version (see its README)
```

## Data layout

`data_prep/prepare_mct_ltdiag.py` writes one folder per case, and both pipelines read this layout:

```text
<data_dir>/<case>/phase_0.nii.gz   non-contrast
                  phase_1.nii.gz   arterial
                  pvp.nii.gz       portal venous (phase_2.nii.gz → pvp.nii.gz)
                  phase_3.nii.gz   delayed
                  liver_mask.nii.gz
                  tumor_mask.nii.gz
```

## Quick start

On AICR, follow [docs/AICR.md](docs/AICR.md) once, then:

```bash
bash cluster/submit.sh ds2net --stage 1 --run_name liver_baseline
bash cluster/submit.sh unet_hybrid --stage 1
bash cluster/submit.sh stage3_paper --backbone efficientnet_b3
```

Locally, with data prepared under `data/mct_ltdiag`:

```bash
cd ds2net && python train.py --stage 1 --smoke_test --no_pretrain
cd unet_hybrid && python train.py --stage 1 --smoke_test --data_dir ../data/mct_ltdiag
```

Research plan: [docs/ROADMAP.md](docs/ROADMAP.md). Results so far: [experiments/](experiments/).
