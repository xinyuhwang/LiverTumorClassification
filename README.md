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
nnunet/                    nnU-Net v2 baseline (E06)
  prepare_dataset.py       prepared data → nnU-Net raw dataset + HERALD split as fold 0
  evaluate_predictions.py  per-case tumor/liver scores with the shared metric code
common/ also holds         metrics.py (per-case metrics), evaluate.py (bootstrap CIs,
                           paired tests, global Dice, --size-bins)
tests/                     pytest tests for common/ (run: pytest tests/)
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

Evaluate any run (per-case CSV) with bootstrap CIs, or compare two runs case by case:

```bash
python common/evaluate.py summary <run dir or per_case_*.csv> --by-type --size-bins 10 50 200 --md
python common/evaluate.py compare <run A> <run B> --md
```

Research plan: [docs/ROADMAP.md](docs/ROADMAP.md). Results so far: [experiments/](experiments/). Current best (2026-10-02): nnU-Net 3D (E06 v1), tumor Dice 0.796 per case / 0.895 global on validation, liver 0.969.
## MCT-LTDiag — Multi-phase CT Liver-Tumor Diagnosis

A two-stage deep learning pipeline for liver segmentation and tumor detection in multi-phase abdominal CT scans, implemented in PyTorch and designed to run in Google Colab.


## Overview

MCT-LTDiag segments the liver from multi-phase contrast-enhanced CT volumes (Stage 1), then uses those liver masks as a spatial prior for tumor detection and classification (Stage 2). The current repository contains Stage 1 (v5).

The model uses a DS²Net architecture — combining Detail Enhancement Modules (DEM) and Semantic Enhancement Modules (SEM) in a U-Net-style encoder-decoder — augmented with a Multi-Head Self-Attention (MHA) bottleneck at the deepest feature scale.

## Pipeline Overview
 
```
Input CT (multi-phase)
        │
        ▼
1. Preprocessing
   └─ Data cleaning, HU windowing, resampling, data leakage check
        │
        ▼
2. Segmentation
   ├─ Stage 1 — Liver segmentation   (DS2Net | UNet-Hybrid)
   │             MHA / Cross-MTA attention
   └─ Stage 2 — Tumor segmentation  within liver ROI
        │
        ▼
3. Classification
   └─ Tumor subtype (EfficientNet | UNet | Swin-Tiny | Swin-Base | ViT)
        │
        ▼
Output: Liver mask + Tumor mask + Subtype label
        (BCLM | CRLM | HCC | HH | ICC)
```
 
---
 
## Dataset — MCT-LTDiag
 
517 contrast-enhanced multi-phase CT cases spanning five hepatic tumor subtypes:
 
| Subtype | Full name | N |
|---|---|---|
| BCLM | Breast cancer liver metastasis | 115 |
| CRLM | Colorectal liver metastasis | 103 |
| HCC | Hepatocellular carcinoma | 103 |
| HH | Hepatic hemangioma | 96 |
| ICC | Intrahepatic cholangiocarcinoma | 100 |
 
Each case includes four CT phases: **non-contrast (NC), arterial (art), portal-venous (pvp), and delayed**. Expert-annotated liver and tumor masks are provided.
 
> Wu et al., *MCT-LTDiag: Multi-phase CT Dataset for Automated Differential Diagnosis of Liver Tumors* (2025).
 
---
 
## Models
 
### Segmentation
 
| Model | Description |
|---|---|
| **DS2Net** | Detail-Semantic Dual-supervision Network. Swin-Tiny backbone (20-channel pseudo-3D input: 5 context slices × 4 phases), DS² deep supervision (5 heads), MHA self-attention bottleneck at d4 (14×14, 8 heads). |
| **UNet-Hybrid** | UNet-style encoder-decoder with transformer attention (Cross-MTA). Encoder initialized from Stage 1 weights for Stage 2 tumor segmentation. |
 
