# E03 · v0 — Stage 3 classification baselines on the shared split

| | |
|---|---|
| Status | planned (code ready, not submitted) |
| Compared with | the HERALD paper's Table 7 (old split; for orientation only) |
| Change | first run of both Stage 3 classifiers on the shared split, with per-patient val and test output |
| Code | commit `<hash>` |
| Run | paper Stage 3: `$HERALD_RESULTS/unet_hybrid/logs/stage3_paper_<backbone>/` and `stage3_paper_ensemble_<members>/`; DS²Net: `$HERALD_RESULTS/runs/ds2net/e03_v0_ds2net_cls/` |
| Date | 2026-10-03 |

## What runs

1. **Paper Stage 3** (`unet_hybrid/stage3_paper.py`, paper hyperparameters):
   - **Input:** PVP only, a 2.5D triplet + the ground-truth tumor mask as a 4th channel, crops with a 25% margin, ≤ 8 slices per patient.
   - **Backbones:** EfficientNet-B3, ViT-B/16, Swin-Tiny, Swin-Base, ResNet-50, and the UNet-Hybrid encoder (initialised from the E05 v2 Stage 2 checkpoint), one job each.
   - **Prediction:** the mean over slices.
   - **Ensembles:** the paper's EfficientNet-B3 + ViT-B/16 + UNet, and all six.
2. **DS²Net Stage 3** (`ds2net/train.py --stage 3`, notebook hyperparameters):
   - **Input:** all 4 phases (EfficientNet-B0 + PhaseNorm + LI-RADS phase attention + CBAM), plus enhancement-curve features from a ring around the tumor.
   - **Patches:** ≤ 16 ground-truth tumor ROI patches per patient.

## Hypothesis

Accuracy should land near the paper's (~0.6 for single backbones, ~0.7 for the ensemble), with wide CIs: 78 patients give about ±0.10 on accuracy. DS²Net's 4-phase classifier might do better than the PVP-only paper models, since tumor types differ mainly in how they enhance across phases.

## How to reproduce

See [`run.sh`](run.sh). Six paper-backbone jobs and one DS²Net job run in parallel (each roughly 1–2 h on 1 GPU), then a short ensemble job.

## Results

```bash
python common/evaluate.py summary <run>/per_case_val.csv --md
python common/evaluate.py compare <run A>/per_case_val.csv <run B>/per_case_val.csv --md
```

## Observations

## Decision
