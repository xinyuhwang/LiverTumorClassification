# E03 · v0 — Stage 3 classification baselines on the shared split

| | |
|---|---|
| Status | done: smoke tests 1176335/1176336; backbones 1176342–1176347 (swin_tiny rerun 1176758 after a crop-cache race), DS²Net 1176354, ensembles 1176794/1176795; 1–3 min each; 2026-10-03 |
| Compared with | the HERALD paper's Table 7 (old split; for orientation only) |
| Change | first run of both Stage 3 classifiers on the shared split, with per-patient val and test output |
| Code | commit `777057f` (crop-cache write made atomic afterwards; see Observations) |
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

Patient-level, 78 val / 78 test patients, ground-truth tumor masks; 95% bootstrap CIs.

| Model | Input | Val accuracy | Val macro-F1 | Val macro-AUC | Test accuracy | Test macro-AUC |
|---|---|---|---|---|---|---|
| EfficientNet-B3 | PVP | 0.590 (0.47–0.69) | 0.599 | 0.855 | 0.615 | 0.884 |
| **ViT-B/16** | PVP | **0.705** (0.60–0.81) | **0.697** | 0.888 | 0.679 | 0.896 |
| Swin-Tiny | PVP | 0.641 (0.54–0.74) | 0.646 | 0.850 | 0.679 | 0.889 |
| Swin-Base | PVP | 0.577 (0.46–0.68) | 0.572 | 0.863 | 0.654 | 0.915 |
| ResNet-50 | PVP | 0.641 (0.53–0.74) | 0.640 | 0.875 | 0.667 | 0.913 |
| UNet encoder | PVP | 0.641 (0.53–0.74) | 0.634 | 0.840 | 0.526 | 0.869 |
| Ensemble: EffNet + ViT + UNet (paper) | PVP | 0.667 (0.56–0.77) | 0.668 | 0.889 | 0.654 | 0.906 |
| **Ensemble: all six** | PVP | 0.667 (0.56–0.77) | 0.673 | **0.894** (0.85–0.94) | **0.692** | **0.920** |
| DS²Net Stage 3 (EfficientNet-B0, PhaseNorm) | 4 phases | 0.603 (0.49–0.71) | 0.595 | 0.839 | 0.487 | 0.761 |

Paired on val (McNemar for accuracy): ViT vs six-model ensemble −0.038 (−0.13 to +0.04), p 0.55; DS²Net vs six-model ensemble +0.064 (−0.05 to +0.18), p 0.38.

### Per-class recall on val

| | BCLM | CRLM | HCC | HH | ICC |
|---|---|---|---|---|---|
| Six-model ensemble | 0.61 | 0.47 | 0.67 | 1.00 | 0.60 |
| ViT-B/16 | 0.78 | 0.33 | 0.87 | 1.00 | 0.53 |
| DS²Net Stage 3 | 0.50 | 0.47 | 0.67 | 1.00 | 0.40 |

Main confusions (ensemble, val): CRLM → BCLM 7 of 15; BCLM → CRLM 4 of 18. DS²Net: ICC → HCC 8 of 15.

Files: [`results/`](results/): `paper/stage3_paper_<model>/per_case_{val,test}.csv` (+ `_probs.json`), `ds2net/`.

## Observations

- **The paper's numbers reproduce on the shared split.** The six-model ensemble reaches 0.692 test accuracy, the paper reported 0.69. Single backbones: 0.53–0.68 test (paper 0.54–0.63).
- **No model is significantly better than another.** CIs are ±0.10 on 78 patients, and the paired differences on val are all n.s. ViT-B/16 is the best single model on val (0.705); the ensembles are steadier across val and test.
- **Ranking is much better than the final call:** macro-AUC 0.84–0.92, while accuracy is 0.6–0.7.
- **Hemangioma is solved (15/15 everywhere). The hard pair is the two metastasis types:** colorectal and breast metastases look alike, and most CRLM errors are called BCLM. Telling them apart from imaging alone is hard even for radiologists; the primary cancer is clinical history.
- **DS²Net's 4-phase classifier is not better than PVP-only models** (val 0.603, test 0.487), although phases should matter most for tumor type. It uses PhaseNorm, the same per-image normalisation that held DS²Net Stage 2 back (E12 v2): it removes the absolute enhancement differences between phases, and its LI-RADS phase gate can't respond to the image. Its main error (ICC called HCC) is exactly the enhancement-pattern distinction.
- **Infrastructure:** the six backbone jobs built the shared crop cache at the same time; Swin-Tiny read a half-written file and failed. Rerun succeeded; `load_crops` now writes to a temporary file and renames it.

## Decision (made on validation)

- **E03 baseline = the six-model ensemble** (val accuracy 0.667, macro-AUC 0.894; test 0.692 / 0.920). A judgement call: ViT-B/16 alone is higher on val accuracy (n.s.), but the ensemble has the best val AUC and is the more stable reference.
- **Next versions:**
  - **v1:** mask-weighted pooling + tumor-area slice weighting (O1).
  - **v2:** ABMIL (G1).
  - **Also worth a version:** DS²Net Stage 3 with `phase_norm=fixed`. The classifier is where phase enhancement should matter most, and E12 showed PhaseNorm hides it.
