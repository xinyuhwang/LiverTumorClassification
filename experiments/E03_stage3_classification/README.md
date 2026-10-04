# E03 · Stage 3 tumor-type classification: baselines, pooling and aggregation

## Question

How well do HERALD's tumor-type classifiers work on the shared split, and does replacing "mean over slices" with mask-weighted pooling (O1, OrganLens) and attention-based MIL (G1, GigaPath-Flash) improve them?

## Setup

- **Task:** 5 classes (BCLM, CRLM, HCC, HH, ICC), one label per patient.
- **Split:** the shared split (360 train / 78 val / 78 test).
- **Metrics:** patient-level accuracy, macro-F1 and macro one-vs-rest AUC with bootstrap 95% CIs, and McNemar tests between versions (`common/evaluate.py`, E02). Decisions on val, report on test.
- **Masks:** v0 uses ground-truth tumor masks (oracle), as the paper did. Predicted masks come in a later version.

## Reference

The HERALD paper (Table 7, old split, PVP only, oracle masks), 78 test patients:
- Single backbones: ResNet-50 0.615, EfficientNet-B3 0.628, ViT-B/16 0.577, Swin-Tiny 0.564, Swin-Base 0.539.
- Late-fusion ensemble (EfficientNet-B3 + ViT-B/16 + UNet encoder): **0.69**.

Not comparable with this experiment: different split, pre-fix code, no per-case output.

## Versions

| Version | Change | Status |
|---|---|---|
| [v0](v0_baselines/) | Baselines on the shared split: the paper's Stage 3 (6 backbones + its 3-model ensemble) and DS²Net's 4-phase Stage 3 | Done: six-model ensemble val 0.667 / AUC 0.894, test 0.692 (paper: 0.69); ViT best single (val 0.705); DS²Net 4-phase 0.603 / test 0.487. All differences n.s. |
| [v1](v1_mask_pooling/) | Mask-weighted pooling + tumor-area slice weighting (O1), all six backbones + ensemble | Done: ensemble val 0.679 (+0.013, n.s.), test 0.667; 4/6 backbones up on val. Not adopted |
| [v2](v2_abmil/) | ABMIL per-case aggregation (G1), all six backbones + ensemble; built on v0 | Done: ensemble val 0.641 (−0.026, n.s.), test 0.756. Not adopted (val decides) |
| [v3](v3_ds2net_fixed_norm/) | DS²Net 4-phase Stage 3 without PhaseNorm (`phase_norm=fixed`) | Done: val 0.603 (±0); ICC→HCC errors 8 → 4 but HCC/BCLM worse. Not adopted |
| [v4](v4_cv_evaluation/) | Evaluation change: 5-fold CV over 438 train + val patients (out-of-fold), re-scoring v0, v1, v2 | Planned: code ready |

## Conclusion (v0–v3, 2026-10-03)

| Version | Val accuracy (six-model ensemble) | Test accuracy | Kept |
|---|---|---|---|
| v0 paper Stage 3 | 0.667 (AUC 0.894) | 0.692 | baseline |
| v1 mask pooling (O1) | 0.679 (+0.013, n.s.) | 0.667 | option |
| v2 ABMIL (G1) | 0.641 (−0.026, n.s.) | 0.756 | option |
| v3 DS²Net without PhaseNorm | 0.603 (±0) | 0.474 | no |

- **None of the three changes measurably improves classification.** Every difference is within ±0.03 on val and not significant.
- **The evaluation can't resolve smaller effects:** on 78 patients one decision is 0.013 accuracy, the 95% CI is about ±0.10, and v2 moved −0.03 on val but +0.06 on test.
- **Ways to get a more decisive signal:**
  1. Evaluate with cross-validation over all 438 train + val patients (5 folds), so each version is scored on 438 patients instead of 78.
  2. Test changes expected to matter more: multi-phase input to the paper backbones (v3 showed phases help ICC), and predicted masks (the realistic setting).
- HH is always 100%; CRLM vs BCLM (both metastases) remains the main confusion.
