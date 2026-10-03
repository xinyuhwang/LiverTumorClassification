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
| [v1](v1_mask_pooling/) | Mask-weighted pooling + tumor-area slice weighting (O1), all six backbones + ensemble | Planned: code ready |
| [v2](v2_abmil/) | ABMIL per-case aggregation (G1), all six backbones + ensemble; built on v0 | Planned: code ready |
| [v3](v3_ds2net_fixed_norm/) | DS²Net 4-phase Stage 3 without PhaseNorm (`phase_norm=fixed`) | Planned: code ready |

## Conclusion

—
