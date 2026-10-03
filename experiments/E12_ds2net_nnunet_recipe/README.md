# E12 · DS²Net Stage 2 with nnU-Net's recipe

## Question

nnU-Net (E06 v1) beats DS²Net on tumors: 0.796 vs 0.726 per case on test, and +0.026 on validation. Which parts of nnU-Net's recipe close that gap when moved into DS²Net Stage 2, while keeping DS²Net's 2.5D model?

## Setup

- **Baseline:** E05 v1, DS²Net Stage 2 inside the liver box, cascade from E01's liver masks.
  - Validation: 0.769 per case, 0.893 global.
  - Test: 0.726 per case.
- **Rules:**
  - One change per version, each compared with the previous version on validation (cascade mode, paired).
  - A change is kept only if it helps on validation.
  - Test is reported at the end.
- **Out of scope:**
  - nnU-Net's 3D network itself; that would make it a different model.
  - Stage 1 liver, which already matches nnU-Net at 0.97.

## What nnU-Net does differently (from its plan for this dataset)

| | nnU-Net 3d_fullres (E06) | DS²Net Stage 2 (E05 v1) |
|---|---|---|
| Voxel spacing | 5 mm × 0.73 × 0.73 mm: native, since all 517 cases have 5 mm slices | resampled to 1 mm in all three axes: 5× interpolation in z |
| Context across slices | 32-slice 3D patch (160 mm) | 3 slices at 1 mm = 2 mm, mostly interpolated between the same two real slices |
| Intensity normalisation | per phase, fixed dataset statistics (clip to 0.5–99.5 percentile, z-score) | HU window to [0, 1], then PhaseNorm: per-slice instance norm per channel, which removes absolute enhancement |
| Training | 1,000 epochs × 250 steps, no early stopping, poly LR | 50 epochs (best at 39), OneCycle, early stopping |
| Loss | Dice + cross-entropy, deep supervision | uncertainty-weighted wIoU + wBCE + boundary, small-tumor branch |
| Ensemble | 5 folds (E06 v2) | single model |

## Versions

| Version | Change (vs previous) | Status |
|---|---|---|
| [v1](v1_native_z/) | slices at native 5 mm instead of 1 mm (`z_spacing=5.0`); same samples per epoch | Done: val 0.767 vs 0.769 (−0.002, n.s.); no effect, kept as base (cheaper) |
| [v2](v2_fixed_norm/) | no PhaseNorm: one fixed, learnable scaling for all slices (`phase_norm=fixed`) | Done: val 0.780 vs 0.767 (+0.013, n.s.); recall +0.035 (sig.); < 10 ml +0.041; test 0.735. Kept as base |
| [v3](v3_long_training/) | 100 epochs instead of 50, no early stopping (on top of v2) | Done: val 0.771 vs 0.780 (−0.009, n.s.); plateau from epoch ~40. Not adopted |
| [v4](v4_dice_ce/) | Dice + cross-entropy loss, nnU-Net style (`loss=dice_ce`; on top of v2, run in parallel with v3) | Done: val 0.751 vs 0.780 (**−0.030, significant**); recall −0.082, small tumors −0.083. Not adopted |
| v5 | 5-fold ensemble on E06 v2's inner folds | Planned |

## Conclusion (after v1–v4, 2026-10-03)

| Change from nnU-Net's recipe | Version | Val Dice vs previous | Kept |
|---|---|---|---|
| Native 5 mm slices | v1 | −0.002 (n.s.) | yes (cheaper) |
| No PhaseNorm | v2 | +0.013 (n.s.); recall +0.035, < 10 ml +0.041 | **yes** |
| 100 epochs | v3 | −0.009 (n.s.) | no |
| Dice + CE loss | v4 | −0.030 (significant) | no |

- Best DS²Net: **v2**, val 0.780 / test 0.735 per case, global 0.900 / 0.871. Notebook baseline E05 v1: val 0.769 / test 0.726.
- Only removing PhaseNorm helped. Training length and nnU-Net's loss don't transfer to 2.5D slice training; native spacing is neutral.
- nnU-Net (E06 v3) is still ahead by about 0.03 per case on val and 0.05 on test, which points to 3D context as the main difference.
- v5 (5-fold DS²Net ensemble) is unlikely to close that: the nnU-Net ensemble added only +0.012.
