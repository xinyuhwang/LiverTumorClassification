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
| [v2](v2_fixed_norm/) | no PhaseNorm: one fixed, learnable scaling for all slices (`phase_norm=fixed`) | Planned: code ready |
| v3 | longer training, no early stopping | Planned |
| v4 | Dice + cross-entropy loss | Planned |
| v5 | 5-fold ensemble on E06 v2's inner folds | Planned |

## Conclusion

—
