# E01 · v0 — notebook labels

| | |
|---|---|
| Status | done (documented from existing outputs, not rerun) |
| Change | baseline: labels as built by the DS²Net notebooks |
| Code | `ds2net/notebooks/liver_segmentation.ipynb` (Cell 7), `segmentation_classification.ipynb` (cell 0, Cell 7), commit `f77e286` |
| Date | 2026-09-29 |

## What the notebooks did

```python
seg   = nib.load("mask_pvp.nii.gz")
liver = (seg >= 1)                                            # both notebooks
tumor = (seg == 2) if 2 in np.unique(seg) else (seg == 1)     # segmentation notebook
```

## Evidence the "liver" label was the tumor

1. **The mask file is tumor-only.** Case `230906d12`, downloaded from Dataverse: `mask_pvp.nii.gz` has labels {0, 1} and 31,147 voxels (48.6 ml). The separate `liver_mask_pvp.nii.gz` in the same archive has 741,400 voxels (1,156.7 ml) and contains 100% of the tumor. So `seg ≥ 1` selects the tumor, and label 2 never exists, which makes the notebook's `liver` and `tumor` identical. Details: [`results/label_check_230906d12.json`](results/label_check_230906d12.json).
2. **The notebook's own outputs show identical channels.** The segmentation notebook's validation "liver" and "tumor" Dice match at every checkpoint (e.g. 0.6863 / 0.6853, 0.7314 / 0.7314), and so do the test metrics (Dice 0.6855 / 0.6854, IoU 0.5932 / 0.5932). See [`results/notebook_metrics.json`](results/notebook_metrics.json).

## Consequences

- `liver_segmentation.ipynb` trained a "liver" model on tumor masks. Its liver predictions are not liver.
- The poster's "DS2Net liver Dice 0.6609" is a tumor number.
- DS²Net's tumor results are unaffected by the label mix-up, because that channel was the tumor all along. Their slice-level vs per-case gap is a separate issue, handled in E02.
- The UNet-Hybrid pipeline read a separate `liver_mask.nii.gz`, so its liver Dice (0.9704) was not affected by this bug.

## Decision

Replace with the official liver masks → [v1](../v1_official_liver_masks/).
