# E01 · v2 — liver label = liver mask ∪ tumor

| | |
|---|---|
| Status | submitted (AICR job 1130505, 2026-09-30) |
| Compared with | v1 (official liver mask as shipped) |
| Change | `--set liver_includes_tumor=True`: the liver label also covers every tumor voxel (the LiTS convention). Nothing else changes: same split, hyperparameters, and test protocol. |
| Code | commit `e685053` |
| Run | `$HERALD_STORE/results/runs/ds2net/e01_v2_liver_union` |
| Date | 2026-09-30 |

## Hypothesis

v1 learned to leave out the tumors that the official masks leave out: per-case `tumor_covered` correlates 0.93 with the official mask's coverage, and HH `tumor_covered` is 0.67 against 0.94–0.97 for the other types. With tumors inside the label:

- **Expected:** `tumor_covered` rises to ≈ 0.95+ for every type, HH most. `Dice_vs_union`, the common reference for both versions, improves, mainly for HH.
- **Uncertain:** `Dice` against v2's own label isn't directly comparable with v1, because the reference differs. Judge on `Dice_vs_union` and `tumor_covered`.
- **Expected unchanged:** over-segmentation of non-liver regions (precision); this version doesn't address it.

## How to reproduce

See [`run.sh`](run.sh).

## Results

To fill in from:

```bash
python common/evaluate.py summary experiments/E01_liver_labels/v2_liver_union_tumor/results --by-type --md
python common/evaluate.py compare experiments/E01_liver_labels/v1_official_liver_masks/results \
       experiments/E01_liver_labels/v2_liver_union_tumor/results \
       --metrics Dice_vs_union tumor_covered Precision Recall --md
```

## Observations

—

## Decision

—
