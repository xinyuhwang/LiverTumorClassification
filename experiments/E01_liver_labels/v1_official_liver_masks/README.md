# E01 · v1 — official liver masks

| | |
|---|---|
| Status | data QA done (2026-09-29); Stage 1 training next |
| Compared with | v0 (notebook labels) |
| Change | Stage 1 trains on the dataset's `liver_mask_pvp.nii.gz` (written as `liver_mask.nii.gz` by `data_prep/`) instead of `mask_pvp ≥ 1` |
| Code | data prep at commit `9c2c3f5` (AICR job 1119015); training commit: fill in from the job log |
| Run | `$HERALD_STORE/results/runs/ds2net/e01_v1_liver` |
| Date | 2026-09-29 |

## Hypothesis

With real liver labels, DS²Net Stage 1 should reach per-case test liver Dice ≥ 0.95. That's the range of published LiTS results (0.96) and UNet-Hybrid's reported 0.9704, which was trained on a separate liver mask.

## Steps

1. **Data preparation and QA** (`data_prep/prepare_mct_ltdiag.py --overwrite`, then `summarize_manifest.py`) on all 517 cases:
   - resample any phase or mask off the PVP grid onto it
   - check masks are binary
   - check the tumor lies inside the liver mask (flag < 90%)
   - check liver volume
   - check phase misregistration
2. **Train** DS²Net Stage 1 on the shared split, with the notebook's hyperparameters (`ds2net/config.py`, `LIVER`).
3. **Evaluate** on the 78 test cases: per case, full volume, 8-fold TTA, original grid. Save predicted liver masks for all cases (`--save_liver_masks`).

Commands are in [`run.sh`](run.sh).

## Results

### Step 1 — data QA (done)

Files: [`results/data_qa.json`](results/data_qa.json) (summary) and [`results/manifest.csv`](results/manifest.csv) (per case).

| Check | Result |
|---|---|
| Cases prepared | 517 / 517, all 7 files, 90 GB |
| Masks binary | 517 / 517 |
| Files resampled onto the PVP grid | 17 cases. 14 had phase headers off by up to 4.5% in-plane scale and 9 mm shift; 3 had a liver mask on a different slice grid. |
| Tumor < 90% inside the liver mask | 126 cases (125 match Yu Zhang's independent audit) |
| Phase misregistration (> 10% of PVP liver mask on air/fat) | NC 18, arterial 4, delayed 16, PVP 1 (231109b01) |
| Median liver-mask air fraction per phase | ≈ 0.1% in every phase |
| Liver volume | median 1,249 ml; 7 cases outside 700–3,500 ml |

### Steps 2–3 — Stage 1 training (pending)

| Metric (test, n = 78, per case) | v0 | v1 |
|---|---|---|
| Liver Dice | n/a (label was tumor) | |
| Liver IoU | n/a | |
| Liver precision / recall | n/a | |
| `Dice_vs_union` | n/a | |
| `tumor_covered` | n/a | |
| Best val Dice (slice-level) | n/a | |

## Observations

- **Phase geometry.** Before preparation, 14 cases could not be stacked voxel-for-voxel. Header-based resampling improved body overlap with the PVP from 0.87–0.95 to 0.92–0.98 in the worst cases (240122d70: 0.875 → 0.951). Both pipelines had stacked these phases misaligned.
- **Breathing motion** is not corrected; it is only measured. About 5–7% of cases have a phase with more than 10% of the liver mask on air or fat. This is a candidate for a later data experiment (deformable registration to PVP).
- **Excluded cases.** 240504b27 and 240504e30 are fixed by resampling their liver masks, and 240504e48 shows no problem; all three were re-included. 231109b01 stays excluded: its liver mask is wrong even on the PVP. The training set is now 360; val/test are unchanged.
- **Tumor outside the liver mask** is confirmed as common (126 cases > 10%). This is the case for v2 (liver ∪ tumor).

## Decision

—
