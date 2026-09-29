# E01 · v1 — official liver masks

| | |
|---|---|
| Status | ready to run on AICR |
| Compared with | v0 (notebook labels) |
| Change | Stage 1 trains on the dataset's `liver_mask_pvp.nii.gz` (written as `liver_mask.nii.gz` by `data_prep/`) instead of `mask_pvp ≥ 1` |
| Code | commit: fill in from the job log; `ds2net/train.py --stage 1`, default config |
| Run | `$HERALD_STORE/results/runs/ds2net/e01_v1_liver` |
| Date | — |

## Hypothesis

With real liver labels, DS²Net Stage 1 should reach per-case test liver Dice ≥ 0.95. That's the range of published LiTS results (0.96) and UNet-Hybrid's reported 0.9704, which was trained on a separate liver mask. Every prepared case should pass QA, with the tumor fully inside the liver mask.

## Steps

1. **Data QA** (`data_prep/prepare_mct_ltdiag.py`, then `data_prep/summarize_manifest.py`) on all 517 cases:
   - masks binary
   - all files on the PVP grid
   - tumor inside the liver mask (flag < 90%)
   - liver volume within 700–3,500 ml (review outliers)
2. **Train** DS²Net Stage 1 on the shared split, with the notebook's hyperparameters (`ds2net/config.py`, `LIVER`).
3. **Evaluate** on the 78 test cases: per case, full volume, 8-fold TTA, original grid. Save predicted liver masks for all cases (`--save_liver_masks`) for later stages.

Commands are in [`run.sh`](run.sh).

Checked locally on one real case (`230906d12`): QA passes, 100% of the tumor is inside the liver mask, and the maximum origin offset between phases is 3.9 mm (less than one slice). The full pipeline was also smoke-tested on synthetic data.

## Results

To fill in after the AICR run. Copy the files with `cluster/pull_results.sh`.

| Metric (test, n = 78, per case) | v0 | v1 |
|---|---|---|
| Liver Dice | n/a (label was tumor) | |
| Liver IoU | n/a | |
| Liver precision / recall | n/a | |
| Best val Dice (slice-level) | n/a | |
| QA: flagged cases / 517 | — | |

## Observations

—

## Decision

—
