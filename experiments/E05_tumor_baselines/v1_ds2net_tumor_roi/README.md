# E05 · v1 — DS²Net Stage 2 on the liver region

| | |
|---|---|
| Status | submitted (AICR job 1139799, 2026-09-30); smoke test job 1139415 passed |
| Compared with | first tumor baseline under the new pipeline (no previous version) |
| Change | DS²Net Stage 2 with `roi=liver` (square box around liver ∪ tumor, 10 mm margin, resized to 224), trained on all liver slices; notebook hyperparameters otherwise (`ds2net/config.py`, `TUMOR`) |
| Code | commit `d88c4d3`; `--set roi=liver slices=liver liver_includes_tumor=True roi_liver_run=e01_v2_liver_union` |
| Run | `$HERALD_STORE/results/runs/ds2net/e05_v1_ds2net_tumor_roi` |
| Date | 2026-09-30 |

## Hypothesis

Restricting the search to the liver removes false positives elsewhere in the body, the dominant error of the notebook version (per-case precision 0.31, recall 0.73). Per-case test Dice should therefore end well above the notebook's 0.375, though that number isn't strictly comparable: different split and bugs.
- **Cascade** should be close to **oracle**, since E01's liver Dice is 0.973.
- The largest gap is expected where the predicted liver leaves out tumor (exophytic hemangiomas, e.g. 231025c23).

## How to reproduce

See [`run.sh`](run.sh).

## Results

To fill in:

```bash
python common/evaluate.py summary experiments/E05_tumor_baselines/v1_ds2net_tumor_roi/results --by-type --md
python common/evaluate.py compare experiments/E05_tumor_baselines/v1_ds2net_tumor_roi/results/per_case_val_oracle.csv \
       experiments/E05_tumor_baselines/v1_ds2net_tumor_roi/results/per_case_val.csv --md   # cascade cost
```

## Observations

—

## Decision

—
