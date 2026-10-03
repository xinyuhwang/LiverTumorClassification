# E02 · v6 — failure rates, per-size tests and per-lesion detection (R1)

| | |
|---|---|
| Status | code done and tested locally; re-scoring of nnU-Net predictions planned |
| Compared with | v5 (global Dice, size bins) |
| Change | R1 from [`ROADMAP.md`](../../docs/ROADMAP.md), sources [2] (per-size analysis with tests) and [6] (reporting failed cases) |
| Code | commit `<hash>` |
| Date | 2026-10-03 |

## What changed

1. **`common/evaluate.py`**
   - `summary` adds **`fail_rate`**, the share of cases with Dice < 0.5, with a bootstrap CI, for every group (overall, per type, per size bin).
   - `compare` adds a paired `fail_rate` row (better = cases that stopped failing). With `--size-bins`, it repeats the paired Dice comparison **within each size group**, with CI and Wilcoxon test. Every row now has a `group` column.
2. **`common/metrics.py`: `lesion_metrics(pred, gt, voxel_ml)`**
   - **Lesions:** 26-connected components of the GT and predicted tumor masks.
   - **Detected:** a GT lesion counts as detected if any predicted voxel overlaps it.
   - **False positive:** a predicted component that touches no GT lesion.
   - **Per case:** `n_gt_lesions`, `n_detected`, `n_pred_lesions`, `n_fp_lesions` and `lesion_recall`.
   - **Per GT lesion:** volume, detected, overlap fraction and lesion Dice.
3. **Where it's used**
   - `nnunet/evaluate_predictions.py` adds the per-case lesion counts to the tumor CSV and writes `tumor/per_lesion_<split>.csv`.
   - `ds2net/train.py` adds `gt_ml`, `pred_ml` and the lesion counts to every Stage 2 tumor row, so future runs no longer need `gt_ml` copied from E06.
4. **Tests:** `tests/test_metrics.py` (lesion detection, false positives, empty volumes) and `tests/test_evaluate.py` (fail rate, size-group comparison). 18 tests pass.

**Why:** a per-case mean hides *how* small tumors fail. Lesion-level recall separates **missed lesions** (detection) from **poor outlines** (segmentation), and the two call for different fixes.

## Re-scoring existing runs

nnU-Net saves its predicted label maps, so E06 v1 and v3 can be re-scored on CPU (see [`run.sh`](run.sh), ~15 min). DS²Net runs don't save masks; they get lesion counts from their next run, or via `--eval_only` (~1 h GPU each).

## Results

## Observations
