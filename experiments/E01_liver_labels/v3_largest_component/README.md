# E01 · v3 — keep the largest connected component

| | |
|---|---|
| Status | done (2026-09-30), **adopted** |
| Compared with | v2 (liver ∪ tumor label) |
| Change | At test time, keep only the largest 3-D connected component of the predicted liver. The model and its probabilities are v2's, unchanged; no retraining. |
| Code | [`rescore_posthoc.py`](rescore_posthoc.py) applied to v2's saved probabilities (`e01_v2_liver_union/liver_masks/`) on AICR (CPU job). The same step is now available in the pipeline as `--set keep_largest_component=True` (`common/metrics.largest_component`, `ds2net/train.py`). |
| Run | `$HERALD_STORE/results/runs/ds2net/e01_v2_lcc_posthoc` |
| Date | 2026-09-30 |

## Why

v1 and v2 both over-segment: precision 0.92 against recall 0.98. Before retraining with non-liver slices (the plan at the end of v2), we measured where the false-positive voxels are (`fp_location.py` on the saved predictions):

| | False positives as a share of reference volume | on liver slices | 1–3 slices outside the liver | further away | Cases with more than one predicted component |
|---|---|---|---|---|---|
| v1 | 0.074 | 40.9% | 8.3% | **50.9%** | 75 / 78 |
| v2 | 0.081 | 41.1% | 6.4% | **52.5%** | 75 / 78 |

About half of the false-positive volume lies in separate blobs far from the liver. The worst v2 case, 240229c27, has 352K false-positive voxels far from the liver in 10 components. The liver is a single organ, so keeping only the largest component is the cheapest fix to try first.

## Results

Paired comparison with v2 (`evaluate.py compare`, 2,000 bootstrap resamples; B − A = v3 − v2):

| metric | v2 | v3 | diff | 95% CI | better | worse |
|---|---|---|---|---|---|---|
| **`Dice_vs_union`** (= Dice vs this label) | 0.9494 | **0.9727** | **+0.0233** | +0.0156 to +0.0330 | 75 | 3 |
| **Precision** | 0.9231 | **0.9677** | **+0.0445** | +0.0316 to +0.0603 | 78 | 0 |
| Recall | 0.9802 | 0.9781 | −0.0021 | −0.0034 to −0.0010 | 0 | 78 |
| `tumor_covered` | 0.9476 | 0.9402 | −0.0074 | −0.0220 to −0.0000 | 0 | 18 |

v3 alone: Dice 0.9727 (95% CI 0.9694–0.9753), median 0.9760, lowest case 0.881. All p < 0.001 (bootstrap and Wilcoxon). Files: [`results/`](results/) (`per_case_test.csv`, `evaluate_summary.json`, `compare_v2_v3.json`).

| Worst v2 cases | v2 | v3 |
|---|---|---|
| 240229c27 (HH) | 0.693 | 0.954 |
| 231206d09 (ICC) | 0.773 | 0.958 |
| 240620a11 (BCLM) | 0.844 | 0.944 |

## Observations

- **Over-segmentation was mostly separate blobs.** Removing them raises precision in all 78 cases and Dice in 75. It costs a uniform, very small amount of recall (−0.002): voxels in small, genuinely disconnected pieces of the liver.
- **One real loss: 231025c23 (HH).** Its large hemangioma (94 ml; the official liver mask covered 1% of it) was predicted as its own component, and the filter discards it: `tumor_covered` 0.573 → 0.005, Dice 0.967 → 0.955. This is the failure mode to watch: tumors that bulge out of the liver and connect to it only thinly or not at all in the prediction. The other 17 `tumor_covered` drops are ≤ 0.004.
- **Retraining with non-liver slices is now much less urgent.** The post-processing removes most of what it targeted, at no training cost.

## Decision

- **Adopt:** Stage 1 = v2 label (`liver_includes_tumor=True`) + `keep_largest_component=True`. Test liver Dice 0.973 (0.969–0.975).
- **Possible refinement (not scheduled):** keep additional components above a size fraction of the largest, to retain large exophytic tumors like 231025c23. Evaluate it post hoc the same way before any retraining.
- **Deprioritised:** non-liver slices in training (the former v3 plan). Revisit only if a later stage needs cleaner probability maps rather than cleaner masks.
