# E01 · v4 — keep liver components near the largest one

| | |
|---|---|
| Status | done (2026-10-01): **not adopted**, v3 stays |
| Compared with | v3 (largest component only) |
| Change | Post-processing only, on v2's saved probabilities (no retraining): besides the largest component, also keep components by size (≥ 5 / 10 / 20% of the largest) or by distance (within 5 / 10 / 20 mm of it) |
| Code | [`rescore_components.py`](rescore_components.py) (AICR CPU srun, 16 cores, ~25 min; output `$HERALD_RESULTS/runs/ds2net/e01_v4_components/<rule>/per_case_{val,train,test}.csv`) |
| Date | 2026-10-01 |

## Why

v3 dropped one bulging (exophytic) hemangioma that the model predicted as a separate component (231025c23). In E05 that case scored tumor Dice 0.888 with the true liver box but 0.000 in the cascade. The aim: keep such tumors without bringing back the spurious far-away blobs v3 removed.

## Results

Chosen on validation; training cases as a secondary check (their predictions are in-sample, so somewhat optimistic). Test was read only after the decision. Full table: [`results/rule_summary.csv`](results/rule_summary.csv).

| Validation (78) | `Dice_vs_union` | Precision | Recall | `tumor_covered` | cases < 0.5 covered |
|---|---|---|---|---|---|
| all (= v2) | 0.9523 | 0.9256 | 0.9826 | 0.9759 | 1 |
| **largest (= v3)** | **0.9711** | **0.9672** | 0.9761 | 0.9759 | 1 |
| size ≥ 5% / 10% / 20% | 0.9678 / 0.9699 / 0.9706 | 0.956–0.961 | 0.9811 | 0.9759 | 1 |
| near 5 mm | 0.9711 (identical to v3) | 0.9672 | 0.9761 | 0.9759 | 1 |
| near 10 / 20 mm | 0.9707 / 0.9705 | 0.966 | 0.977 | 0.9759 | 1 |

Training (360) shows the same pattern: near 5 mm is identical to largest (0.9772), and every other rule is slightly lower. `tumor_covered` is 0.9795–0.9803 for every rule.

## Observations

- **The targeted failure doesn't occur in validation or training.** `tumor_covered` is identical under every rule, including keeping all components. The one low-coverage case per split is a tumor the liver model never predicted, which no component rule can fix.
- **Size rules cost precision** (they re-admit spurious blobs). Distance rules are neutral at 5 mm and slightly worse at 10–20 mm.
- **Test diagnostic (after the decision).** For 231025c23, no size or distance rule recovers the tumor (`tumor_covered` 0.005 under all of them). Only keeping all components does (0.573). The predicted tumor blob is ≥ 1,000 voxels, but more than 20 mm from the main liver component and less than 5% of its size. A component rule can't separate it from spurious blobs.

## Decision

- **Keep v3** (largest component). No rule improves validation, so the pipeline doesn't change.
- **The exophytic-tumor miss needs a model-level fix**, not post-processing. Options for later:
  - a liver model that keeps such tumors connected, e.g. more weight on tumor-adjacent liver boundary;
  - a Stage 2 search region not limited to the predicted liver (e.g. a dilated box);
  - a second-pass tumor detector over the whole abdomen for large lesions.

  Not scheduled; it's one case in 78 on test and none in validation.
