# E05 · v1 — DS²Net Stage 2 on the liver region

| | |
|---|---|
| Status | done (AICR job 1139799, 6 h 19 min on 1× RTX PRO 6000, finished 2026-10-01) |
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

Best val Dice (slice-level, crops) 0.8555 at epoch 39. Test slice-level 0.854 (cached crops only, for reference).

### Per case, full volume, 8-fold TTA (95% bootstrap CI)

| Split | Mode | Tumor Dice | Precision | Recall |
|---|---|---|---|---|
| test | **cascade** (E01 liver) | **0.726** (0.676–0.772), median 0.799 | 0.785 | 0.725 |
| test | oracle (GT liver) | 0.740 (0.692–0.784), median 0.812 | 0.792 | 0.735 |
| val | cascade | 0.769 | 0.813 | 0.769 |
| val | oracle | 0.772 | 0.819 | 0.772 |

**Cascade cost, decided on validation:** −0.002 Dice (CI −0.007 to +0.003); 35 cases better, 42 worse. See [`results/compare_val_oracle_cascade.json`](results/compare_val_oracle_cascade.json).

### By tumor type (test, cascade)

| Type | n | Dice (95% CI) | Median tumor volume |
|---|---|---|---|
| BCLM | 17 | 0.611 (0.521–0.697) | 5.8 ml |
| CRLM | 16 | 0.752 (0.669–0.819) | 16.9 ml |
| HCC | 16 | 0.678 (0.538–0.801) | 69.0 ml |
| HH | 14 | 0.813 (0.670–0.901) | 133.0 ml |
| ICC | 15 | 0.797 (0.712–0.858) | 81.4 ml |

Full tables: [`results/evaluate_summary.json`](results/evaluate_summary.json).

## Observations

- **Size drives accuracy.** Spearman ρ(Dice, tumor volume) = 0.65. Mean Dice by tumor volume:

  | Tumor volume | Mean Dice | n |
  |---|---|---|
  | < 10 ml | 0.58 | 20 |
  | 10–50 ml | 0.69 | 22 |
  | 50–200 ml | 0.78 | 22 |
  | > 200 ml | 0.90 | 14 |

  BCLM is lowest because its tumors are the smallest (median 5.8 ml).
- **The cascade costs almost nothing on average.** E01's liver is good enough that predicted and true liver boxes give the same mean (validation Δ −0.002). The one exception is **231025c23** (exophytic HH): oracle 0.888 → cascade 0.000. E01 v3's largest-component step removed that tumor from the predicted liver, so the box never covered it. This is the failure flagged in E01 v3, and the strongest argument for keeping large secondary components there.
- **Misses:** 230816b07 (HCC, 2.5 ml) is missed in both modes. 5 test cases have Dice < 0.3: 3 small tumors (≤ 3 ml), and 2 larger ones with low recall (231109b04 HCC 118 ml, recall 0.13; 231206d09 ICC 15 ml, recall 0.19).
- **Comparison with older numbers** (indicative only): the notebook's whole-slice DS²Net had per-case Dice 0.375 (precision 0.31) on a different split with known bugs. The liver ROI plus the fixes roughly double it, and precision is now 0.79. The paper's UNet-Hybrid 0.631 isn't comparable (oracle liver, single phase, unknown evaluation); E05 v2 will give the paired comparison.

## Decision

- **Keep as the DS²Net tumor baseline:** cascade 0.726 (0.676–0.772).
- **Follow-ups, each its own version, decided on validation:**
  1. E01 refinement: keep large secondary liver components, so exophytic tumors stay inside the ROI (231025c23).
  2. Small-tumor recall (BCLM, < 10 ml): candidates are a higher input resolution for small livers, or a tumor-size-aware loss / sampler.
- **Next in E05:** UNet-Hybrid Stage 2 (v2), for the paired tumor comparison.
