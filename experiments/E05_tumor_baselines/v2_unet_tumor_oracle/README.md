# E05 · v2 — UNet-Hybrid Stage 2 (oracle liver crop)

| | |
|---|---|
| Status | done: training AICR job 1145658 (6 h 27 min, early stop at epoch 33); validation via `--eval_only` job 1151355 (33 min), both 2026-10-01 |
| Compared with | v1 DS²Net Stage 2, **oracle** mode (both given the true liver) |
| Change | UNet-Hybrid Stage 2 as in `unet_hybrid/` (warm-started from its Stage 1, [`../unet_stage1_liver/`](../unet_stage1_liver/)): per slice, a rectangular crop around the **official** liver mask, 12-ch (4 phases × 3 slices), no TTA |
| Code | training at commit `d88c4d3`; validation via `--eval_only` at `52b736f` |
| Run | `$HERALD_STORE/results/unet_hybrid/logs/stage2/` |
| Date | 2026-10-01 |

## Results (test, 78 cases, per case, full volume)

The eval-only job reproduced `per_case_test.csv` exactly (all 78 cases, max |ΔDice| = 0).

| | Tumor Dice (95% CI) | Precision | Recall |
|---|---|---|---|
| UNet-Hybrid (oracle) | **0.657** (0.600–0.709), median 0.724 | 0.667 | 0.727 |
| DS²Net v1 (oracle) | 0.740 (0.692–0.784), median 0.812 | 0.792 | 0.735 |

Paired, UNet-Hybrid − DS²Net (2,000 bootstrap resamples): Dice **−0.083 (−0.127 to −0.041)**, 21 better / 56 worse; precision −0.126 (−0.168 to −0.084); recall −0.008 (−0.056 to +0.038, n.s.). Files: [`results/compare_test_ds2net_oracle_vs_unet.json`](results/compare_test_ds2net_oracle_vs_unet.json), [`results/evaluate_summary.json`](results/evaluate_summary.json).

By type (Dice, UNet-Hybrid vs DS²Net cascade from v1): BCLM 0.548 vs 0.611 · CRLM 0.688 vs 0.752 · HCC 0.614 vs 0.678 · HH 0.666 vs 0.813 · ICC 0.784 vs 0.797.

The paper reported 0.6313 for UNet-Hybrid Stage 2. The 0.657 here uses all four phases (the phase-key fix), the corrected split, and the shared metric code.

## Results (validation, 78 cases) — the basis for the decision

| | Tumor Dice (95% CI) | Precision | Recall |
|---|---|---|---|
| UNet-Hybrid (oracle) | 0.754 (0.715–0.791), median 0.823 | 0.754 | 0.805 |
| DS²Net v1 (oracle) | 0.772 | 0.819 | 0.772 |

Paired, UNet-Hybrid − DS²Net: Dice **−0.018 (−0.045 to +0.009)**, bootstrap p = 0.18, Wilcoxon p = 0.011, 27 better / 51 worse; precision −0.065 (−0.091 to −0.035); recall **+0.033** (−0.004 to +0.068). File: [`results/compare_val_ds2net_oracle_vs_unet.json`](results/compare_val_ds2net_oracle_vs_unet.json).

## Observations

- **The test gap is larger than the validation gap.** Test −0.083 vs validation −0.018. Validation's mean difference isn't significant, but the per-case direction is consistent (DS²Net better in 51 of 78; Wilcoxon p = 0.011). UNet-Hybrid also drops more from validation to test (0.754 → 0.657) than DS²Net (0.772 → 0.740).
- **DS²Net's consistent advantage is precision** (validation −0.065, test −0.126, both CIs exclude 0). UNet-Hybrid trades it for somewhat higher recall on validation (+0.033, n.s.).
- **Part of the gap is the crop, not the model.** "Oracle" doesn't mean the same box for both: UNet-Hybrid's rectangle follows the official liver mask; DS²Net's square follows liver ∪ tumor (E01's lesson).

  | Test cases | n | Δ Dice (UNet − DS²Net) mean / median |
  |---|---|---|
  | tumor ≥ 99% inside official liver mask | 26 | −0.084 / −0.031 |
  | tumor < 99% inside | 52 | −0.083 / −0.048 |
  | tumor < 90% inside | 19 | **−0.155 / −0.128** (UNet recall 0.58 vs 0.71) |

  Where the official mask leaves out more than 10% of the tumor, UNet-Hybrid's box cuts it off and the gap roughly doubles. A gap remains even when the box contains the whole tumor.
- HH shows the largest type difference (0.666 vs 0.813), consistent with hemangiomas being the type the official mask most often leaves out.

## Decision (made on validation)

- **Comparable mean tumor Dice under oracle conditions**, with DS²Net more precise and better in most cases (validation Δ −0.018, CI −0.045 to +0.009; precision Δ −0.065). Test shows a larger gap (−0.083), but decisions follow validation.
- **Keep DS²Net Stage 2 as the primary tumor model** for the pipeline: equal or better Dice, clearly better precision, and it already has the cascade (predicted-liver) mode.
- **UNet-Hybrid remains a viable alternative**, not ruled out. If it's pursued, v3 = UNet-Hybrid Stage 2 with a liver ∪ tumor crop, to separate model quality from the crop protocol (the 19 cases with < 90% of the tumor inside the official mask show a test gap of −0.155).
