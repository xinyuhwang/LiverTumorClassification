# E05 · UNet-Hybrid Stage 1 (liver) — prerequisite for v2

| | |
|---|---|
| Status | done (AICR job 1137474, 7 h 29 min, 40 epochs, finished 2026-10-01) |
| Purpose | UNet-Hybrid Stage 2 starts from this checkpoint; its test output also gives the first paired liver comparison with DS²Net |
| Code | commit `c742027`; `bash cluster/submit.sh unet_hybrid --stage 1` (official liver mask, no TTA, no post-processing) |
| Results | [`results/`](results/): `per_case_test.csv`, `metrics.json`, and the comparison with DS²Net |

## Results (test, 78 cases, per case)

Best val Dice (batch-pooled slices) 0.9413. Per-case test liver Dice **0.971 ± 0.021**, which reproduces the paper's 0.9704.

| Comparison (test, paired, 2,000 bootstrap resamples) | DS²Net | UNet-Hybrid | diff (95% CI) |
|---|---|---|---|
| Same official label, as trained: DS²Net E01 v1 vs UNet-Hybrid, Dice | 0.950 | **0.971** | +0.021 (+0.015 to +0.029); 69 better / 9 worse |
| … precision | 0.921 | 0.963 | +0.042 (+0.031 to +0.055) |
| … recall | 0.983 | 0.980 | −0.003 (−0.006 to −0.000) |
| DS²Net as adopted (E01 v3) vs UNet-Hybrid, `Dice_vs_union` | **0.973** | 0.968 | −0.004 (−0.008 to −0.001); Wilcoxon p = 0.71 |
| … `tumor_covered` | **0.940** | 0.887 | −0.053 (−0.088 to −0.025) |

## Observations

- **Same label, UNet-Hybrid is clearly better**, almost entirely through precision. It trains with 15% of slices containing no liver (`LiverDataset.BG_RATIO`); DS²Net doesn't. That matches E01 v3's finding that DS²Net's false positives are blobs away from the liver.
- **After E01's changes, DS²Net matches UNet-Hybrid** on the common reference (difference −0.004, not robust: Wilcoxon p = 0.71). It covers more tumor (0.940 vs 0.887), because UNet-Hybrid still learns the official mask's tumor gaps.
- **Not yet applied to UNet-Hybrid:** the E01 label (liver ∪ tumor) and largest-component post-processing. UNet-Hybrid doesn't save probability maps or per-case validation output yet, so these comparisons are on test only and are reported, not used for decisions.
- **Follow-up for E02:** done in E02 v4 (per-case validation output). Saving liver probabilities is still open.
