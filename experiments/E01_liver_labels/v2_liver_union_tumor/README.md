# E01 · v2 — liver label = liver mask ∪ tumor

| | |
|---|---|
| Status | done (AICR job 1130505, 8 h 47 min on 1× RTX PRO 6000, finished 2026-09-30) |
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

60 epochs, no early stop. Best val Dice 0.9371 (epoch 42), scored against v2's own label. Test on 78 cases: full volume, 8-fold TTA, original grid. Liver masks for all 516 cases are on AICR (`e01_v2_liver_union/liver_masks/`).

### Paired comparison with v1

`python common/evaluate.py compare <v1> <v2> --md` (2,000 bootstrap resamples; B − A = v2 − v1):

| metric | v1 | v2 | diff | 95% CI | p (bootstrap) | p (Wilcoxon) | better | worse |
|---|---|---|---|---|---|---|---|---|
| **`tumor_covered`** | 0.9006 | 0.9476 | **+0.0469** | +0.0212 to +0.0796 | < 0.001 | < 0.001 | 59 | 2 |
| `Dice_vs_union` | 0.9481 | 0.9494 | +0.0013 | −0.0062 to +0.0068 | 0.66 | < 0.001 | 52 | 26 |
| Precision | 0.9206 | 0.9231 | +0.0026 | −0.0081 to +0.0114 | 0.59 | 0.003 | 50 | 28 |
| Recall | 0.9827 | 0.9802 | −0.0025 | −0.0057 to −0.0003 | 0.007 | 0.03 | 28 | 50 |

Median per-case change in `Dice_vs_union` is +0.004. Full output: [`results/compare_v1_v2.json`](results/compare_v1_v2.json).

### By tumor type

| Type | n | `tumor_covered` v1 → v2 | Δ `Dice_vs_union` mean / median |
|---|---|---|---|
| BCLM | 17 | 0.942 → 0.954 | −0.002 / +0.003 |
| CRLM | 16 | 0.966 → 0.972 | +0.003 / +0.005 |
| HCC | 16 | 0.935 → 0.951 | +0.004 / +0.003 |
| **HH** | 14 | **0.672 → 0.881** | −0.003 / +0.007 |
| ICC | 15 | 0.960 → 0.972 | +0.003 / +0.001 |

v2 summary with CIs: [`results/evaluate_summary.json`](results/evaluate_summary.json). Per case: [`results/per_case_test.csv`](results/per_case_test.csv).

## Observations

- **The hypothesis holds for tumor coverage.** It rises significantly (+0.047, CI excludes 0; 59 cases better, 2 worse), and most for hemangiomas (0.67 → 0.88). v1's three failures are recovered: 230218c4 12% → 97%, 231025c23 0.5% → 57%, 231025c14 18% → 67%. No test case has less than half of its tumor covered any more.
- **Overall liver agreement is unchanged in the mean** (`Dice_vs_union` +0.001, CI −0.006 to +0.007) **and slightly better for most cases** (median +0.004; 52 better, 26 worse).
- **A few cases over-segment more.** The largest drops are all precision losses with unchanged recall. The worst is 240229c27 (HH, 0.917 → 0.693, 1.79× the reference volume), then 230218a10 (BCLM, 0.932 → 0.872). Learning to include tumors that reach the liver surface seems to also teach the model to extend beyond it. This is v1's over-segmentation weakness, amplified in a few cases.
- **Large tumors are still not fully covered** (231025c23 at 57%). The union label fixes the label, but 2.5D slices of very large lesions remain hard.

## Decision

- **Adopt v2 as the liver label** from now on (`liver_includes_tumor=True`). The tumor-coverage gain matters downstream, because Stage 2 and ROI cropping rely on the liver region: a liver that leaves out the tumor hides it. It costs nothing in mean liver agreement.
- **Next (v3): reduce over-segmentation.** Add non-liver slices to Stage 1 training so the model learns "no liver here", keeping `liver_includes_tumor=True`. Success means precision up (worst cases such as 240229c27 and 231206d09) with `tumor_covered` held.
