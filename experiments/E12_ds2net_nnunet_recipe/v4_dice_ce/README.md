# E12 · v4 — DS²Net Stage 2 with nnU-Net's loss (Dice + cross-entropy)

| | |
|---|---|
| Status | done: smoke test 1169285 (6 min), full run 1173294 (5 h 17 min on 1× RTX PRO 6000), 2026-10-03 |
| Compared with | E12 v2 (`e12_v2_fixed_norm`), cascade mode, on validation. Runs in parallel with v3, so it builds on v2 (50 epochs), not v3 |
| Change | `loss=dice_ce`: soft Dice pooled over the batch + unweighted BCE on each head, deep-supervision weights halving from the finest head, the small-tumor branch with the same loss (weight 0.4). Replaces DS²Net's uncertainty-weighted wIoU + wBCE (pos_weight 10, +5 warm-up) + boundary loss. Inference (head fusion, threshold 0.5, TTA) unchanged |
| Code | commit `6631567`; v2's `--set` plus `loss=dice_ce` |
| Run | `$HERALD_STORE/results/runs/ds2net/e12_v4_dice_ce` |
| Date | 2026-10-03 |

## Hypothesis

DS²Net's loss up-weights tumor pixels strongly (BCE pos_weight 10, plus 5 extra during warm-up, plus a boundary term). After v2 the model finds more tumor (recall +0.035) but draws looser borders (precision −0.015). A heavily positive-weighted loss pushes in exactly that direction. nnU-Net's Dice + CE has no positive weighting; Dice already balances foreground and background. Expect better precision at similar recall, so a higher Dice, most visible on mid-size tumors.

**Risk:** small tumors may lose some of v2's recall gain, since the positive weighting helped faint lesions.

## How to reproduce

See [`run.sh`](run.sh): a ~10 min smoke test, then the full run (~5.5 h).

## Results

Best slice-level val Dice 0.8537 at epoch 30 (v2: 0.8712). Test slice-level 0.853.

### Per case, full volume, cascade mode, vs v2 (B − A)

| | v2 (DS²Net loss) | v4 (Dice + CE) | Diff (95% CI) | Better / worse |
|---|---|---|---|---|
| **Val tumor Dice** | 0.780 | **0.751** | **−0.030 (−0.048 to −0.013)**, bootstrap p 0.001, Wilcoxon p 0.004 | 31 / 47 |
| Val precision | 0.804 | 0.852 | **+0.048 (+0.028 to +0.069)** | 69 / 7 |
| Val recall | 0.795 | 0.713 | **−0.082 (−0.102 to −0.064)** | 2 / 76 |
| Val global Dice | 0.900 | 0.884 | | |
| Val cases with Dice < 0.5 | 6 | 14 | | |
| Test tumor Dice (report) | 0.735 | 0.713 | | |
| Test precision / recall | 0.758 / 0.754 | 0.830 / 0.669 | | |

### By total tumor volume (val, per-case mean, diff with 95% CI)

| Group | n | v2 | v4 | Diff |
|---|---|---|---|---|
| < 10 ml | 16 | 0.554 | 0.470 | **−0.083 (−0.143 to −0.028)** |
| 10–50 ml | 20 | 0.795 | 0.771 | −0.024 (−0.069 to +0.007) |
| 50–200 ml | 16 | 0.805 | 0.791 | −0.013 (−0.031 to +0.004) |
| ≥ 200 ml | 26 | 0.893 | 0.883 | −0.010 (−0.027 to +0.006) |

By type (val): BCLM 0.758, CRLM 0.701, HCC 0.599, HH 0.916, ICC 0.779. vs nnU-Net E06 v3: −0.058 (−0.088 to −0.027).

## Observations

- **Significantly worse, and the risk named in the hypothesis happened.** Precision rose as predicted (+0.048), but recall fell much more (−0.082, lower in 76 of 78 cases), so Dice dropped. Failed cases more than doubled (6 → 14), and small tumors lost most (−0.083, significant).
- **Why nnU-Net's loss doesn't transfer:** nnU-Net pairs Dice + CE with 3D patches and forced foreground oversampling (a third of patches contain tumor). DS²Net trains on whole 2D liver slices, most with no or little tumor; batch Dice + unweighted CE then lets the model under-segment. DS²Net's positive weighting compensates for that, so it's part of what makes 2.5D slices work, not a mistake.
- **Threshold is a confound worth noting:** the v4 model is more conservative, and inference still uses threshold 0.5. A lower threshold tuned on val might recover some recall, but that would be a separate post-processing version.

## Decision (made on validation)

- **Not adopted; keep DS²Net's loss.** v2 remains the E12 base.
- E12's original plan ends here except v5 (ensemble). The remaining gap to nnU-Net (−0.03 per case on val) looks like 3D context, which the 2.5D design can't supply.
