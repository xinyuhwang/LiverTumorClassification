# E12 · v4 — DS²Net Stage 2 with nnU-Net's loss (Dice + cross-entropy)

| | |
|---|---|
| Status | planned (code ready, not submitted) |
| Compared with | E12 v2 (`e12_v2_fixed_norm`), cascade mode, on validation. Runs in parallel with v3, so it builds on v2 (50 epochs), not v3 |
| Change | `loss=dice_ce`: soft Dice pooled over the batch + unweighted BCE on each head, deep-supervision weights halving from the finest head, the small-tumor branch with the same loss (weight 0.4). Replaces DS²Net's uncertainty-weighted wIoU + wBCE (pos_weight 10, +5 warm-up) + boundary loss. Inference (head fusion, threshold 0.5, TTA) unchanged |
| Code | commit `<hash>`; v2's `--set` plus `loss=dice_ce` |
| Run | `$HERALD_STORE/results/runs/ds2net/e12_v4_dice_ce` |
| Date | 2026-10-03 |

## Hypothesis

DS²Net's loss up-weights tumor pixels strongly (BCE pos_weight 10, plus 5 extra during warm-up, plus a boundary term). After v2 the model finds more tumor (recall +0.035) but draws looser borders (precision −0.015). A heavily positive-weighted loss pushes in exactly that direction. nnU-Net's Dice + CE has no positive weighting; Dice already balances foreground and background. Expect better precision at similar recall, so a higher Dice, most visible on mid-size tumors.

**Risk:** small tumors may lose some of v2's recall gain, since the positive weighting helped faint lesions.

## How to reproduce

See [`run.sh`](run.sh): a ~10 min smoke test, then the full run (~5.5 h).

## Results

```bash
python common/evaluate.py compare ../v2_fixed_norm/results/per_case_val.csv results/per_case_val.csv --md
```

## Observations

## Decision
