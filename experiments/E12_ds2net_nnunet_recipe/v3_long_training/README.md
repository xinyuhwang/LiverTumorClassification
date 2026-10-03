# E12 · v3 — DS²Net Stage 2, twice as long training

| | |
|---|---|
| Status | running: job 1169257 (submitted 2026-10-03 ~01:35, ~10.5 h) |
| Compared with | E12 v2 (`e12_v2_fixed_norm`), cascade mode, on validation |
| Change | `epochs=100` (v2: 50) and `early_stop_patience=1000` (never stops). The OneCycle schedule stretches over 100 epochs; samples per epoch unchanged. Checkpoint = best slice-level val Dice, as before |
| Code | commit `b4810cd`; v2's `--set` plus `epochs=100 early_stop_patience=1000` |
| Run | `$HERALD_STORE/results/runs/ds2net/e12_v3_long` |
| Date | 2026-10-03 |

## Hypothesis

nnU-Net trains far longer (1,000 epochs × 250 steps). The E12 plan included testing whether DS²Net is undertrained.

**Expected: no gain.** In v1 and v2, early stopping never triggered (both ran all 50 epochs), and slice-level val Dice levelled off around epoch 30 (v2: 0.8712 at 30, 0.8676 at 50) while train Dice kept rising (0.97). That looks like the start of overfitting, not undertraining. This version confirms the plateau before moving on. It was run in parallel with v4, at the user's request.

## How to reproduce

See [`run.sh`](run.sh). ~10.5 h on 1 GPU (100 × ~350 s + scoring).

## Results

```bash
python common/evaluate.py compare ../v2_fixed_norm/results/per_case_val.csv results/per_case_val.csv --md
```

## Observations

## Decision
