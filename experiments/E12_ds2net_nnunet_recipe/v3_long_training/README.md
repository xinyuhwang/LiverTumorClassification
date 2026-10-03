# E12 · v3 — DS²Net Stage 2, twice as long training

| | |
|---|---|
| Status | done: job 1169257 (10 h 9 min on 1× RTX PRO 6000), 2026-10-03 |
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

Slice-level val Dice over 100 epochs (every 12th check): 0.553 (3) → 0.853 (15) → 0.862 (27) → **0.865 best (42)** → 0.858 (51) → 0.858 (63) → 0.859 (87) → 0.858 (99). Train Dice kept rising.

### Per case, full volume, cascade mode, vs v2 (B − A)

| | v2 (50 epochs) | v3 (100 epochs) | Diff (95% CI) | Better / worse |
|---|---|---|---|---|
| **Val tumor Dice** | 0.780 | 0.771 | −0.009 (−0.024 to +0.008) | 37 / 41 |
| Val precision | 0.804 | 0.814 | +0.010 (−0.007 to +0.028) | 46 / 31 |
| Val recall | 0.795 | 0.783 | −0.011 (−0.032 to +0.015) | 29 / 48 |
| Val global Dice | 0.900 | 0.897 | | |
| Val cases with Dice < 0.5 | 6 | 8 | | |
| Test tumor Dice (report) | 0.735 | 0.728 | | |

By size (val): < 10 ml −0.029 (−0.091 to +0.039); other groups within ±0.005.

## Observations

- **As expected, no gain:** slightly worse everywhere, within noise. Slice-level val Dice peaked at epoch 42 and then stayed flat or drifted down for 58 more epochs while train Dice rose: the model overfits, it isn't undertrained.
- The best checkpoint came from a phase where the stretched schedule still had a higher learning rate than v2's best epoch, which may explain the small loss.
- Cost: 10 h vs 5.4 h.

## Decision (made on validation)

- **Not adopted; keep 50 epochs.** DS²Net's gap to nnU-Net isn't training length. nnU-Net's longer schedule works with 3D patches and heavier augmentation over far more distinct samples; on 2.5D slices from 360 cases, DS²Net saturates by epoch 30–40.
