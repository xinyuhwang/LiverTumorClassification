# E12 · v1 — DS²Net Stage 2 at native 5 mm slice spacing

| | |
|---|---|
| Status | done: smoke test 1159286 (11 min), full run 1159797 (5 h 44 min on 1× RTX PRO 6000), 2026-10-02 |
| Compared with | E05 v1 (`e05_v1_ds2net_tumor_roi`), cascade mode, on validation |
| Change | `z_spacing=5.0`: keep the native 5 mm between slices instead of resampling z to 1 mm. In-plane stays 1 mm. `samples_per_epoch=59314` keeps E05 v1's slices per epoch, so the number of training steps and the LR schedule are unchanged |
| Code | commit `070abed`; `--set roi=liver slices=liver liver_includes_tumor=True roi_liver_run=e01_v2_liver_union z_spacing=5.0 samples_per_epoch=59314` |
| Run | `$HERALD_STORE/results/runs/ds2net/e12_v1_native_z` |
| Date | 2026-10-02 |

## Hypothesis

All 517 cases have 5 mm slices. At 1 mm, DS²Net's three context slices span 2 mm and are linear interpolations between the same two real slices, so the "2.5D" input carries almost no extra information. At 5 mm, the context becomes the real slices above and below (±5 mm), which is the part of nnU-Net's 3D context a 2.5D model can use.

- **Expected:** a gain mainly on small and mid-size tumors (< 50 ml), where one neighbouring slice decides whether a faint lesion is real.
- **Side effects:**
  - About 5× fewer distinct training slices (~12k instead of 59k). With the same samples per epoch, each real slice is seen about 5× more often, under augmentation.
  - Validation and the checkpoint choice use 5 mm slices.
  - Final scores are on the original grid as before.
- **Risk:** in-plane boxes and masks are unchanged, but a slice that only grazes a tumor is no longer smoothed into several 1 mm slices, so slice-level Dice may look lower. The per-case full-volume score is what counts.

## How to reproduce

See [`run.sh`](run.sh). Before the full run: a ~10 min smoke test on `rtx-devel`. Full run: ~6 h on 1 GPU. Training time is about the same as E05 v1 (same samples per epoch), but the cache and per-case prediction are faster.

Needs E01 v2's saved liver masks (`$HERALD_WORK/runs/ds2net/e01_v2_liver_union/liver_masks`, written 2026-09-30). `/scratch` purges files after 30 days, so run before about 2026-10-30, or regenerate them with `--eval_only --save_liver_masks`.

## Results

Training: 11,722 distinct training slices (E05 v1: 59,314), the same 59,314 draws per epoch. Best slice-level val Dice 0.8716 at epoch 36 (E05 v1: 0.8555 at 39), but slice-level numbers aren't comparable across slice spacings. Test slice-level 0.862.

### Per case, full volume, cascade mode, vs E05 v1 (B − A)

| | E05 v1 | E12 v1 | Diff (95% CI) | Better / worse |
|---|---|---|---|---|
| **Val tumor Dice** | 0.769 | **0.767** | −0.002 (−0.022 to +0.015), Wilcoxon p 0.68 | 39 / 38 |
| Val precision | 0.813 | 0.820 | +0.006 (−0.009 to +0.021) | 44 / 33 |
| Val recall | 0.769 | 0.759 | −0.010 (−0.030 to +0.008) | 37 / 40 |
| Val global Dice | 0.893 | 0.901 | | |
| Val cases with Dice < 0.5 | 9 | 7 | | |
| Test tumor Dice (report) | 0.726 | 0.722 | | |
| Test global Dice | 0.865 | 0.864 | | |

### By total tumor volume (val, per-case mean, diff with 95% CI)

| Group | n | E05 v1 | E12 v1 | Diff |
|---|---|---|---|---|
| < 10 ml | 16 | 0.506 | 0.512 | +0.006 (−0.054 to +0.065) |
| 10–50 ml | 20 | 0.794 | 0.777 | −0.018 (−0.069 to +0.022) |
| 50–200 ml | 16 | 0.814 | 0.811 | −0.004 (−0.019 to +0.007) |
| ≥ 200 ml | 26 | 0.884 | 0.891 | +0.006 (−0.002 to +0.016) |

vs nnU-Net (E06 v1) on val: −0.028 (−0.069 to +0.019), 31 better / 47 worse.

Files: [`results/`](results/): per-case CSVs (with `gt_ml` added from E06 for size bins), `summary_val.json`, `compare_val_vs_e05v1.json`.

## Observations

- **No effect.** Per-case Dice, the size groups and test are all within ±0.02 of E05 v1, with CIs centred on 0. Small tumors, where real neighbouring slices should matter most, didn't improve (+0.006, CI −0.05 to +0.07).
- **The hypothesis was wrong, or something else blocks it.** Real ±5 mm context replaced interpolated ±1 mm context without changing results. PhaseNorm was still on in this version. Two untested explanations:
  - PhaseNorm normalises each input channel (each phase of each slice) separately, removing absolute intensity relations across slices and phases. That may limit what the model gets from real neighbours, though shape and edges still come through. **v2 tests this** by removing PhaseNorm and changing nothing else.
  - Or 2.5D context (one slice each side) is simply too little; nnU-Net sees 32 slices.
- **It's cheaper at no cost.** 5× fewer cached slices (smaller cache, faster preprocessing), same training steps, run 35 min shorter.
- Slightly fewer failed cases on val (7 vs 9) and a slightly higher global Dice (0.901 vs 0.893), both too small to count.

## Decision (made on validation)

- **Not an improvement, but equivalent and cheaper:** keep `z_spacing=5.0` as the base for the next versions. It's the more faithful input (real slices), and it makes v2's test cleaner, since PhaseNorm's effect across slices is now across real slices.
- **Next: v2 = remove PhaseNorm** (fixed per-phase intensity scaling), on top of v1. It's one of the clearest differences from nnU-Net, and it may be why the extra context didn't help; v2 tells us whether it is.
- R3 (slice-interaction module) was conditional on v1 showing that slice context helps. It didn't, so R3 stays parked until v2 is in.
