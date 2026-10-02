# E12 · v1 — DS²Net Stage 2 at native 5 mm slice spacing

| | |
|---|---|
| Status | planned (code ready, not submitted) |
| Compared with | E05 v1 (`e05_v1_ds2net_tumor_roi`), cascade mode, on validation |
| Change | `z_spacing=5.0`: keep the native 5 mm between slices instead of resampling z to 1 mm. In-plane stays 1 mm. `samples_per_epoch=59314` keeps E05 v1's slices per epoch, so the number of training steps and the LR schedule are unchanged |
| Code | commit `<hash>`; `--set roi=liver slices=liver liver_includes_tumor=True roi_liver_run=e01_v2_liver_union z_spacing=5.0 samples_per_epoch=59314` |
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

```bash
python common/evaluate.py summary results/per_case_val.csv --by-type --size-bins 10 50 200 --md
python common/evaluate.py compare ../../E05_tumor_baselines/v1_ds2net_tumor_roi/results/per_case_val.csv results/per_case_val.csv --md
```

## Observations

## Decision
