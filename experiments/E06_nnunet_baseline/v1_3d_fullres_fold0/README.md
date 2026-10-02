# E06 · v1 — nnU-Net 3d_fullres, fold 0

| | |
|---|---|
| Status | code ready (2026-10-01); not yet run on AICR |
| Compared with | DS²Net E05 v1 (tumor) and E01 v3 (liver) |
| Change | nnU-Net v2 (2.8.x) `3d_fullres`, default `nnUNetTrainer`, trained on fold 0 = HERALD's 360 train / 78 val |
| Run | `$nnUNet_results/Dataset501_MCTLTDiag/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0`; scores in `$HERALD_RESULTS/runs/nnunet/e06_v1_3d_fullres_fold0/{tumor,liver}/per_case_{val,test}.csv` |

## Hypothesis

nnU-Net should at least match DS²Net on tumors. It sees full 3D context at native z-spacing (DS²Net's z-resampling makes its 2.5D context mostly interpolation), and it uses the four phases without PhaseNorm's per-channel erasure. A per-case mean above 0.90 is unlikely: published liver-tumor results are about 0.80–0.83 per case and 0.86–0.87 global on LiTS. Global Dice above 0.90 on validation is plausible.

## Steps

See [`run.sh`](run.sh):

1. Install `nnunetv2` in `~/envs/herald`.
2. `nnunet_prep`: build the raw dataset, then plan and preprocess with the integrity check. CPU, ~1–2 h, ~100–200 GB on scratch.
3. `nnunet_train`: about 1–2 GPU-days; resubmit with `--c` after each 24 h limit.
4. `nnunet_predict`: predict test, then score validation and test with HERALD's metric code.

## Local verification (2026-10-01, synthetic data)

- `prepare_dataset.py` → nnU-Net's `--verify_dataset_integrity` passed; 3d_fullres planned and preprocessed with CT normalisation on all 4 phases; our `splits_final.json` was kept.
- `evaluate_predictions.py` with ground-truth labels as predictions → Dice 1.0 for tumor and liver; `evaluate.py` global Dice and size bins work on its output.

## Results

—

## Decision

—
