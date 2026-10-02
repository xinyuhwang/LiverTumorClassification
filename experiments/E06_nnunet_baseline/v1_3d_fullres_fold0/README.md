# E06 · v1 — nnU-Net 3d_fullres, fold 0

| | |
|---|---|
| Status | done: preprocessing job 1154179 (10 min), training job 1154309 (all 1,000 epochs in 5 h 21 min, 18 s/epoch), prediction + scoring job 1157913 (10 min); 2026-10-02 |
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

## Preprocessing (AICR job 1154179, 10 min, commit `b092880`)

- 438 training cases (360 train + 78 val) and 78 test cases converted; 438 preprocessed for `3d_fullres` (44 GB on scratch). Our `splits_final.json` was kept (fold 0 = 360 / 78).
- `--verify_dataset_integrity` passed, with **warnings for 11 cases**: "direction mismatch" between the phase images and the label. The difference is about 1e-7 in the direction cosines (a tilt of ~1e-5°, < 0.0001 mm across a slice): rounding from NIfTI's quaternion orientation when the labels were re-saved with nibabel. Spacing, origin and shape match. Harmless; no fix needed.

## Results

nnU-Net's own summary ("Mean Validation Dice" 0.874, averaged over liver and tumor) isn't comparable with ours. All numbers below come from HERALD's metric code: per case, full volume, original grid, nnU-Net's default mirroring TTA.

### Overall (95% bootstrap CI)

| | Validation | Test |
|---|---|---|
| **Tumor Dice, per-case mean** | **0.796** (0.755–0.833), median 0.841 | **0.796** (0.756–0.832), median 0.851 |
| **Tumor Dice, global** | **0.895** (0.858–0.915) | **0.883** (0.837–0.929) |
| Tumor precision / recall | 0.847 / 0.800 | 0.847 / 0.772 |
| Liver Dice (vs liver ∪ tumor) | 0.969 (0.962–0.975) | 0.974 (0.969–0.977) |
| Liver `tumor_covered` | 0.979 | 0.954 |

### Paired comparison with DS²Net (B − A = nnU-Net − DS²Net)

| | Validation (decision) | Test (report) |
|---|---|---|
| Tumor Dice vs DS²Net cascade (E05 v1) | +0.026 (−0.020 to +0.069); bootstrap p 0.22, Wilcoxon p 0.027; 49 better / 29 worse | **+0.070 (+0.029 to +0.112)**; 56 / 21 |
| Tumor precision vs DS²Net cascade | +0.034 (−0.018 to +0.080) | +0.062 (+0.028 to +0.100) |
| Tumor Dice vs DS²Net oracle | +0.024 (−0.022 to +0.066) | — |
| Liver `Dice_vs_union` vs DS²Net E01 v3 | −0.002 (−0.007 to +0.002) | — |
| Tumor global Dice (DS²Net vs nnU-Net) | 0.893 vs 0.895 | 0.865 vs 0.883 |

### Tumor Dice by size and type (per-case mean)

| Group | Validation (n) | Test (n) |
|---|---|---|
| < 10 ml | 0.653 (16) | 0.714 (20) |
| 10–50 ml | 0.765 (20) | 0.715 (22) |
| 50–200 ml | 0.792 (16) | 0.880 (22) |
| **≥ 200 ml** | **0.909** (26) | **0.910** (14) |
| BCLM | 0.782 | 0.723 |
| CRLM | 0.722 | 0.784 |
| HCC | 0.752 | 0.756 |
| **HH** | **0.929** | **0.901** |
| ICC | 0.797 | 0.836 |

Files: [`results/`](results/) — `tumor/` and `liver/` per-case CSVs, summaries with CIs, and the paired comparisons.

## Observations

- **nnU-Net is the strongest tumor model so far, and the most stable.** Per-case Dice is 0.796 on both validation and test, while DS²Net dropped from 0.769 (val) to 0.726 (test). On validation the advantage over DS²Net is consistent per case (49 vs 29, Wilcoxon p = 0.027), but the mean's CI includes 0. On test it's clear (+0.070).
- **Liver is equivalent** to DS²Net's adopted E01 v3 (−0.002, n.s.), with no post-processing and in one model.
- **What's achievable versus the 0.90 goal:**
  - Large tumors (≥ 200 ml) and hemangiomas reach 0.90–0.93.
  - Global Dice is 0.895 (val) / 0.883 (test), with CIs including 0.90.
  - The per-case mean over all tumors is about 0.80, held down by small lesions (< 10 ml: 0.65–0.71).

  That matches published liver-tumor results (~0.80–0.83 per case on LiTS). A per-case mean of 0.90 over all tumor sizes is beyond the current state of the art on this kind of data.
- **Where DS²Net and nnU-Net differ:** on global Dice they tie on validation (0.893 vs 0.895). nnU-Net's per-case gain comes from small and mid-size tumors: full 3D context at native z-spacing, and all four phases without PhaseNorm.
- **Cost:** a single model with no cascade, 5 h 21 min to train, ~10 min to predict and score 78 cases.

## Decision (made on validation)

- **nnU-Net 3D is the leading candidate for the main segmentation model.** It's at least as good as DS²Net on liver and tumor on validation (tumor +0.026, n.s. by CI but consistent per case), more precise, stable across splits, and simpler: one model, no cascade.
- **Next:**
  1. **v2 = 5-fold ensemble** (the standard nnU-Net protocol, usually +1–3 points). Needs 5-fold CV over train + val, with validation predictions per fold. About 5 × 5.5 GPU-hours.
  2. **Small tumors (< 50 ml)** are where the gap to 0.90 is. Candidates: nnU-Net's `3d_cascade_fullres` or a higher-resolution plan, a small-lesion-weighted loss or sampling, and per-lesion detection metrics.
- **The 0.90 target needs a definition with the team.** It's met for large tumors and hemangiomas, nearly met on global Dice, and not met as a per-case mean over all tumors.
