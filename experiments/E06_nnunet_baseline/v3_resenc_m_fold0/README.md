# E06 · v3 — nnU-Net ResEnc M preset, fold 0

| | |
|---|---|
| Status | done: plans on CPU (~1 min), training job 1164464 (1,000 epochs in 6 h 41 min), prediction + scoring job 1173295 (15 min); 2026-10-03 |
| Compared with | v1 (default `nnUNetPlans`, fold 0), on validation |
| Change | plans `nnUNetResEncUNetMPlans` (planner `nnUNetPlannerResEncM`): residual encoder U-Net, planned for ~8 GB GPU memory. Same 3d_fullres preprocessed data (ResEnc reuses `nnUNetPlans_3d_fullres`; patch 28 × 224 × 224 vs v1's 32 × 256 × 224, batch 2), same fold 0 = HERALD's 360 / 78 split, same trainer and 1,000 epochs |
| Code | commit `b4810cd`; jobs `nnunet_train` / `nnunet_predict` with `NNUNET_PLANS=nnUNetResEncUNetMPlans` |
| Run | `$nnUNet_results/Dataset501_MCTLTDiag/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_0`; scores in `$HERALD_RESULTS/runs/nnunet/e06_v3_resenc_m_fold0/{tumor,liver}/per_case_{val,test}.csv` |
| Date | 2026-10-02 |

## Hypothesis

v2 showed that averaging five default models doesn't help: the default network plateaus around 0.80 per case, held down by small tumors. nnU-Net's authors now recommend the residual-encoder presets over the default (Isensee et al., "nnU-Net Revisited", MICCAI 2024), with consistent gains on public CT benchmarks including liver tumors. More encoder capacity may pick up faint small lesions. Expect at most a few points, mostly < 50 ml.

## Steps and cost

See [`run.sh`](run.sh):
1. Plan only (`nnUNetv2_plan_experiment -pl nnUNetPlannerResEncM`, ~1 min on CPU). No re-preprocessing; `splits_final.json` is untouched.
2. Train fold 0: ~6–8 h on 1 GPU (ResEnc M is larger than the default network).
3. Predict test and score val + test: ~15 min.

## Results

### v3 vs v1 (B − A, paired, 95% bootstrap CI)

| | v1 (default) | v3 (ResEnc M) | Diff (95% CI) | Better / worse |
|---|---|---|---|---|
| **Val tumor Dice** | 0.796 | **0.808** | +0.013 (−0.015 to +0.047), Wilcoxon p 0.50 | 42 / 36 |
| Val tumor precision | 0.847 | 0.862 | +0.015 (−0.011 to +0.052) | 38 / 40 |
| Val tumor recall | 0.800 | 0.795 | −0.005 (−0.020 to +0.009) | 40 / 37 |
| Val tumor global Dice | 0.895 | **0.913** (0.886–0.927) | | |
| Val liver Dice | 0.969 | 0.973 | **+0.004 (+0.001 to +0.008)** | 56 / 22 |
| Val cases with tumor Dice < 0.5 | 4 | 5 | | |
| Test tumor Dice (report) | 0.796 | 0.790 (0.749–0.828) | | |
| Test tumor global Dice | 0.883 | 0.890 (0.855–0.928) | | |
| Test liver Dice | 0.974 | 0.975 | | |

vs v2 (5-fold ensemble) on val: +0.001 (−0.011 to +0.012), 43 / 35: one ResEnc M model matches the ensemble.

### Tumor Dice by total tumor volume and type (val, per-case mean; diff vs v1 with 95% CI)

| Group | n | v1 | v3 | Diff |
|---|---|---|---|---|
| < 10 ml | 16 | 0.653 | 0.681 | +0.027 (−0.066 to +0.142) |
| 10–50 ml | 20 | 0.765 | 0.796 | +0.031 (−0.026 to +0.123) |
| 50–200 ml | 16 | 0.792 | 0.778 | −0.015 (−0.058 to +0.009) |
| ≥ 200 ml | 26 | 0.909 | 0.916 | +0.006 (+0.000 to +0.012) |

By type (val): BCLM 0.780, CRLM 0.826, HCC 0.705, HH 0.933, ICC 0.804.

Files: [`results/`](results/): `tumor/` and `liver/` per-case CSVs, summaries, `compare_val_vs_v1.json`.

## Observations

- **Same picture as v2:** a +0.013 per-case gain on val that isn't significant, and a flat test (0.790 vs 0.796). Global Dice rises on both (val 0.913, test 0.890).
- **One ResEnc M model ≈ the 5-model ensemble** (+0.001 vs v2), at a fifth of v2's training and prediction cost (6.7 vs ~26 GPU-h).
- **Liver improves significantly** (+0.004), as in v2.
- **Small and mid-size tumors move up** (+0.027, +0.031) but with wide CIs (16 and 20 cases). The per-case mean for all tumors still sits at about 0.80–0.81.
- **Three nnU-Net variants now agree:** default, 5-fold ensemble and ResEnc M all land at 0.796–0.808 per case on val and 0.79–0.80 on test. Network capacity and ensembling aren't the limit; small and faint tumors are.

## Decision (made on validation)

- **Adopt v3 (ResEnc M) as the nnU-Net reference** for later comparisons. A judgement call: its tumor gain over v1 is not significant, but it equals the ensemble with one model, improves liver significantly, has the highest val global Dice so far (0.913), and is nnU-Net's recommended preset.
- **Stop scaling nnU-Net.** ResEnc L or an ensemble of ResEnc models is unlikely to move the per-case mean. Next efforts go to small tumors: per-lesion detection metrics and failure rates (R1), label checks (R6), and methods aimed at small lesions.
