# E06 · v3 — nnU-Net ResEnc M preset, fold 0

| | |
|---|---|
| Status | planned (scripts ready, not submitted) |
| Compared with | v1 (default `nnUNetPlans`, fold 0), on validation |
| Change | plans `nnUNetResEncUNetMPlans` (planner `nnUNetPlannerResEncM`): residual encoder U-Net, planned for ~8 GB GPU memory. Same 3d_fullres preprocessed data (ResEnc reuses `nnUNetPlans_3d_fullres`), same fold 0 = HERALD's 360 / 78 split, same trainer and 1,000 epochs |
| Code | commit `<hash>`; jobs `nnunet_train` / `nnunet_predict` with `NNUNET_PLANS=nnUNetResEncUNetMPlans` |
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

```bash
python common/evaluate.py summary results/tumor/per_case_val.csv --by-type --size-bins 10 50 200 --md
python common/evaluate.py compare ../v1_3d_fullres_fold0/results/tumor/per_case_val.csv results/tumor/per_case_val.csv --md
```

## Observations

## Decision
