# E03 · v4 — cross-validated evaluation of v0, v1 and v2 (438 patients)

| | |
|---|---|
| Status | planned (code ready, not submitted) |
| Compared with | v0 vs v1 and v0 vs v2, paired, on out-of-fold predictions |
| Change | the **evaluation**, not the models. `--cv_folds 5`: 5 folds over the 360 train + 78 val patients, stratified by type (`common.splits.cv_folds`, seed 0). Each fold trains on ~300 patients, early-stops on ~50 others from its own training part (an inner split, seed 1), and predicts its ~88 held-out patients. Every one of the 438 patients gets one out-of-fold prediction. Test (78) = the mean of the 5 fold models, reported only |
| Code | commit `<hash>`; `stage3_paper.py --cv_folds 5 --tag cv_v0` / `--pooling mask --tag cv_v1` / `--mil abmil --tag cv_v2` |
| Run | `$HERALD_RESULTS/unet_hybrid/logs/stage3_paper_<backbone>_cv_v{0,1,2}/`; ensembles `stage3_paper_ensemble_<members>/` |
| Date | 2026-10-03 |

## Why

E03 v1–v3 changed val accuracy by −0.03 to +0.01 on 78 patients, where one patient is 0.013 and the 95% CI is about ±0.10. v2 even moved in opposite directions on val and test. With 438 out-of-fold patients:
- one patient is worth 0.0023;
- the CI shrinks to about ±0.04;
- paired tests between versions get ~5.6× more cases.

Training data per model drops slightly (300 + 50 early stopping vs 360 + 78 for early stopping), so absolute accuracy may be a little lower than v0's.

**Note:** the 78 val patients are now used for training in 4 of 5 folds. That's fine for comparing versions, which is the point here. "Decide on val" becomes "decide on out-of-fold", and test stays untouched.

## How to reproduce

See [`run.sh`](run.sh): 18 jobs (3 configurations × 6 backbones, 5 folds each, ~10–15 min per job), then 3 ensemble jobs.

## Results

```bash
python common/evaluate.py compare <v0 run>/per_case_val.csv <v1 run>/per_case_val.csv --md
```

## Observations

## Decision
