# E06 · v2 — nnU-Net 5-fold ensemble

| | |
|---|---|
| Status | planned (scripts ready, not submitted) |
| Compared with | v1 (single model trained on all 360 train cases) |
| Change | ensemble of 5 models from 5-fold inner CV over the 360 train cases, instead of one model. Same plans, preprocessing, trainer and 1,000 epochs |
| Code | commit `<hash>`; `prepare_dataset.py --cv_folds 5 --splits_only`; jobs `nnunet_train` (`NNUNET_FOLD=1..5`), `nnunet_predict_ensemble` |
| Run | `$nnUNet_results/Dataset501_MCTLTDiag/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_{1..5}`; scores in `$HERALD_RESULTS/runs/nnunet/e06_v2_5fold_ensemble/{tumor,liver}/per_case_{val,test}.csv` |
| Date | 2026-10-02 |

## Design

The usual nnU-Net protocol runs 5-fold CV over all labelled cases. Here that would put HERALD's 78 val cases into training, leaving no held-out val to compare v2 with v1. So the folds are **inner folds over the 360 train cases only**:

- **Folds:** `splits_final.json` keeps v1's fold 0 (360 / 78) and adds folds 1–5 (288 train / 72 held out each), stratified by tumor type (`common.splits.cv_folds`, seed 0). The v1 model and its results are untouched.
- **Val and test stay out of every fold.** The ensemble predicts both sets: val for the decision, test for the report.
- **Each fold model sees 288 cases**, 80% of v1's 360. That's part of the comparison, and it's what we'd deploy: does averaging 5 models beat one model trained on everything?
- **Free by-product:** each fold writes held-out predictions for its 72 train cases (`fold_N/validation/`), giving a cross-validated estimate over all 360 train cases. These aren't used for the decision.

## Hypothesis

The ensemble improves tumor Dice by about 1–3 points per case, mostly through fewer false positives and fewer missed small lesions, where single models disagree most. Liver is already at 0.97 and should barely change. A per-case mean of 0.90 is still not expected (small tumors).

## Steps and cost

See [`run.sh`](run.sh).

1. Pull the code on AICR (after you push), then write the new splits: `prepare_dataset.py --cv_folds 5 --splits_only` (seconds; no re-preprocessing, the 44 GB preprocessed data is reused).
2. Train folds 1–5: five jobs of ~5.5 h each on 1 GPU (~27 GPU-hours). They run in parallel if the queue allows.
3. `nnunet_predict_ensemble`: ensemble prediction of 78 val + 78 test cases and scoring (~1–2 h; 5 models × mirroring per case).

`/scratch` purges files older than 30 days. The preprocessed data was written on 2026-10-01; if it has been purged, rerun `nnunet_prep` first (~10 min).

## Results

Paste the output of:

```bash
python common/evaluate.py summary results/tumor/per_case_val.csv --by-type --size-bins 10 50 200 --md
python common/evaluate.py compare ../v1_3d_fullres_fold0/results/tumor/per_case_val.csv results/tumor/per_case_val.csv --md
```

## Observations

## Decision
