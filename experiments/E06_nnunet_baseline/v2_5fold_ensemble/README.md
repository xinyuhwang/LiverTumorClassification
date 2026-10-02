# E06 · v2 — nnU-Net 5-fold ensemble

| | |
|---|---|
| Status | done: folds 1–5 = jobs 1159281–1159285 (5 h 11–15 min each, in parallel), ensemble prediction + scoring = job 1162587 (1 h 32 min); 2026-10-02 |
| Compared with | v1 (single model trained on all 360 train cases) |
| Change | ensemble of 5 models from 5-fold inner CV over the 360 train cases, instead of one model. Same plans, preprocessing, trainer and 1,000 epochs |
| Code | commit `070abed`; `prepare_dataset.py --cv_folds 5 --splits_only`; jobs `nnunet_train` (`NNUNET_FOLD=1..5`), `nnunet_predict_ensemble` |
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

All numbers from HERALD's metric code (per case, full volume, original grid); ensemble = softmax average of folds 1–5 with nnU-Net's mirroring.

### v2 vs v1 (B − A, paired, 95% bootstrap CI)

| | v1 (1 model, 360 cases) | v2 (5 × 288 cases) | Diff (95% CI) | Better / worse |
|---|---|---|---|---|
| **Val tumor Dice** | 0.796 | **0.807** | +0.012 (−0.015 to +0.048), Wilcoxon p 0.54 | 36 / 42 |
| Val tumor precision | 0.847 | 0.863 | +0.017 (−0.015 to +0.056), Wilcoxon p 0.028 | 44 / 34 |
| Val tumor recall | 0.800 | 0.788 | −0.012 (−0.024 to −0.003) | 25 / 52 |
| Val tumor global Dice | 0.895 | **0.903** (0.877–0.921) | | |
| Val liver Dice | 0.969 | 0.972 | +0.003 (+0.000 to +0.008) | 62 / 16 |
| Test tumor Dice (report) | 0.796 | 0.789 (0.748–0.828) | | |
| Test tumor global Dice | 0.883 | 0.883 (0.836–0.929) | | |
| Test liver Dice | 0.974 | 0.974 | | |

### Tumor Dice by total tumor volume (val, per-case mean)

| Group | n | v1 | v2 |
|---|---|---|---|
| < 10 ml | 16 | 0.653 | 0.656 |
| 10–50 ml | 20 | 0.765 | 0.811 |
| 50–200 ml | 16 | 0.792 | 0.788 |
| ≥ 200 ml | 26 | 0.909 | 0.910 |

Files: [`results/`](results/): `tumor/` and `liver/` per-case CSVs, summaries, `compare_val_vs_v1.json`.

## Observations

- **No clear tumor gain.** +0.012 on val with a CI spanning 0, and more cases worse (42) than better (36). Test is flat (0.789 vs 0.796). The ensemble is more conservative: precision up, recall down (significant, −0.012).
- **Global Dice crosses 0.90 on val (0.903), but that's within noise** of v1's 0.895, and test global Dice is unchanged (0.883).
- **Liver improves slightly but consistently** (+0.003, 62 of 78 cases better), from higher precision.
- **Small tumors (< 10 ml) didn't move** (0.653 → 0.656). The only group that rose is 10–50 ml (+0.046, not tested separately).
- **Interpretation:** five models trained on 288 cases each are about as good as one model on 360. Averaging removes some false positives but doesn't find missed small lesions. nnU-Net's default setup seems to plateau around 0.80 per case on this data.
- **Cost:** 5× training (~26 GPU-h) and 5× prediction time for no meaningful tumor gain.

## Decision (made on validation)

- **Not adopted as the main model on accuracy grounds:** the tumor gain is not significant, and test doesn't move. v1 stays the reference for comparisons; v2's held-out fold predictions remain available.
- **The plateau points to small tumors, not variance.** Next candidates for E06:
  - **v3: residual-encoder preset (`nnUNetResEncUNetMPlans`)**, single fold 0 vs v1. More capacity, the current recommended nnU-Net default.
  - Small-tumor work: higher in-plane resolution or a small-lesion-weighted loss/sampling; R1 failure-rate reporting to track it.
