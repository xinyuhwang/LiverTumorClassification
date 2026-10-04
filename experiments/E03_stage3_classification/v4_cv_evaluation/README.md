# E03 · v4 — cross-validated evaluation of v0, v1 and v2 (438 patients)

| | |
|---|---|
| Status | done: smoke 1177318; 18 jobs 1177320–1177337 (2–11 min each); ensembles 1177475–1177477; 2026-10-03 |
| Compared with | v0 vs v1 and v0 vs v2, paired, on out-of-fold predictions |
| Change | the **evaluation**, not the models. `--cv_folds 5`: 5 folds over the 360 train + 78 val patients, stratified by type (`common.splits.cv_folds`, seed 0). Each fold trains on ~300 patients, early-stops on ~50 others from its own training part (an inner split, seed 1), and predicts its ~88 held-out patients. Every one of the 438 patients gets one out-of-fold prediction. Test (78) = the mean of the 5 fold models, reported only |
| Code | commit `0079c89`; `stage3_paper.py --cv_folds 5 --tag cv_v0` / `--pooling mask --tag cv_v1` / `--mil abmil --tag cv_v2` |
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

Out-of-fold (OOF) = 438 train + val patients, each predicted by a fold model that never saw it. Paired tests: bootstrap (1,000 resamples for ensembles, 300 per backbone) and McNemar for accuracy.

### Six-model ensembles

| | OOF accuracy (95% CI) | OOF macro-F1 | OOF macro-AUC | Test accuracy (78) |
|---|---|---|---|---|
| v0 paper | 0.605 (0.557–0.651) | 0.608 | 0.870 | 0.692 |
| **v1 mask pooling** | **0.626** (0.582–0.669) | **0.631** | **0.878** | 0.744 |
| v2 ABMIL | 0.628 (0.582–0.669) | 0.630 | 0.871 | 0.769 |

Paired vs v0: v1 accuracy +0.021 (−0.018 to +0.059), McNemar p 0.35, 41 better / 32 worse; AUC +0.008 (−0.002 to +0.017), p 0.12. v2 accuracy +0.023 (−0.016 to +0.059), p 0.30, 43 / 33; AUC +0.001.

**v2 vs v1 directly:** accuracy +0.002 (−0.034 to +0.042), McNemar p 1.0, 45 / 44; AUC −0.007 (−0.018 to +0.005). **The v1 and v2 ensembles are indistinguishable.**

### Per backbone, OOF (difference vs v0, 95% CI)

| Backbone | v0 acc | v1 acc | v1 − v0 accuracy | v1 − v0 AUC | v2 acc | v2 − v0 accuracy | v2 − v0 AUC |
|---|---|---|---|---|---|---|---|
| EfficientNet-B3 | 0.543 | 0.580 | +0.036 (−0.011, +0.081) | **+0.025 (+0.009, +0.040)** | 0.516 | −0.027 | **−0.031** |
| ViT-B/16 | 0.612 | 0.603 | −0.009 | −0.010 | 0.598 | −0.014 | −0.011 |
| Swin-Tiny | 0.575 | 0.598 | +0.023 | +0.008 | 0.582 | +0.007 | +0.011 |
| Swin-Base | 0.521 | 0.555 | +0.034 (−0.011, +0.078) | +0.008 | 0.521 | 0.000 | **−0.030** |
| ResNet-50 | 0.575 | 0.619 | **+0.043 (+0.001, +0.081)**, p 0.048 | **+0.023** | 0.566 | −0.009 | −0.004 |
| UNet encoder | 0.470 | 0.571 | **+0.101 (+0.052, +0.144)**, p < 0.001 | **+0.031** | 0.420 | **−0.050**, p 0.04 | **−0.072** |

Bold = 95% CI excludes 0.

Files: [`results/`](results/): per backbone and configuration `stage3_paper_<bb>_cv_v{0,1,2}/per_case_{val,test}.csv` (val = OOF) and `_probs.json` (with per-fold accuracies); ensembles; `compare_ensemble_*.json`.

## Observations

- **The evaluation does its job:** CIs are about ±0.045 (78 patients: ±0.10), and per-backbone effects now separate from noise.
- **Mask pooling (v1) helps consistently:**
  - Accuracy rises for 5 of 6 backbones and AUC for 5 of 6.
  - The gain is significant for ResNet-50 (accuracy and AUC), EfficientNet-B3 (AUC) and the UNet encoder (+0.10 accuracy). The UNet encoder is fully frozen, so how its features are pooled matters most there.
  - ViT is the exception (−0.009, n.s.); its CLS token already attends selectively.
  - **Ensemble:** +0.021 accuracy, +0.008 AUC, not significant: averaging six models already smooths out some of the benefit.
- **ABMIL (v2) is mixed:** the ensemble gains a similar +0.023 (n.s.), but per backbone it's flat or worse, with significant losses in AUC for 3 backbones and in accuracy for the UNet encoder. The ensemble gain likely comes from more diverse members, not from better models.
- **Absolute accuracy is lower than on the 78-patient val split** (0.605 vs 0.667 for v0): fewer training patients per model, and the 78-patient val split was probably on the easy side.
- **Test (reported only)** agrees in direction: 0.692 → 0.744 (v1), 0.769 (v2).
- **At the ensemble level, v0, v1 and v2 can't be told apart** (all pairwise differences n.s.; v2 − v1 = +0.002). The difference between v1 and v2 shows only for single backbones: v1 helps or is neutral for every backbone, v2 hurts several.
- **Multiple comparisons:** 12 per-backbone tests per metric, so one nominally significant result could be chance. ResNet-50's accuracy gain (p 0.048) is borderline. The UNet encoder's gain (p < 0.001) and v1's consistent direction are the firmer evidence.

## Decision (made on out-of-fold, 438 patients)

- **Use mask-weighted pooling (v1, `--pooling mask`) as the Stage 3 default.** A judgement call: it improves or leaves unchanged every single backbone, costs nothing, and suits predicted masks later. **It is not a demonstrated ensemble improvement:** the six-model ensemble is n.s. vs v0 and tied with v2.
- **Keep ABMIL off:** it reaches the same ensemble accuracy, but its single models are worse, so it adds nothing over v1.
- **Use 5-fold OOF evaluation for future Stage 3 versions.** Each run costs minutes.
