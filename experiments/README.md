# Experiments

Every change to HERALD is tried as an experiment, and every attempt within it as a numbered version. The plan and its priorities are in [`docs/ROADMAP.md`](../docs/ROADMAP.md).

## Layout

```text
experiments/
  README.md                  this index
  TEMPLATE.md                copy for each new version
  E01_liver_labels/
    README.md                question, metric, version table, conclusion
    v0_notebook_labels/      one folder per version
      README.md              what changed, exact command, results, decision
      run.sh                 the command(s) that produced the results
      results/               metrics.json, per_case_test.csv, … (pulled from AICR)
    v1_official_liver_masks/
```

## Rules

1. **One change per version.** A version differs from the one it's compared with in exactly one thing, so any change in results can be attributed to it.
2. **Code changes go behind flags, not copies.** New behaviour lives in `ds2net/`, `unet_hybrid/`, `common/` and is switched on with a flag or `--set key=value`. Old versions stay reproducible from the same code. The version folder holds the command, not a copy of the code.
3. **Record the commit.** Every job log prints `git commit: <hash>`; copy it into the version README.
4. **Results come from AICR**, via `cluster/pull_results.sh`, and only small files are tracked (JSON/CSV). Checkpoints stay on AICR under `$HERALD_STORE/results`.
5. **Finished versions are never edited.** If something was wrong, add a new version and say so.
6. **Decide on validation, report on test.** Choosing between versions (and tuning post-processing or thresholds) uses validation results only. Test is read once a choice is made. E01 broke this rule and re-checked both decisions on validation afterwards.
7. **Same evaluation for everyone.** Shared split (`common/mct_ltdiag_split.csv`), per-case test metrics, and bootstrap 95% CIs plus a paired comparison against the previous version from `common/evaluate.py` (see E02).

## Index

| Exp | Question | Versions | Status | Headline |
|---|---|---|---|---|
| [E01](E01_liver_labels/) | Are the liver labels correct, and how good is Stage 1 with correct ones? | v0–v3 | **Done**: v2 label + v3 post-processing adopted | Stage 1 liver Dice 0.950 → **0.973 (0.969–0.975)**; liver ∪ tumor label fixes hemangioma omission (HH `tumor_covered` 0.67 → 0.88) |
| [E02](E02_evaluation/) | Evaluation harness: CIs, paired tests, compute, failures and lesions | v1, v2, v4–v6 | v1, v2, v4, v5 done; v6 (R1) code done; v3 planned | `common/evaluate.py` CIs + paired tests, global Dice, size bins; v6 adds failure rate, per-size tests, per-lesion detection |
| [E03](E03_stage3_classification/) | Stage 3 tumor-type classification: baselines on the shared split, then pooling / aggregation (mask-weighted, ABMIL) | v0–v3 | v0 done; v1–v3 code ready | Six-model ensemble (PVP, GT masks): val acc 0.667 / AUC 0.894, test 0.692 / 0.920, reproduces the paper's 0.69. HH 100%; CRLM vs BCLM hardest; DS²Net 4-phase not better (PhaseNorm) |
| E04 | Joint segment + classify model | — | Planned | |
| [E05](E05_tumor_baselines/) | Tumor segmentation baselines (DS²Net, UNet-Hybrid) on the liver region | v1, v2 | v1–v2 done | DS²Net tumor 0.726 cascade / 0.740 oracle (primary). UNet-Hybrid tumor 0.657 oracle; on val comparable Dice (−0.018, n.s.) but less precise (−0.065). UNet-Hybrid liver 0.971 |
| [E06](E06_nnunet_baseline/) | nnU-Net 3D baseline: what's achievable on this dataset (goal: tumor Dice > 0.90) | v1–v3 | v1–v3 done; v3 adopted | Tumor ≈ 0.80 per case across all variants (default 0.796, 5-fold ensemble 0.807, ResEnc M 0.808 val; test 0.79–0.80). ResEnc M: global 0.913 val / 0.890 test, liver 0.973. ≥ 200 ml and HH 0.90+; < 10 ml ~0.65–0.70 |
| [E12](E12_ds2net_nnunet_recipe/) | Which parts of nnU-Net's recipe improve DS²Net Stage 2 (native slice spacing, normalisation, training length, loss, ensemble)? | v1–v4 | v1–v4 done | Only removing PhaseNorm helped (v2: val 0.780, +0.013 n.s.; recall +0.035, small tumors +0.041; test 0.735). Native 5 mm slices neutral; 100 epochs no gain; Dice + CE loss significantly worse (−0.030). Gap to nnU-Net ≈ 3D context |

## Segmentation Dice by tumor type (test set)

Current best version of each model on the shared test split (78 cases), mean ± SD of per-case full-volume Dice. Per-model text tables: [`ds2net/results/segmentation_by_tumor.txt`](../ds2net/results/segmentation_by_tumor.txt). The older [`unet_hybrid/results/segmentation_by_tumor.txt`](../unet_hybrid/results/segmentation_by_tumor.txt) came with the original code, before the shared split and bug fixes, so it isn't comparable with this table.

### Liver

| Type | N | DS²Net (E01 v3) | UNet-Hybrid (E05) | nnU-Net (E06 v1) |
|---|---|---|---|---|
| BCLM | 17 | 0.9739 ± 0.0096 | 0.9708 ± 0.0208 | 0.9732 ± 0.0159 |
| CRLM | 16 | 0.9776 ± 0.0048 | 0.9790 ± 0.0032 | 0.9797 ± 0.0027 |
| HCC | 16 | 0.9756 ± 0.0060 | 0.9759 ± 0.0066 | 0.9766 ± 0.0074 |
| HH | 14 | 0.9609 ± 0.0247 | 0.9523 ± 0.0383 | 0.9614 ± 0.0344 |
| ICC | 15 | 0.9739 ± 0.0099 | 0.9747 ± 0.0112 | 0.9750 ± 0.0119 |
| **Overall** | **78** | **0.9727 ± 0.0136** | **0.9710 ± 0.0213** | **0.9735 ± 0.0181** |

### Tumor

| Type | N | DS²Net cascade (E05 v1) | DS²Net oracle (E05 v1) | UNet-Hybrid oracle (E05 v2) | nnU-Net (E06 v1) |
|---|---|---|---|---|---|
| BCLM | 17 | 0.6112 ± 0.1921 | 0.6192 ± 0.2158 | 0.5479 ± 0.2283 | 0.7233 ± 0.1056 |
| CRLM | 16 | 0.7524 ± 0.1596 | 0.7504 ± 0.1623 | 0.6879 ± 0.2421 | 0.7838 ± 0.2048 |
| HCC | 16 | 0.6784 ± 0.2804 | 0.6720 ± 0.2894 | 0.6139 ± 0.3119 | 0.7560 ± 0.2532 |
| HH | 14 | 0.8125 ± 0.2429 | 0.8768 ± 0.0716 | 0.6660 ± 0.2758 | 0.9012 ± 0.0516 |
| ICC | 15 | 0.7972 ± 0.1538 | 0.8112 ± 0.1252 | 0.7835 ± 0.0941 | 0.8361 ± 0.1104 |
| **Overall** | **78** | **0.7258 ± 0.2193** | **0.7401 ± 0.2084** | **0.6567 ± 0.2490** | **0.7960 ± 0.1717** |

*Cascade* searches for the tumor inside the model's own predicted liver (realistic). *Oracle* uses the ground-truth liver box (an upper bound for the two-stage models). nnU-Net segments liver and tumor together in one 3D model. Decisions between versions are made on validation; these test numbers are for reporting.
