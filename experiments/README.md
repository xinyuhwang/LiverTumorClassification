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
| [E02](E02_evaluation/) | Evaluation harness: CIs, paired tests, compute | v1, v2 | v1–v2 done; v3 planned | `common/evaluate.py` CIs + paired tests; UNet-Hybrid now writes per-case test output |
| E03 | Stage 3 pooling / aggregation (mask-weighted, area-weighted, ABMIL) | — | Planned | |
| E04 | Joint segment + classify model | — | Planned | |
| [E05](E05_tumor_baselines/) | Tumor segmentation baselines (DS²Net, UNet-Hybrid) on the liver region | v1 | v1 running (AICR job 1139799) | — |
