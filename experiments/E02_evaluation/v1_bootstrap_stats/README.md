# E02 · v1 — bootstrap statistics

| | |
|---|---|
| Status | done |
| Change | new `common/evaluate.py` + `tests/test_evaluate.py` |
| Code | commit: the one adding these files |
| Date | 2026-09-29 |

## Goal

Turn any run's per-case test output into patient-level estimates with uncertainty, and compare two versions case by case.

## Method

| | Segmentation | Classification |
|---|---|---|
| Input | `per_case_test.csv`: `case_id`, `tumor_type`, Dice, IoU, Precision, Recall (+ `Dice_vs_union`, `tumor_covered` for liver) | `per_case_test.csv`: `case_id`, `true`, `pred`, `p_<class>`; or `stage3_paper_<backbone>_probs.json` |
| Point estimate | mean over patients | accuracy, macro-F1, macro one-vs-rest AUC |
| 95% CI | percentile bootstrap over patients (2,000 resamples, seed 0) | same, recomputing the metric on each resample |
| Paired test (B − A) | bootstrap CI and p of the mean difference; Wilcoxon signed-rank | bootstrap CI and p of the metric difference; exact McNemar on per-case correctness |
| Also reported | SD, median; per tumor type (`--by-type`); cases better / worse | per true class (`--by-type`); cases fixed / broken |

Only cases present in both runs are compared; the script reports any case found in only one of them.

## Validation

`pytest tests/test_evaluate.py`: 9 tests, all pass (40 s on a laptop).

| Test | Result |
|---|---|
| 95% CI coverage over 200 simulated test sets of 78 patients | within 88–99% (nominal 95%) |
| CI width vs normal theory (2 × 1.96 · SD/√n) | within 15% |
| Identical runs | diff 0, p = 1, 0 better / 0 worse |
| Constant +0.05 Dice on every case | diff 0.05, CI excludes 0, p < 0.01 |
| Runs with different case sets | compares only the common cases and lists the rest |
| Classification with 15 cases fixed, 0 broken | McNemar p < 0.001; accuracy diff 0.30 |
| `stage3_paper` JSON input, CLI with `--md` and `--out` | loaded and written correctly |

Also run on the real per-case output of the AICR smoke test (job 1119353, 2 cases). The mechanics work; with 2 cases the intervals mean nothing.

## Usage

```bash
python common/evaluate.py summary experiments/E01_liver_labels/v1_official_liver_masks/results --by-type --md
python common/evaluate.py compare <run A> <run B> --metrics Dice Dice_vs_union --md --out compare.json
```

## Decision

Adopted: every experiment version from E01 v1 on reports `summary` and `compare` tables from this script.
