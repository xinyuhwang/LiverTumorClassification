# E02 · v4 — per-case validation output for UNet-Hybrid

| | |
|---|---|
| Status | done (code); first real outputs come with the next UNet-Hybrid run or `--eval_only` |
| Compared with | v2 (UNet-Hybrid wrote per-case **test** output only) |
| Change | `unet_hybrid/train.py` also writes `per_case_val.csv` for Stages 1–3 |
| Code | `unet_hybrid/train.py`; commit: the one adding this README |
| Date | 2026-10-01 |

## Why

The rule from E01 is to decide on validation and report on test. `ds2net/` already writes per-case validation results; UNet-Hybrid didn't, so any choice about it could only be made on test. (v3, compute cost, is still planned; this version was done first because it was needed for E05.)

## What changed

| Stage | New file in `<log_dir>/stage<N>/` | Notes |
|---|---|---|
| 1 liver | `per_case_val.csv` | same columns as `per_case_test.csv`, including `Dice_vs_union` and `tumor_covered` |
| 2 tumor | `per_case_val.csv` | same protocol as test (GT liver crop) |
| 3 classification | `per_case_val.csv` | `true`, `pred`, `p_<class>` for the best-validation checkpoint |

- `metrics.json` (Stages 1–2) now has `val_per_case_mean / _std / _by_type` and `n_val_cases` next to the unchanged `test_*` keys.
- `evaluate_seg_test(..., val_ids=...)` evaluates validation first, then test. With `--eval_only`, both are written for an existing checkpoint.

## Validation

Smoke test of all 3 stages on synthetic data (CPU): each stage writes both files, and `common/evaluate.py summary` reads all three validation files.

## Note on running jobs

UNet-Hybrid Stage 2 (AICR job 1145658) started from the previous commit, so it writes test output only. Its validation output can be added afterwards without retraining:

```bash
bash cluster/submit.sh unet_hybrid --stage 2 --eval_only
```

## Decision

Adopted. Both pipelines now write per-case validation and test output.
