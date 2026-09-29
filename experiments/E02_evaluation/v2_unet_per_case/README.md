# E02 · v2 — per-case test output for UNet-Hybrid

| | |
|---|---|
| Status | done (code); first real outputs come with the next UNet-Hybrid runs on AICR |
| Compared with | v1 (statistics only; UNet-Hybrid had no per-case output) |
| Change | `unet_hybrid/train.py` evaluates every stage on the test set, one row per case, readable by `common/evaluate.py` |
| Code | `unet_hybrid/train.py`, `unet_hybrid/datasets.py`, new `common/metrics.py`; commit: the one adding this README |
| Date | 2026-09-29 |

## Why

Before this, UNet-Hybrid Stages 1 and 2 were only ever scored on validation, as batch-pooled slice Dice. The paper's test numbers (0.9704 / 0.6313) came from a script that isn't in the repository. Stage 3 printed test accuracy but no per-case predictions. Without per-case rows, UNet-Hybrid couldn't get confidence intervals or be compared case by case with DS²Net.

## What changed

| Stage | New test output (`<log_dir>/stage<N>/`) | Protocol |
|---|---|---|
| 1 liver | `per_case_test.csv` + `metrics.json`: Dice, IoU, precision, recall, `Dice_vs_union`, `tumor_covered` | every slice of the volume, whole slice (as trained), threshold 0.5, original NIfTI grid |
| 2 tumor | same, without the liver extras | slices with GT liver, inside each slice's GT liver bbox (the training protocol: **oracle liver**); 0 elsewhere |
| 3 classification | `per_case_test.csv`: `true`, `pred`, `p_<class>` | the model's own case-level prediction (centre slice + 12-slice volume stream) |

Also:
- **`--eval_only`** re-scores an existing best checkpoint on the test set without retraining, for every stage.
- **`common/metrics.py`** holds `volume_metrics` and `liver_extras`, now shared by `ds2net/` and `unet_hybrid/`, so per-case numbers are computed by the same code.
- **`datasets.py` refactor.** The 12-channel slice inputs are built by `liver_input()` / `tumor_input()`, which both the datasets and test prediction use. Verified to give identical arrays to before.

## Comparability with DS²Net

| | DS²Net | UNet-Hybrid | Directly comparable? |
|---|---|---|---|
| Stage 1 liver | full volume, 8-fold TTA, 3-D connected-component filter | full volume, no TTA, no filter | **Yes**: same cases and grid, same metric code. The protocol differences are part of each method. |
| Stage 2 tumor | full volume, every slice (no liver prior) | inside GT liver crops (oracle liver) | **No**: UNet-Hybrid gets the ground-truth liver. A fair comparison needs Stage 2 on predicted liver crops, which is a later version. |
| Stage 3 | GT tumor ROI patches (oracle) | whole-slice centre + volume stream (no tumor mask) | Comparable as case-level predictions, but the inputs differ |

## Validation

- Smoke test of all 3 stages on synthetic data (CPU): each writes `per_case_test.csv`, and `common/evaluate.py summary` reads all three.
- `--eval_only` reproduces the trained run's per-case numbers exactly.
- `pytest tests/`: 9 / 9 pass.

## Usage

```bash
bash cluster/submit.sh unet_hybrid --stage 1                  # trains, then writes stage1/per_case_test.csv
bash cluster/submit.sh unet_hybrid --stage 1 --eval_only      # re-score an existing checkpoint
python common/evaluate.py summary $HERALD_STORE/results/unet_hybrid/logs/stage1 --by-type --md
python common/evaluate.py compare <ds2net stage-1 run> $HERALD_STORE/results/unet_hybrid/logs/stage1 --md
```

## Decision

Adopted. Next versions:
- **E02 v3:** compute cost.
- **Separate follow-up:** UNet-Hybrid Stage 2 on predicted liver crops, for a fair tumor comparison.
