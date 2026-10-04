# E03 · v3 — DS²Net 4-phase Stage 3 without PhaseNorm

| | |
|---|---|
| Status | done: smoke test 1177108, run 1177126 (2 min 26 s), 2026-10-03 |
| Compared with | v0's DS²Net Stage 3 (`e03_v0_ds2net_cls`), on validation |
| Change | `phase_norm=fixed`: replace PhaseNorm with `FixedPhaseScale` (as in E12 v2). As a direct consequence, the LI-RADS phase gate and the enhancement-curve features see real intensities: the curve's patch means now come from the windowed input, on the same scale as the liver-ring background means. Everything else as v0 |
| Code | commit `3f07b35`; `ds2net/train.py --stage 3 --set phase_norm=fixed` |
| Run | `$HERALD_RESULTS/runs/ds2net/e03_v3_ds2net_fixed_norm` |
| Date | 2026-10-03 |

## Hypothesis

Tumor types differ by how they enhance across the four phases (e.g. HCC: arterial enhancement then washout; ICC: progressive delayed enhancement). PhaseNorm normalises every phase of every patch to the same mean and spread, so v0's 4-phase classifier couldn't use that, and its main confusion was ICC → HCC (8 of 15). In v0, the curve features and the phase gate were effectively constant. With fixed scaling, expect DS²Net's classifier to move from 0.60 toward or past the PVP-only models (0.64–0.71 val), with the largest gain on ICC vs HCC.

## How to reproduce

See [`run.sh`](run.sh): one job, ~3–5 min.

## Results

| | v0 (PhaseNorm) | v3 (fixed) |
|---|---|---|
| Val accuracy / macro-F1 / macro-AUC | 0.603 / 0.595 / 0.839 | 0.603 / 0.597 / 0.832 |
| Test accuracy / macro-F1 / macro-AUC | 0.487 / 0.506 / 0.761 | 0.474 / 0.492 / 0.774 |
| Val recall BCLM / CRLM / HCC / HH / ICC | 0.50 / 0.47 / 0.67 / 1.00 / 0.40 | 0.39 / 0.53 / 0.47 / 1.00 / **0.67** |
| ICC called HCC (val, of 15) | 8 | **4** |

**Paired on val:** accuracy ±0.000 (−0.077 to +0.077), McNemar p 1.0, 5 better / 5 worse; macro-AUC −0.008.

## Observations

- **No net change,** but the errors moved where the hypothesis said: ICC → HCC confusions halved (8 → 4) and ICC recall rose 0.40 → 0.67. The classifier now uses enhancement differences.
- **The gain was paid back elsewhere:** HCC recall fell 0.67 → 0.47 and BCLM 0.50 → 0.39, so ICC vs HCC is now split differently rather than better.
- DS²Net's Stage 3 stays below the PVP-only paper models (val 0.60 vs 0.58–0.71; test 0.47 vs 0.53–0.69). Its small EfficientNet-B0 and 16-patch, curve-feature design may matter more than normalisation.

## Decision (made on validation)

- **Not adopted on accuracy** (no change). The shifted confusion pattern is worth keeping in mind: phase information helps ICC; a multi-phase input to the stronger paper backbones may be the better route.
