# E03 · v3 — DS²Net 4-phase Stage 3 without PhaseNorm

| | |
|---|---|
| Status | planned (code ready, not submitted) |
| Compared with | v0's DS²Net Stage 3 (`e03_v0_ds2net_cls`), on validation |
| Change | `phase_norm=fixed`: replace PhaseNorm with `FixedPhaseScale` (as in E12 v2). As a direct consequence, the LI-RADS phase gate and the enhancement-curve features see real intensities: the curve's patch means now come from the windowed input, on the same scale as the liver-ring background means. Everything else as v0 |
| Code | commit `<hash>`; `ds2net/train.py --stage 3 --set phase_norm=fixed` |
| Run | `$HERALD_RESULTS/runs/ds2net/e03_v3_ds2net_fixed_norm` |
| Date | 2026-10-03 |

## Hypothesis

Tumor types differ by how they enhance across the four phases (e.g. HCC: arterial enhancement then washout; ICC: progressive delayed enhancement). PhaseNorm normalises every phase of every patch to the same mean and spread, so v0's 4-phase classifier couldn't use that, and its main confusion was ICC → HCC (8 of 15). In v0, the curve features and the phase gate were effectively constant. With fixed scaling, expect DS²Net's classifier to move from 0.60 toward or past the PVP-only models (0.64–0.71 val), with the largest gain on ICC vs HCC.

## How to reproduce

See [`run.sh`](run.sh): one job, ~3–5 min.

## Results

## Observations

## Decision
