# E03 · v5 — four contrast phases as input

| | |
|---|---|
| Status | planned (code ready, not submitted) |
| Compared with | v1 config (mask pooling) under 5-fold OOF evaluation: `<backbone>_cv_v1` from v4, per backbone and six-model ensemble |
| Change | `--input 4phase`: crops are [arterial, PVP, delayed, non-contrast] at the tumor slice + tumor mask (5 channels) instead of the PVP triplet (slices s−1, s, s+1) + mask. Arterial / PVP / delayed get the pretrained RGB filters; non-contrast and the mask get their mean (UNet encoder: each phase's centre-slice filter from Stage 2). Same HU window for all phases. Everything else as v1 (`--pooling mask`), evaluated with `--cv_folds 5` |
| Code | commit `<hash>`; `stage3_paper.py --input 4phase --pooling mask --cv_folds 5 --tag cv_v5` |
| Date | 2026-10-03 |

## Hypothesis

Tumor types are told apart by how they enhance across phases:
- HCC: arterial enhancement, then washout;
- ICC: progressive delayed enhancement;
- HH: peripheral nodular filling;
- metastases: rim enhancement.

The paper models only see PVP. E03 v3 showed phase information shifts DS²Net's ICC vs HCC errors (8 → 4), but its weaker backbone gave no net gain. Expect gains mainly on HCC and ICC, possibly on CRLM vs BCLM.

**Trade-off:** this gives up the PVP triplet's ±1-slice context for phase information, so it changes the input composition, not only adds to it. That's one change of input design, compared on 438 OOF patients.

## How to reproduce

See [`run.sh`](run.sh): smoke test, 6 backbone jobs (~5 min each), 1 ensemble.

## Results

## Observations

## Decision
