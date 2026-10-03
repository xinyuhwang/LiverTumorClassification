# E03 · v2 — attention-based MIL over a case's slices (G1)

| | |
|---|---|
| Status | planned (code ready, not submitted) |
| Compared with | v0 (paper Stage 3), per backbone and six-model ensemble, on validation. Built on v0, not v1, so the two run in parallel |
| Change | `--mil abmil`. Training is per case instead of per crop: each case's crops (≤ 8 slices) → backbone embeddings (paper pooling) → gated attention pooling (Ilse et al. 2018, hidden 128) → the same head → one prediction per case. 4 cases per batch. Prediction: the attention-pooled bag, with the same 4-view TTA. Everything else as v0 |
| Source | G1, GigaPath-Flash (Usuyama et al., 2026): ABMIL aggregation of tile embeddings into one slide label |
| Code | commit `<hash>`; `stage3_paper.py --mil abmil --tag v2` |
| Run | `$HERALD_RESULTS/unet_hybrid/logs/stage3_paper_<backbone>_v2/`, ensemble `stage3_paper_ensemble_<members>/` |
| Date | 2026-10-03 |

## Hypothesis

v0 trains every slice with the case label, even slices where the tumor is a sliver at its edge, and then averages slices equally. ABMIL trains on the case label directly and learns which slices to trust. Expect gains on cases whose slices disagree.

**Risk:** 360 training bags instead of ~2,900 crops, so fewer, noisier gradient steps. The early stopping on val accuracy still applies.

## How to reproduce

See [`run.sh`](run.sh). Six backbone jobs + one ensemble job.

## Results

## Observations

## Decision
