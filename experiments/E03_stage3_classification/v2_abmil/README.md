# E03 · v2 — attention-based MIL over a case's slices (G1)

| | |
|---|---|
| Status | done: jobs 1177115–1177125 (odd = v2), ensemble 1177246; 2–3 min each; 2026-10-03 |
| Compared with | v0 (paper Stage 3), per backbone and six-model ensemble, on validation. Built on v0, not v1, so the two run in parallel |
| Change | `--mil abmil`. Training is per case instead of per crop: each case's crops (≤ 8 slices) → backbone embeddings (paper pooling) → gated attention pooling (Ilse et al. 2018, hidden 128) → the same head → one prediction per case. 4 cases per batch. Prediction: the attention-pooled bag, with the same 4-view TTA. Everything else as v0 |
| Source | G1, GigaPath-Flash (Usuyama et al., 2026): ABMIL aggregation of tile embeddings into one slide label |
| Code | commit `3f07b35`; `stage3_paper.py --mil abmil --tag v2` |
| Run | `$HERALD_RESULTS/unet_hybrid/logs/stage3_paper_<backbone>_v2/`, ensemble `stage3_paper_ensemble_<members>/` |
| Date | 2026-10-03 |

## Hypothesis

v0 trains every slice with the case label, even slices where the tumor is a sliver at its edge, and then averages slices equally. ABMIL trains on the case label directly and learns which slices to trust. Expect gains on cases whose slices disagree.

**Risk:** 360 training bags instead of ~2,900 crops, so fewer, noisier gradient steps. The early stopping on val accuracy still applies.

## How to reproduce

See [`run.sh`](run.sh). Six backbone jobs + one ensemble job.

## Results

| Model | v0 val acc / AUC | v2 val acc / AUC | v0 test acc / AUC | v2 test acc / AUC |
|---|---|---|---|---|
| EfficientNet-B3 | 0.590 / 0.855 | 0.577 / 0.842 | 0.615 / 0.884 | 0.615 / 0.857 |
| ViT-B/16 | 0.705 / 0.888 | 0.692 / 0.892 | 0.679 / 0.896 | 0.705 / 0.919 |
| Swin-Tiny | 0.641 / 0.850 | 0.667 / 0.862 | 0.679 / 0.889 | 0.679 / 0.919 |
| Swin-Base | 0.577 / 0.863 | 0.590 / 0.827 | 0.654 / 0.915 | 0.654 / 0.900 |
| ResNet-50 | 0.641 / 0.875 | 0.628 / 0.862 | 0.667 / 0.913 | 0.628 / 0.926 |
| UNet encoder | 0.641 / 0.840 | 0.526 / 0.787 | 0.526 / 0.869 | 0.538 / 0.826 |
| **Six-model ensemble** | 0.667 / 0.894 | 0.641 / 0.895 | 0.692 / 0.921 | 0.756 / 0.942 |

**Paired on val (six-model ensembles, v2 − v0):** accuracy −0.026 (−0.090 to +0.026), McNemar p 0.69, 2 better / 4 worse; macro-F1 −0.030; macro-AUC +0.001.

Files: [`results/`](results/), `compare_val_vs_v0.json`.

## Observations

- **Slightly worse on val, clearly better on test:** ensemble val 0.641 vs 0.667, test **0.756** vs 0.692 (the highest test accuracy in E03, AUC 0.942). Single backbones mostly lose on val; the UNet encoder (fully frozen, only the attention and head train) drops most (0.526).
- **That val/test split is the main lesson:** a −0.03 / +0.06 swing on the same change shows how noisy 78-patient accuracy is. Neither number is significant.
- Training on 360 bags gives ~90 steps per epoch instead of ~180; early stopping came at similar epochs.

## Decision (made on validation)

- **Not adopted:** val doesn't improve, and choosing it for its test score would break the decide-on-val rule. Kept as an option (`--mil abmil`).
