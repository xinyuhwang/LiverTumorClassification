# E03 · v1 — mask-weighted pooling + tumor-area slice weighting (O1)

| | |
|---|---|
| Status | done: jobs 1177114–1177124 (even = v1), ensemble 1177245; 1–2 min each; 2026-10-03 |
| Compared with | v0 (paper Stage 3), per backbone and six-model ensemble, on validation |
| Change | `--pooling mask`. Each crop's feature vector = the backbone's spatial features averaged with weights from the tumor-mask channel (downsampled to the feature grid), instead of global average pooling (CNNs, Swin) or the CLS token (ViT). A case's slice predictions are averaged weighted by tumor area instead of equally. Everything else as v0 |
| Source | O1, OrganLens (Ge et al., 2026): mask-weighted patch pooling + area-weighted slices raised AUROC 0.832 → 0.856 in their ablation |
| Code | commit `3f07b35`; `stage3_paper.py --pooling mask --tag v1` |
| Run | `$HERALD_RESULTS/unet_hybrid/logs/stage3_paper_<backbone>_v1/`, ensemble `stage3_paper_ensemble_<members>/` |
| Date | 2026-10-03 |

## Hypothesis

The crop includes a 25% margin of liver around the tumor, and global pooling mixes that background into the tumor's feature vector. Pooling under the mask keeps tumor features separate. The mask channel stays in the input, so the surrounding context is still visible to the network. Weighting slices by tumor area lets the slices that show the tumor best count more. Expect a few points of accuracy or AUC, within v0's ±0.10 CIs, so likely not significant on 78 patients.

**Note:** this is the OrganLens proposal as one change (pooling + slice weighting), as their ablation tested them together.

## How to reproduce

See [`run.sh`](run.sh). Six backbone jobs (~1–3 min each) + one ensemble job.

## Results

Patient level, 78 val / 78 test patients.

| Model | v0 val acc / AUC | v1 val acc / AUC | v0 test acc / AUC | v1 test acc / AUC |
|---|---|---|---|---|
| EfficientNet-B3 | 0.590 / 0.855 | 0.641 / 0.866 | 0.615 / 0.884 | 0.641 / 0.923 |
| ViT-B/16 | 0.705 / 0.888 | 0.654 / 0.894 | 0.679 / 0.896 | 0.641 / 0.920 |
| Swin-Tiny | 0.641 / 0.850 | 0.679 / 0.892 | 0.679 / 0.889 | 0.538 / 0.885 |
| Swin-Base | 0.577 / 0.863 | 0.641 / 0.851 | 0.654 / 0.915 | 0.667 / 0.917 |
| ResNet-50 | 0.641 / 0.875 | 0.679 / 0.883 | 0.667 / 0.913 | 0.654 / 0.927 |
| UNet encoder | 0.641 / 0.840 | 0.641 / 0.860 | 0.526 / 0.869 | 0.564 / 0.877 |
| **Six-model ensemble** | 0.667 / 0.894 | 0.679 / 0.901 | 0.692 / 0.921 | 0.667 / 0.932 |

**Paired on val (six-model ensembles, v1 − v0):** accuracy +0.013 (−0.064 to +0.090), McNemar p 1.0, 5 better / 4 worse; macro-F1 +0.011; macro-AUC +0.007 (−0.017 to +0.029).

Files: [`results/`](results/), `compare_val_vs_v0.json`.

## Observations

- **A small, consistent shift on val, nowhere near significant.** Val accuracy rose for 4 of 6 backbones (EfficientNet +0.05, Swin-Base +0.06, Swin-Tiny and ResNet +0.04), fell for ViT (−0.05), and val AUC rose for 5 of 6. The ensemble moved +0.013 accuracy, +0.007 AUC.
- **Test doesn't confirm it:** ensemble test accuracy 0.667 vs 0.692 (AUC 0.932 vs 0.921). Swin-Tiny test dropped to 0.538.
- OrganLens's gain (+0.024 AUROC) would be within our CIs anyway: on 78 patients a single class decision moves accuracy by 0.013.

## Decision (made on validation)

- **Not adopted as the new baseline:** the val gain is not significant and test goes the other way. Kept as an option (`--pooling mask`), the more natural choice once Stage 3 runs on predicted masks.
