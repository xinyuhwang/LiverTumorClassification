# E03 · v1 — mask-weighted pooling + tumor-area slice weighting (O1)

| | |
|---|---|
| Status | planned (code ready, not submitted) |
| Compared with | v0 (paper Stage 3), per backbone and six-model ensemble, on validation |
| Change | `--pooling mask`. Each crop's feature vector = the backbone's spatial features averaged with weights from the tumor-mask channel (downsampled to the feature grid), instead of global average pooling (CNNs, Swin) or the CLS token (ViT). A case's slice predictions are averaged weighted by tumor area instead of equally. Everything else as v0 |
| Source | O1, OrganLens (Ge et al., 2026): mask-weighted patch pooling + area-weighted slices raised AUROC 0.832 → 0.856 in their ablation |
| Code | commit `<hash>`; `stage3_paper.py --pooling mask --tag v1` |
| Run | `$HERALD_RESULTS/unet_hybrid/logs/stage3_paper_<backbone>_v1/`, ensemble `stage3_paper_ensemble_<members>/` |
| Date | 2026-10-03 |

## Hypothesis

The crop includes a 25% margin of liver around the tumor, and global pooling mixes that background into the tumor's feature vector. Pooling under the mask keeps tumor features separate. The mask channel stays in the input, so the surrounding context is still visible to the network. Weighting slices by tumor area lets the slices that show the tumor best count more. Expect a few points of accuracy or AUC, within v0's ±0.10 CIs, so likely not significant on 78 patients.

**Note:** this is the OrganLens proposal as one change (pooling + slice weighting), as their ablation tested them together.

## How to reproduce

See [`run.sh`](run.sh). Six backbone jobs (~1–3 min each) + one ensemble job.

## Results

## Observations

## Decision
