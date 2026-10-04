# E03 · v6 — classification from predicted tumor masks (end to end)

| | |
|---|---|
| Status | done: jobs 1177684–1177689 (10–60 s each), ensemble 1177721; 2026-10-03 |
| Compared with | the same trained models scored on ground-truth masks: E03 v1 (`<backbone>_v1`, trained on the 360 train patients with GT masks, early-stopped on val) |
| Change | evaluation only. `--eval_only --mask_source pred`: val and test crops come from nnU-Net E06 v3's predicted tumor (label 2). The prediction decides which slices are used, the crop box and the mask channel. Models are not retrained. Patients with no predicted tumor would count as wrong (none expected: E06 v3 predicts tumor in all 156) |
| Code | commit `d2da6ec`; `stage3_paper.py --backbone <bb> --pooling mask --eval_only --ckpt_name <bb>_v1 --mask_source pred --pred_val_dir … --pred_test_dir … --tag v6pred` |
| Date | 2026-10-03 |

## Why

Every Stage 3 number so far used ground-truth tumor masks, an upper bound. A deployed pipeline has only nnU-Net's predicted mask. This measures the drop, on the 78 val and 78 test patients that nnU-Net did not train on. It's a paired comparison: same models, same patients, GT vs predicted masks. 5-fold OOF isn't possible here, since nnU-Net predictions exist only for val and test.

## Hypothesis

A small drop. nnU-Net finds 90% of lesions ≥ 1 cm and has tumor Dice ~0.81, and the classifier uses the largest tumor slices, which nnU-Net segments best (≥ 200 ml: 0.92). Expect the losses on patients with small or multifocal tumors, where a missed or extra lesion changes which slices are cropped.

## How to reproduce

See [`run.sh`](run.sh): 6 jobs (~1 min each) + 1 ensemble. The ground-truth-mask reference is E03 v1.

## Results

Same models (E03 v1, mask pooling), same patients; only the masks differ. No patient lost its tumor: nnU-Net predicts tumor in all 156.

| Model | Val accuracy, GT → predicted | Val AUC, GT → predicted | Test accuracy, GT → predicted | Test AUC, GT → predicted |
|---|---|---|---|---|
| EfficientNet-B3 | 0.641 → 0.615 | 0.866 → 0.861 | 0.641 → 0.628 | 0.923 → 0.914 |
| ViT-B/16 | 0.654 → 0.641 | 0.894 → 0.850 | 0.641 → 0.641 | 0.920 → 0.897 |
| Swin-Tiny | 0.679 → 0.628 | 0.892 → 0.873 | 0.538 → 0.590 | 0.885 → 0.882 |
| Swin-Base | 0.641 → 0.615 | 0.851 → 0.828 | 0.667 → 0.654 | 0.917 → 0.868 |
| ResNet-50 | 0.679 → 0.564 | 0.883 → 0.842 | 0.654 → 0.603 | 0.927 → 0.906 |
| UNet encoder | 0.641 → 0.577 | 0.860 → 0.858 | 0.564 → 0.538 | 0.877 → 0.867 |
| **Six-model ensemble** | **0.679 → 0.615** | **0.901 → 0.870** | **0.667 → 0.718** | **0.932 → 0.919** |

**Paired, ensemble:**
- Val: accuracy −0.064 (−0.141 to +0.013), McNemar p 0.18, 2 patients better / 7 worse; **AUC −0.030 (−0.055 to −0.008)**.
- Test: accuracy +0.051 (−0.013 to +0.128), p 0.29, 6 better / 2 worse; AUC −0.014 (−0.030 to +0.002).

Single models average −0.049 accuracy on val, −0.009 on test.

Files: [`results/`](results/), `compare_{val,test}_gt_vs_pred.json`.

## Observations

- **Predicted masks cost a little, consistently in AUC.** AUC drops for all 6 backbones on both val and test (−0.003 to −0.049), and significantly for the ensemble on val (−0.030).
- **Accuracy is noisy:** −0.064 on val, +0.051 on test for the ensemble, neither significant. On average across single models, about −0.05 on val and −0.01 on test.
- **The realistic end-to-end accuracy is about 0.62–0.72** (val–test, six-model ensemble, nnU-Net masks), against 0.67–0.68 with ground-truth masks: within a few points, as hypothesised.
- **Why so small:** the classifier uses the slices with the largest tumor area, which nnU-Net segments well. Mismatched slices or boxes come mostly from small or multifocal cases.
- Caveat: models were trained on ground-truth crops. Training on predicted masks (e.g. nnU-Net cross-validation predictions for train patients) could close part of the gap but would need nnU-Net predictions for all train cases.

## Decision

- **Report the end-to-end number alongside the ground-truth-mask number:** the pipeline loses about 0.03 AUC and a few points of accuracy without ground-truth masks. Not a version to adopt; it's the realistic measurement.
