# E03 · v6 — classification from predicted tumor masks (end to end)

| | |
|---|---|
| Status | planned (code ready, not submitted) |
| Compared with | the same trained models scored on ground-truth masks: E03 v1 (`<backbone>_v1`, trained on the 360 train patients with GT masks, early-stopped on val) |
| Change | evaluation only. `--eval_only --mask_source pred`: val and test crops come from nnU-Net E06 v3's predicted tumor (label 2). The prediction decides which slices are used, the crop box and the mask channel. Models are not retrained. Patients with no predicted tumor would count as wrong (none expected: E06 v3 predicts tumor in all 156) |
| Code | commit `<hash>`; `stage3_paper.py --backbone <bb> --pooling mask --eval_only --ckpt_name <bb>_v1 --mask_source pred --pred_val_dir … --pred_test_dir … --tag v6pred` |
| Date | 2026-10-03 |

## Why

Every Stage 3 number so far used ground-truth tumor masks, an upper bound. A deployed pipeline has only nnU-Net's predicted mask. This measures the drop, on the 78 val and 78 test patients that nnU-Net did not train on. It's a paired comparison: same models, same patients, GT vs predicted masks. 5-fold OOF isn't possible here, since nnU-Net predictions exist only for val and test.

## Hypothesis

A small drop. nnU-Net finds 90% of lesions ≥ 1 cm and has tumor Dice ~0.81, and the classifier uses the largest tumor slices, which nnU-Net segments best (≥ 200 ml: 0.92). Expect the losses on patients with small or multifocal tumors, where a missed or extra lesion changes which slices are cropped.

## How to reproduce

See [`run.sh`](run.sh): 6 jobs (~1 min each) + 1 ensemble. The ground-truth-mask reference is E03 v1.

## Results

## Observations

## Decision
