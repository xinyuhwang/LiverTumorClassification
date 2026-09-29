# LiverTumorClassification

HERALD (Hepatic Evaluation and Reporting with AI-Driven Diagnostics): liver segmentation, tumor segmentation and tumor subtype classification (BCLM, CRLM, HCC, HH, ICC) on the MCT-LTDiag multi-phase CT dataset.

## Layout

```text
ds2net/                              DS²Net pipeline (Google Colab notebooks)
  liver_segmentation.ipynb           Stage 1: liver segmentation, Swin-Tiny + MHA
  segmentation_classification.ipynb  Liver/tumor segmentation + EfficientNet-B0 classifier
unet_hybrid/                         UNet-Hybrid pipeline (SLURM / HPC scripts)
  datasets.py                        Volume loading and per-stage datasets
  models.py                          UNetTransformer, classifiers, losses
  train.py                           Entry point for all three stages (original Stage 3)
  stage3_paper.py                    Stage 3 as described in the paper
  STAGE3_DESIGN.md                   How the two Stage 3 implementations differ
  mct_ltdiag_split.csv               Train/val/test split (stratified by type and age bin)
  results/                           Training logs and test-set Dice by tumor type
```

## UNet-Hybrid usage

```bash
cd unet_hybrid
python train.py --stage 1   # liver segmentation
python train.py --stage 2   # tumor segmentation (warm-started from stage 1)
python train.py --stage 3   # tumor classification
python train.py --stage 0   # all stages

python stage3_paper.py --backbone efficientnet_b3   # paper-version Stage 3
python stage3_paper.py --mode ensemble              # after training each backbone
```

See [unet_hybrid/STAGE3_DESIGN.md](unet_hybrid/STAGE3_DESIGN.md) for the two Stage 3 implementations.

`--data_dir` must contain one folder per case with `pvp.nii.gz`, `liver_mask.nii.gz`, `tumor_mask.nii.gz` and `phase_0..3.nii.gz` (non-contrast, arterial, portal venous, delayed). Add `--smoke_test` for a 6-case, 2-epoch run.
