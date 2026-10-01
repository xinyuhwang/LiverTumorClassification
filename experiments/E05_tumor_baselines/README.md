# E05 · Tumor segmentation baselines

## Question

How well do DS²Net and UNet-Hybrid segment tumors under the new pipeline: shared split, correct labels, per-case metrics, decisions on validation? And how much is lost when the tumor search region comes from predicted rather than true liver?

## Background

No valid tumor numbers existed under the new pipeline:
- The DS²Net notebook reported 0.685 slice-level / 0.375 per case, with its own split and bugs.
- The paper's UNet-Hybrid 0.631 used the true liver, a single phase, and an evaluation script that isn't in the repository.

Tumors grow in the liver, so Stage 2 searches only a box around the liver (`roi="liver"`, `ds2net/PORTING.md`).

## Metric

- **Primary:** per-case tumor Dice, full volume, 8-fold TTA, original grid.
- **Reported in two modes:**
  - **cascade:** liver box from E01's adopted Stage 1 (v2 model + largest component); the real pipeline.
  - **oracle:** box from the true liver + tumor mask; isolates the tumor model.
- **Secondary:** IoU, precision, recall, per tumor type.
- Versions are chosen on validation; test reports.

## Versions

| Version | Model / change | Status | Test tumor Dice (cascade / oracle) |
|---|---|---|---|
| [v1](v1_ds2net_tumor_roi/) | DS²Net Stage 2, liver ROI, notebook hyperparameters | Done | **0.726** (0.676–0.772) / 0.740 (0.692–0.784) |
| v2 | UNet-Hybrid Stage 2 (Stage 1 done: [`unet_stage1_liver/`](unet_stage1_liver/), liver Dice 0.971) | Next | — |
