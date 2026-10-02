# E06 · nnU-Net 3D baseline — what's achievable on MCT-LTDiag

## Question

What liver and tumor segmentation accuracy can a strong, standard method reach on this dataset? The project goal is tumor Dice > 0.90, and nnU-Net is the reference method on liver-tumor benchmarks: it configures preprocessing, network and training itself. Its result tells us whether our models or the data/task are the limit.

## Setup

- **Data:** the prepared four-phase CT, all on the PVP grid (`data_prep/`). Labels: 0 background, 1 liver (official liver ∪ tumor, E01), 2 tumor.
- **Split:** HERALD's shared split as nnU-Net fold 0 (`splits_final.json`): 360 train / 78 val. The 78 test cases are predicted once at the end.
- **Code:**
  - `nnunet/prepare_dataset.py`: raw dataset with symlinked images, labels, and split
  - `nnunet/evaluate_predictions.py`: per-case tumor and liver metrics with HERALD's shared code, including `gt_ml` for size bins
  - jobs `cluster/jobs/nnunet_{prep,train,predict}.sbatch`
- **Metrics:** per-case mean Dice and **global Dice** (all voxels pooled), with 95% CIs, by tumor type and **by tumor size** (`common/evaluate.py summary --size-bins 10 50 200`). Liver and tumor are both reported. Decisions use validation; test reports.

## Reference points (test unless noted)

| Model | Tumor Dice per case | Tumor global Dice | Liver Dice |
|---|---|---|---|
| DS²Net Stage 2, cascade (E05 v1) | 0.726 (0.676–0.772); val 0.769 | 0.865 (0.824–0.908); val 0.893 | 0.973 (E01 v3) |
| UNet-Hybrid Stage 2, oracle liver (E05 v2) | 0.657; val 0.754 | val 0.858 | 0.971 (official label) |

## Versions

| Version | Change | Status |
|---|---|---|
| [v1](v1_3d_fullres_fold0/) | nnU-Net v2 `3d_fullres`, default trainer (1,000 epochs), fold 0 = HERALD split | Code ready; locally verified on synthetic data |
| v2 (if promising) | 5-fold ensemble | Planned |
