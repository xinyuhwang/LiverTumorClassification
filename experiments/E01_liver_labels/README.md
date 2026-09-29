# E01 · Liver labels

**Roadmap item:** O3 (tier 1, impact 5, cost 1).

## Question

Are the liver labels that Stage 1 trains on actually liver? And what does Stage 1 achieve once they are?

## Background

The DS²Net notebooks built both labels from `mask_pvp.nii.gz`: `liver = mask ≥ 1`, and `tumor = mask == 2` if label 2 exists, else `mask == 1`. OrganLens suggested using TotalSegmentator to make liver masks. Inspecting the dataset showed that MCT-LTDiag (v3.1) already ships an official `liver_mask_pvp.nii.gz` for each case.

## Metric

- **Primary:** per-case liver Dice on the 78 test cases. Full volume, 8-fold TTA, original NIfTI grid.
- **Secondary:** IoU, precision, recall; per-tumor-type Dice.
- **Data QA:** number of flagged cases; tumor-inside-liver fraction; liver volume range.

## Versions

| Version | Change | Status | Test liver Dice (per case) |
|---|---|---|---|
| [v0](v0_notebook_labels/) | Notebook labels: liver = `mask_pvp ≥ 1` | Done (documented, not rerun) | Not measurable: the label was the tumor |
| [v1](v1_official_liver_masks/) | Official `liver_mask_pvp.nii.gz` via `data_prep/` | Ready to run on AICR | — |
| v2 (only if needed) | TotalSegmentator cross-check of flagged cases | Conditional on v1 QA | — |

## Conclusion so far

v0's "liver" label was the tumor mask, so no DS²Net liver result so far is valid (see [v0](v0_notebook_labels/)). v1 fixes the labels at the data level, for both pipelines. It still needs to be run.
