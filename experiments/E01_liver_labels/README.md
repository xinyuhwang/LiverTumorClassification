# E01 · Liver labels

**Roadmap item:** O3 (tier 1, impact 5, cost 1).

## Question

Are the liver labels that Stage 1 trains on actually liver? And what does Stage 1 achieve once they are?

## Background

The DS²Net notebooks built both labels from `mask_pvp.nii.gz`: `liver = mask ≥ 1`, and `tumor = mask == 2` if label 2 exists, else `mask == 1`. OrganLens suggested using TotalSegmentator to make liver masks. Inspecting the dataset showed that MCT-LTDiag (v3.1) already ships an official `liver_mask_pvp.nii.gz` for each case.

### The official liver mask often leaves out part of the tumor

This comes from Yu Zhang's full-dataset audit (branch `setup/data-and-env`, commit `1c1742c`, run `p01-r04-full-dataset-audit`, 2026-07-03), which covers all 517 cases. The audit code describes the liver mask as an "automatic liver pseudo-label".

| Tumor fraction outside `liver_mask_pvp` | Cases |
|---|---|
| > 1% | 339 |
| > 10% | 125 (24%) |
| > 50% | 32 |

- **Median fraction outside:** 0.027 overall; by type BCLM 0.014, CRLM 0.017, HCC 0.017, ICC 0.017, **HH 0.231**.
- **Worst cases:** 240229c11 (HH, 1.00), 231206d19 (ICC, 0.99), 231025c23 (HH, 0.99), 230218b3 (HCC, 0.98).
- **Mismatched liver masks:** 3 of the 4 `BAD_CASES` (231109b01, 240504b27, 240504e30) have a liver mask with a different slice count from the PVP. That explains their exclusion. After v1's data preparation resampled those masks onto the PVP grid, 240504b27 and 240504e30 pass QA and were re-included, along with 240504e48, which shows no problem. Only 231109b01 stays excluded (training set 357 → 360).

A liver model trained on these masks learns to exclude such tumors, most of all hemangiomas. That motivates v2: liver label = liver mask ∪ tumor mask. LiTS uses the same convention, counting tumor as liver.

## Metric

- **Primary:** per-case liver Dice on the 78 test cases. Full volume, 8-fold TTA, original NIfTI grid.
- **Secondary:** IoU, precision, recall; per-tumor-type Dice.
- **For comparing label variants:** `Dice_vs_union` (Dice against liver ∪ tumor, the same reference for every version) and `tumor_covered` (fraction of GT tumor voxels inside the predicted liver). Both are in `per_case_test.csv` and `metrics.json`.
- **Data QA:** number of flagged cases; tumor-inside-liver fraction; liver volume range.

## Versions

| Version | Change | Status | Test liver Dice (per case) |
|---|---|---|---|
| [v0](v0_notebook_labels/) | Notebook labels: liver = `mask_pvp ≥ 1` | Done (documented, not rerun) | Not measurable: the label was the tumor |
| [v1](v1_official_liver_masks/) | Official `liver_mask_pvp.nii.gz` via `data_prep/` | Done | **0.950** (0.941–0.957); HH `tumor_covered` 0.67 |
| v2 | Liver label = official liver mask ∪ tumor mask (`--set liver_includes_tumor=True`) | Next | — |
| v3 (only if needed) | TotalSegmentator cross-check of cases still flagged | Conditional on v2 | — |

v1 and v2 are both scored against their own label definition, and also against the union label, so they can be compared on the same reference.

## Conclusion so far

v0's "liver" label was the tumor mask, so no DS²Net liver result so far is valid (see [v0](v0_notebook_labels/)). v1 fixes the labels at the data level and gives a Stage 1 baseline of per-case Dice 0.950 (95% CI 0.941–0.957). The model reproduces the official mask's omission of tumors, most of all hemangiomas (`tumor_covered` 0.67). That's what v2 tests.
