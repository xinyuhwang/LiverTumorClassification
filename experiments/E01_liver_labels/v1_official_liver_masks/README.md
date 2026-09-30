# E01 · v1 — official liver masks

| | |
|---|---|
| Status | done (AICR job 1119412, 8 h 47 min on 1× RTX PRO 6000, finished 2026-09-30) |
| Compared with | v0 (notebook labels) |
| Change | Stage 1 trains on the dataset's `liver_mask_pvp.nii.gz` (written as `liver_mask.nii.gz` by `data_prep/`) instead of `mask_pvp ≥ 1` |
| Code | data prep at commit `9c2c3f5` (AICR job 1119015); training at commit `369e977` (job 1119412; smoke test job 1119353 passed) |
| Run | `$HERALD_STORE/results/runs/ds2net/e01_v1_liver` |
| Date | 2026-09-29 |

## Hypothesis

With real liver labels, DS²Net Stage 1 should reach per-case test liver Dice ≥ 0.95. That's the range of published LiTS results (0.96) and UNet-Hybrid's reported 0.9704, which was trained on a separate liver mask.

## Steps

1. **Data preparation and QA** (`data_prep/prepare_mct_ltdiag.py --overwrite`, then `summarize_manifest.py`) on all 517 cases:
   - resample any phase or mask off the PVP grid onto it
   - check masks are binary
   - check the tumor lies inside the liver mask (flag < 90%)
   - check liver volume
   - check phase misregistration
2. **Train** DS²Net Stage 1 on the shared split, with the notebook's hyperparameters (`ds2net/config.py`, `LIVER`).
3. **Evaluate** on the 78 test cases: per case, full volume, 8-fold TTA, original grid. Save predicted liver masks for all cases (`--save_liver_masks`).

Commands are in [`run.sh`](run.sh).

## Results

### Step 1 — data QA (done)

Files: [`results/data_qa.json`](results/data_qa.json) (summary) and [`results/manifest.csv`](results/manifest.csv) (per case).

| Check | Result |
|---|---|
| Cases prepared | 517 / 517, all 7 files, 90 GB |
| Masks binary | 517 / 517 |
| Files resampled onto the PVP grid | 17 cases. 14 had phase headers off by up to 4.5% in-plane scale and 9 mm shift; 3 had a liver mask on a different slice grid. |
| Tumor < 90% inside the liver mask | 126 cases (125 match Yu Zhang's independent audit) |
| Phase misregistration (> 10% of PVP liver mask on air/fat) | NC 18, arterial 4, delayed 16, PVP 1 (231109b01) |
| Median liver-mask air fraction per phase | ≈ 0.1% in every phase |
| Liver volume | median 1,249 ml; 7 cases outside 700–3,500 ml |

### Steps 2–3 — Stage 1 training and test (done)

60 epochs, no early stop. Best val Dice (slice-level) 0.9307 at epoch 51. Test on 78 cases: full volume, 8-fold TTA, original grid. Per-case liver masks for all 516 cases are on AICR (`e01_v1_liver/liver_masks/`).

`python common/evaluate.py summary results --by-type --md` (2,000 bootstrap resamples):

| Metric (test, per case) | v0 | v1 mean | 95% CI | median |
|---|---|---|---|---|
| Liver Dice | n/a (label was tumor) | **0.9498** | 0.9414–0.9569 | 0.9579 |
| IoU | n/a | 0.9063 | 0.8927–0.9182 | 0.9193 |
| Precision | n/a | 0.9206 | 0.9075–0.9320 | 0.9303 |
| Recall | n/a | 0.9827 | 0.9798–0.9850 | 0.9854 |
| `Dice_vs_union` | n/a | 0.9481 | 0.9390–0.9559 | 0.9559 |
| `tumor_covered` | n/a | 0.9006 | 0.8566–0.9409 | 0.9780 |

Slice-level test Dice (cached liver slices only) is 0.9373, for reference.

By tumor type (mean, 95% CI):

| Type | n | Liver Dice | `tumor_covered` |
|---|---|---|---|
| BCLM | 17 | 0.940 (0.922–0.957) | 0.943 (0.892–0.986) |
| CRLM | 16 | 0.954 (0.947–0.961) | 0.966 (0.945–0.984) |
| HCC | 16 | 0.960 (0.951–0.966) | 0.935 (0.886–0.978) |
| HH | 14 | 0.948 (0.929–0.963) | **0.672 (0.487–0.831)** |
| ICC | 15 | 0.948 (0.917–0.967) | 0.960 (0.912–0.988) |

Full tables: [`results/evaluate_summary.json`](results/evaluate_summary.json). Per case: [`results/per_case_test.csv`](results/per_case_test.csv).

## Observations

- **Hypothesis borderline.** Mean Dice is 0.9498, but the CI (0.941–0.957) extends below 0.95. Recall is very high (0.983), and precision (0.921) is the weaker side.
- **The model learned the label gap.** For each case, the fraction of tumor inside the predicted liver (`tumor_covered`) correlates 0.93 with the fraction inside the official mask (data QA). Every test case with less than half of its tumor covered is a hemangioma: 231025c23 (0.5% covered, official mask 1%), 230218c4 (12% vs 11%), 231025c14 (18% vs 21%). HH `tumor_covered` is 0.67, against 0.94–0.97 for the other types. This is the audit's finding reproduced by a trained model, and the motivation for v2.
- **The remaining error is over-segmentation.** The 6 lowest-Dice cases all predict 13–53% more liver than the reference, with recall ≥ 0.92. The worst is 231206d09 (ICC, Dice 0.755, 1.53× the reference volume). This fits the smoke-test observation: training uses only slices that contain liver, so the model has never been shown "no liver here". It's a separate change from the label question, so it belongs in its own version.
- **Phase-geometry fixes.** Several resampled cases are in the test set (e.g. 240122d65), and none is among the worst cases.
- **Not comparable with the paper's UNet-Hybrid 0.9704:** different split, test cases, and evaluation code. A paired comparison needs UNet-Hybrid Stage 1 rerun with E02 v2's per-case output.
- **GPU-bound.** The GPU ran at 98–100% utilisation (measured during the run), about 343 s per epoch.

## Decision

- **Keep v1** as the DS²Net Stage 1 baseline with correct labels (Dice 0.950, CI 0.941–0.957).
- **Next: v2**, liver label = liver mask ∪ tumor (`--set liver_includes_tumor=True`), compared with v1 via `evaluate.py compare`. Expected: HH `tumor_covered` rises towards the other types and `Dice_vs_union` improves, with little change for the non-HH types.
- **Later version:** add non-liver slices to training to reduce over-segmentation. This needs a small code change.
