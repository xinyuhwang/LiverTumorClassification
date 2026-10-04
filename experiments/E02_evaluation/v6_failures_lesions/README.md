# E02 · v6 — failure rates, per-size tests and per-lesion detection (R1)

| | |
|---|---|
| Status | done: code (18 tests pass) and CPU re-scoring of E06 v1 and v3 (~15 min), 2026-10-03 |
| Compared with | v5 (global Dice, size bins) |
| Change | R1 from [`ROADMAP.md`](../../docs/ROADMAP.md), sources [2] (per-size analysis with tests) and [6] (reporting failed cases) |
| Code | commit `777057f` |
| Date | 2026-10-03 |

## What changed

1. **`common/evaluate.py`**
   - `summary` adds **`fail_rate`**, the share of cases with Dice < 0.5, with a bootstrap CI, for every group (overall, per type, per size bin).
   - `compare` adds a paired `fail_rate` row (better = cases that stopped failing). With `--size-bins`, it repeats the paired Dice comparison **within each size group**, with CI and Wilcoxon test. Every row now has a `group` column.
2. **`common/metrics.py`: `lesion_metrics(pred, gt, voxel_ml)`**
   - **Lesions:** 26-connected components of the GT and predicted tumor masks.
   - **Detected:** a GT lesion counts as detected if any predicted voxel overlaps it.
   - **False positive:** a predicted component that touches no GT lesion.
   - **Per case:** `n_gt_lesions`, `n_detected`, `n_pred_lesions`, `n_fp_lesions` and `lesion_recall`.
   - **Per GT lesion:** volume, detected, overlap fraction and lesion Dice.
3. **Where it's used**
   - `nnunet/evaluate_predictions.py` adds the per-case lesion counts to the tumor CSV and writes `tumor/per_lesion_<split>.csv`.
   - `ds2net/train.py` adds `gt_ml`, `pred_ml` and the lesion counts to every Stage 2 tumor row, so future runs no longer need `gt_ml` copied from E06.
4. **Tests:** `tests/test_metrics.py` (lesion detection, false positives, empty volumes) and `tests/test_evaluate.py` (fail rate, size-group comparison). 18 tests pass.

**Why:** a per-case mean hides *how* small tumors fail. Lesion-level recall separates **missed lesions** (detection) from **poor outlines** (segmentation), and the two call for different fixes.

## Re-scoring existing runs

nnU-Net saves its predicted label maps, so E06 v1 and v3 can be re-scored on CPU (see [`run.sh`](run.sh), ~15 min). DS²Net runs don't save masks; they get lesion counts from their next run, or via `--eval_only` (~1 h GPU each).

## Results

The re-scored per-case CSVs (now with lesion counts) and `per_lesion_{val,test}.csv` are in `experiments/E06_nnunet_baseline/v{1,3}_*/results/tumor/`. Per-case Dice is unchanged by the re-scoring.

### Lesion detection, nnU-Net E06 v3 (val; v1 within ±0.02 in every bin)

309 GT lesions in 78 patients (median 2 per patient, up to 44).

| GT lesion volume | n | Detected | Lesion Dice when detected |
|---|---|---|---|
| < 0.1 ml | 94 | 0.06 | 0.02 |
| 0.1–0.5 ml | 47 | 0.30 | 0.59 |
| 0.5–1 ml | 24 | 0.71 | 0.55 |
| 1–5 ml | 50 | 0.84 | 0.65 |
| 5–10 ml | 17 | 0.94 | 0.73 |
| 10–50 ml | 31 | 1.00 | 0.80 |
| ≥ 50 ml | 46 | 1.00 | 0.86 |
| **All** | 309 | **0.56** (test 0.56) | |
| **≥ 0.52 ml** (1 cm sphere) | 167 | **0.90** | |
| 0.1–0.52 ml | 48 | 0.31 | |

- Missed lesions hold 0.17% of all GT tumor volume (test 0.44%).
- False-positive components: 38 in 24 of 78 patients (test 40 in 26).

## Observations

- **The headline lesion recall (0.56) is mostly labelling fragments.** The 94 GT "lesions" under 0.1 ml have a median of about 4 voxels, and 62 of them lie in patients whose largest lesion is ≥ 10 ml. One case (230525a4) alone has 16. They are almost certainly specks left by the annotation, not tumors; they should be excluded or reported separately in lesion-level metrics. They barely affect per-case Dice.
- **Real lesions ≥ 1 cm are found 90% of the time.** The 16 missed ones on val are 0.6–5.5 ml: satellite lesions next to a large hemangioma (HH, 5 cases), small multifocal HCC and BCLM.
- **Sub-centimetre lesions (0.1–0.5 ml) are the detection gap:** 30% found. When found, their outline Dice is ~0.6.
- **Small-tumor patients lose Dice two ways:** missed sub-centimetre lesions, and loose outlines of found 1–5 ml lesions (Dice ~0.65). Both matter for the < 10 ml group.
- ResEnc M (v3) and default (v1) behave the same at lesion level.

## Next

- **Fragment review:** [`review_fragments.sh`](review_fragments.sh) draws every GT component < 0.1 ml in the six val cases with the most fragments (PVP slice, liver window; red = component, yellow = other GT tumor, cyan = E06 v3 prediction), plus `fragments.csv` with size, slices spanned and distance to the nearest other tumor (`data_prep/review_lesion_fragments.py`).

- Report lesion metrics with a minimum lesion size (e.g. ≥ 0.1 ml, or ≥ 5 voxels) and check the fragments: are they annotation artefacts or real tiny satellite lesions? A visual check of the top cases (230525a4, 231025c16, 240722e100) would settle it.
- Small-lesion methods should target detection of 0.1–1 ml lesions (higher in-plane resolution, lesion-level sampling) rather than outlines of large tumors.

## Fragment review (2026-10-03)

Images and table: [`fragment_review/`](fragment_review/), one PNG per case and `fragments.csv`. 43 GT components < 0.1 ml in the six val cases with the most fragments. **Every one lies on a single slice**; median 8 voxels; nnU-Net (E06 v3) overlaps 2 of 43.

**This is a non-expert visual read** on PVP only, one slice per component. It needs a clinician to confirm, ideally looking at all four phases.

| Kind | Examples | What it looks like |
|---|---|---|
| **Boundary slivers** | 231025c16 #2–6, 231025c02 #4–6, 240722e100 #2–3, 240112d26 #6, 240620a22 #7 | 1–18 voxel pieces along the edge of a large lesion on its first or last slice, where the drawn outline breaks into dots. Annotation artefacts; harmless for Dice |
| **Uniform round marks in normal-looking liver** | 230525a4 (16 near-identical ~21-voxel discs, 7–30 mm from the main tumors), 231025c02 #2 | Same shape and size each time, like brush clicks, with no lesion visible on PVP underneath. Either tool artefacts or markers of tiny metastases visible on another phase (230525a4 is BCLM). This case has 44 GT "lesions" and nnU-Net Dice 0.48 |
| **Outside the liver** | 240112d26 #1–2 (fat between rib and liver), possibly 240620a22 #15–16 (next to bowel gas) | Labels on structures that don't look like liver tumor |
| **Next to unlabelled lesions** | 240722e100 #5–6 (specks touching clearly visible hypodense lesions that have no label), #4; 231025c02 #7 (a 13-voxel mark inside a lesion nnU-Net segments in full) | Suggests **incomplete annotation**: some real lesions are unlabelled or only marked by a speck |

### What it means

- **Most fragments aren't lesions,** so lesion-level metrics should exclude single-slice components below a size threshold (e.g. < 0.1 ml on one slice), or report them separately.
- **The ground truth may miss real lesions** (240722e100, 231025c02). Then some nnU-Net "false positives" are correct detections, and small-lesion precision and Dice are underestimated. That bears directly on defining the 0.90 target.
- **Worth a clinician's look:** 230525a4 (are the 16 marks real metastases?), 240722e100 (unlabelled hypodense lesions), 240112d26 (labels outside the liver).
- A dataset-wide version (all 516 cases, counts per kind) would show how common each pattern is; the script supports any case list.
