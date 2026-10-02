# DS²Net: notebooks → scripts

`ds2net/` now runs from Python scripts laid out like `unet_hybrid/`. The original Colab notebooks are kept unchanged in `notebooks/` for reference.

| Notebook | Script |
|---|---|
| `notebooks/liver_segmentation.ipynb` | `train.py --stage 1`: `DS2NetLiver` |
| `notebooks/segmentation_classification.ipynb`, cell 0 | `train.py --stage 2`: `DS2NetUNet`, tumor only |
| `notebooks/segmentation_classification.ipynb`, cell 1 | `train.py --stage 3`: `EfficientNet4Phase` |

| File | Contents |
|---|---|
| `config.py` | Hyperparameters per stage, copied from the notebooks' `CFG` dicts |
| `datasets.py` | Loading, resampling, slice cache, augmentation, Stage 3 ROI patches |
| `models.py` | Architectures, losses, probability fusion |
| `train.py` | Training, validation, test evaluation for all three stages |

## Kept as in the notebooks

- **Architectures and training setup:**
  - Swin-Tiny backbone, DEM/SEM, MHA bottleneck (Stage 1)
  - PhaseNorm + LI-RADS phase attention + small-tumor branch (Stage 2)
  - EfficientNet-B0 + CBAM + relative enhancement curve (Stage 3)
  - DS² uncertainty-adaptive loss, OneCycle schedule
  - Every hyperparameter in `config.py`
- **Input:** 4 phases, per-phase HU windows, resampling to 1 mm on all three axes, 224 × 224 slices, 2.5D context-major channel order (Stage 1: 5 slices × 4 phases = 20 channels; Stage 2: 3 slices × 4 = 12).
- **Sampling:** weighted by liver area (Stage 1), or by tumor voxels × tumor-type multiplier (Stage 2).
- **Inference:** 8-fold TTA; 3-D connected-component filter for liver masks.

## Changed

### Behaviour changes

| # | Change | Why |
|---|---|---|
| 1 | **Labels come from separate `liver_mask.nii.gz` and `tumor_mask.nii.gz`**, produced by `data_prep/` from the dataset's `liver_mask_pvp.nii.gz` and `mask_pvp.nii.gz`. | The notebooks read only `mask_pvp.nii.gz`, which is **tumor-only** (labels {0,1}; checked on a real case, and shown by identical liver/tumor Dice in the notebook output). Their "liver" was the tumor. See `experiments/E01_liver_labels`. |
| 2 | **Stage 2 predicts tumor only** (1 output channel instead of 2). | Liver is Stage 1's job, as in `unet_hybrid/`. The notebook's second channel was a copy of the tumor mask anyway. |
| 3 | **All pipelines use one split**, `common/mct_ltdiag_split.csv` (360/78/78; 1 case excluded, see `common/splits.py`). | The two notebooks and `unet_hybrid/` each used a different split, so their results weren't comparable, and Stage 1's test cases could be in Stage 2's training set. |
| 4 | **Test reports both slice-level and per-case metrics.** Per-case means full volume, 8-fold TTA, on the original NIfTI grid. | The notebook's headline 0.685 was slice-level on tumor slices only; per-case it was 0.375. Both are now saved in `metrics.json`. |
| 5 | **Inference builds its input exactly like training**, resampling to 1 mm in z. | The notebook's Stage 2 inference skipped the z-resampling that training used. That changed what the ±1 context slices contained. |

### Bug fixes carried to both segmentation stages

The notebooks had diverged. The liver notebook had fixed these; the segmentation notebook had not. The scripts share one implementation and use the fixed version everywhere.

| Bug | Notebook behaviour | Script |
|---|---|---|
| OneCycle steps | Seg notebook: `steps_per_epoch = len(loader)` but stepped every 3 batches, so LR never decayed (1.78e-4 at epoch 50) | Optimizer steps per epoch = ⌈batches / accumulation⌉ |
| Boundary loss | Seg notebook selected interior pixels | Dilation − erosion (liver notebook FIX-12) |
| Up-down flip | Seg notebook p = 0.5 | p = 0.05 (liver notebook FIX-13) |
| Resume | Seg notebook didn't restore the scheduler | Model, optimizer, scheduler, scaler and history all restored |
| Stage 3 slice choice | Patch worker sorted by z, then took the first 16 slices | The 16 slices with the largest tumor area |

### Engineering changes (no effect on results)

- **No Colab code.** The Drive mounting, FUSE retries and Drive → `/tmp` → npz → npy copying are gone. The cache is float16 `.npy` in `<work_dir>/cache/ds2net/<task>_<hash>/`, memory-mapped by the dataset. The hash covers every setting that changes the cache.
- **Faster caching.** Each phase slice is resized once, not once per context copy.
- **Head storage.** The 5 heads are stored as `heads` (a `ModuleList`) instead of `head0..head4`. The notebook checkpoints can't be loaded anyway: different channel count, and trained on tumor-as-liver labels.
- **Run outputs.** Each run writes to `<work_dir>/runs/ds2net/<run_name>/`: `config.json`, `history.json`, `best.pth`, `last.pth`, `metrics.json`, `per_case_test.csv`.

## Known issues deliberately kept

These are real design problems. They stay so the ported baseline matches the notebooks, and fixing them is left to experiments (see `docs/ROADMAP.md`).

- **Stage 2 trains only on tumor slices** (`slices="tumor"`, the notebook's effective behaviour). The model never sees tumor-free liver, which drives the low per-case precision. Try `--set slices=liver`.
- **PhaseNorm erases phase differences.** It normalises every channel of each slice separately, which removes the between-phase intensity differences. As a result:
  - the LI-RADS phase attention weights don't depend on the image
  - Stage 3's relative-enhancement `patch_means` are near-constant
- **Context slices are only 1 mm apart.** Resampling z to 1 mm puts the ±1–2 slices within one original 5 mm slice, so the "2.5D context" is mostly interpolation.
- **Stage 3 uses ground-truth tumor masks** (an oracle upper bound), as in the notebook.

## Options added after the port

All are off by default, so the ported baseline stays reproducible; experiments switch them on with `--set`.

| Option (`config.py`) | Stage | Effect | Experiment |
|---|---|---|---|
| `liver_includes_tumor=True` | 1, 2 | liver label = official liver mask ∪ tumor mask | E01 v2 (adopted) |
| `keep_largest_component=True` | 1 | keep only the largest connected liver component at test time | E01 v3 (adopted) |
| `roi="liver"`, `roi_margin_mm`, `roi_liver_run` | 2 | search only a square box around the liver(+tumor) per slice. Test is scored in two modes: **oracle** (GT box) and **cascade** (box from a Stage 1 run's saved liver masks, E01 post-processing) | E05 v1 |
| `z_spacing=5.0` | 1, 2 | keep the native 5 mm between slices instead of resampling z to `target_spacing`, so context slices are real neighbours | E12 v1 |
| `samples_per_epoch=N` | 1, 2 | slices drawn per epoch (default: all cached slices). Keeps the training budget fixed when the slice count changes | E12 v1 |
| `phase_norm=fixed` | 2 | replace PhaseNorm (per-slice instance norm) with one learnable per-channel affine shared by all slices; LI-RADS phase attention then depends on the image | E12 v2 |

Every run also writes **`per_case_val.csv`** next to `per_case_test.csv` (and `_oracle` variants in liver-ROI mode). Versions are chosen on validation.

## Usage

```bash
cd ds2net
python train.py --stage 1 --run_name liver_baseline
python train.py --stage 2 --run_name tumor_baseline
python train.py --stage 2 --run_name tumor_liverslices --set slices=liver
python train.py --stage 3 --run_name cls_baseline
python train.py --stage 2 --run_name tumor_roi --set roi=liver slices=liver liver_includes_tumor=True roi_liver_run=<stage-1 run>
python train.py --stage 1 --smoke_test --no_pretrain          # quick pipeline check
```

`--data_dir` and `--work_dir` default to `$HERALD_DATA` and `$HERALD_WORK`, which `cluster/env.sh` sets on AICR.
