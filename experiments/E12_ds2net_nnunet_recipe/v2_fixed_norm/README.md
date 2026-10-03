# E12 · v2 — DS²Net Stage 2 without PhaseNorm

| | |
|---|---|
| Status | done: smoke test 1164419 (10 min), full run 1164534 (5 h 24 min on 1× RTX PRO 6000), 2026-10-03 |
| Compared with | E12 v1 (`e12_v1_native_z`), cascade mode, on validation |
| Change | `phase_norm=fixed`: replace PhaseNorm (instance normalisation per slice and per channel) with one learnable per-channel affine shared by all slices (`FixedPhaseScale`, initialised to (x − 0.5) / 0.25 on the HU-windowed input). Everything else as v1 |
| Code | commit `b4810cd` (option added in `e1fcf3a`); v1's `--set` plus `phase_norm=fixed` |
| Run | `$HERALD_STORE/results/runs/ds2net/e12_v2_fixed_norm` |
| Date | 2026-10-02 |

## Hypothesis

PhaseNorm subtracts each channel's own mean and divides by its own standard deviation, for every slice separately. Each phase is already windowed to a fixed HU range, so this step removes what the window preserved:

- **across phases:** how much a lesion enhances in arterial vs portal venous vs delayed (the pattern radiologists use);
- **across slices:** brightness relations between the centre slice and its real ±5 mm neighbours (v1);
- **across patients:** absolute HU levels.

nnU-Net normalises with fixed dataset-wide statistics instead. If PhaseNorm is what kept v1's real slice context from helping, v2 should gain, most on small and mid-size tumors.

**Side effect, part of the same change:** DS²Net's LI-RADS phase attention gates each phase using its mean intensity. After PhaseNorm that mean is a constant, so the gate ignored the image (noted in `models.py`). With fixed scaling the gate becomes input-dependent. The two can't be separated without a further version.

**If v2 shows no gain,** PhaseNorm wasn't the blocker, and the more likely limit is the 2.5D context itself (one slice each side vs nnU-Net's 32).

## How to reproduce

See [`run.sh`](run.sh): a ~10 min smoke test, then the full run (~5.5–6 h). Reuses v1's slice cache. Needs E01 v2's liver masks on scratch (purge ~2026-10-30).

## Results

Best slice-level val Dice 0.8712 at epoch 30 (v1: 0.8716 at 36). Test slice-level 0.872 (v1: 0.862).

### Per case, full volume, cascade mode (B − A, paired, 95% bootstrap CI)

| | v1 (PhaseNorm) | v2 (fixed) | Diff (95% CI) | Better / worse |
|---|---|---|---|---|
| **Val tumor Dice** | 0.767 | **0.780** | +0.013 (−0.003 to +0.030), bootstrap p 0.11, Wilcoxon p 0.37 | 40 / 38 |
| Val recall | 0.759 | 0.795 | **+0.035 (+0.018 to +0.055)** | 62 / 16 |
| Val precision | 0.820 | 0.804 | −0.015 (−0.031 to 0.000), Wilcoxon p < 0.001 | 18 / 60 |
| Val global Dice | 0.901 | 0.900 | | |
| Val cases with Dice < 0.5 | 7 | 6 | | |
| Test tumor Dice (report) | 0.722 | 0.735 (0.680–0.784) | | |
| Test global Dice | 0.864 | 0.871 | | |
| Test cases with Dice < 0.5 | 11 | 9 | | |

Against other references on val: vs E05 v1 (notebook DS²Net in the liver box) +0.011 (−0.003 to +0.025), 45 / 33; vs nnU-Net E06 v1 −0.016 (−0.051 to +0.029), 26 / 52.

### By total tumor volume (val, per-case mean, diff with 95% CI)

| Group | n | v1 | v2 | Diff |
|---|---|---|---|---|
| < 10 ml | 16 | 0.512 | 0.554 | +0.041 (−0.012 to +0.092) |
| 10–50 ml | 20 | 0.777 | 0.795 | +0.018 (−0.011 to +0.064) |
| 50–200 ml | 16 | 0.811 | 0.805 | −0.006 (−0.017 to +0.005) |
| ≥ 200 ml | 26 | 0.891 | 0.893 | +0.002 (−0.007 to +0.015) |

By type (val): BCLM 0.780, CRLM 0.755, HCC 0.681, HH 0.919, ICC 0.767.

Files: [`results/`](results/): per-case CSVs (with `gt_ml` from E06), `summary_val.json`, `compare_val_vs_v1.json`.

## Observations

- **A small, consistent gain, not significant on its own.** Val +0.013 (CI just includes 0); test moves the same way (+0.013); fewer failed cases on both.
- **The gain is where expected:** small tumors (< 10 ml) +0.041 and 10–50 ml +0.018, while large tumors don't change. Neither size-group CI excludes 0 (16 and 20 cases).
- **The model now finds more tumor:** recall +0.035 is clearly significant (62 of 78 cases up), at a small cost in precision (−0.015). With absolute intensities and image-dependent phase gating, faint lesions are less often missed but borders are a little looser.
- **So was PhaseNorm the blocker?** Partly. Removing it gives the first gain in E12, concentrated on small tumors, and it's the first DS²Net version within noise of nnU-Net on val (−0.016, n.s.). But it's modest, so limited 2.5D context likely still matters too.
- The phase-attention gate now depends on the image; this version can't separate that from the normalisation change (see Hypothesis).

## Decision (made on validation)

- **Keep `phase_norm=fixed` as the base**, although the per-case gain isn't significant: the direction holds on val and test, recall improves significantly, failures drop, the gain sits on small tumors, and it removes a known defect (input-independent phase gating). Recorded as a judgement call, not a significant result.
- **Next: v3 = longer training without early stopping** (the best epoch was 30 of 50, so check whether a longer, lower-LR schedule helps), then v4 (Dice + CE loss). R3 (slice-interaction module) becomes worth trying now that removing PhaseNorm helped small tumors.
