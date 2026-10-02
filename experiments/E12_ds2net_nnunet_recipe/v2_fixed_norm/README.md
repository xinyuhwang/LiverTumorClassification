# E12 · v2 — DS²Net Stage 2 without PhaseNorm

| | |
|---|---|
| Status | planned (code ready, not submitted) |
| Compared with | E12 v1 (`e12_v1_native_z`), cascade mode, on validation |
| Change | `phase_norm=fixed`: replace PhaseNorm (instance normalisation per slice and per channel) with one learnable per-channel affine shared by all slices (`FixedPhaseScale`, initialised to (x − 0.5) / 0.25 on the HU-windowed input). Everything else as v1 |
| Code | commit `<hash>`; v1's `--set` plus `phase_norm=fixed` |
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

```bash
python common/evaluate.py summary results/per_case_val.csv --by-type --size-bins 10 50 200 --md
python common/evaluate.py compare ../v1_native_z/results/per_case_val.csv results/per_case_val.csv --md
```

## Observations

## Decision
