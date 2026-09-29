# E02 · Evaluation harness

**Roadmap items:** O5 + G4 (tier 1, impact 5, cost 1).

## Question

How do we score every version so that differences between versions are real and measurable, not artefacts of different splits, metrics, or chance on 78 test patients?

## What it provides

1. **One split.** `common/mct_ltdiag_split.csv` via `common/splits.py`, used by every pipeline: 360/78/78, with 1 case excluded (E01).
2. **Per-case test output.** Every run writes `per_case_test.csv`, one row per test patient.
3. **Patient-level statistics.** `common/evaluate.py`:
   - `summary`: bootstrap 95% CIs per metric, optionally per tumor type
   - `compare`: paired difference with its CI and p-values
4. **Compute cost.** Parameter count, FLOPs, and inference time per case, reported alongside accuracy (G4).

## Versions

| Version | Change | Status |
|---|---|---|
| [v1](v1_bootstrap_stats/) | `common/evaluate.py`: bootstrap CIs, paired comparison (bootstrap + Wilcoxon / McNemar), Markdown output; tests | Done |
| v2 | Per-case test output for `unet_hybrid/train.py` (Stages 1–3), so UNet-Hybrid can be compared with DS²Net case by case | Planned |
| v3 | Compute-cost columns (params, FLOPs, latency per case) in every `metrics.json` | Planned |

## How every experiment reports results (from now on)

```bash
python common/evaluate.py summary <run> --by-type --md            # one version
python common/evaluate.py compare <previous run> <this run> --md  # Δ vs previous
```

Paste both tables into the version README, and save the JSON (`--out`) in its `results/`.

A difference is called real only if its 95% CI excludes 0. With 78 test patients, the CI half-width for a mean Dice with SD 0.1 is about ±0.022.
