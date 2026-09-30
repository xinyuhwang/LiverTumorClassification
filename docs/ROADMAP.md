# HERALD research roadmap

These are 11 proposals from two papers: OrganLens (O1–O7) and GigaPath-Flash / GigaTIME-Flash (G1–G4). Each is ranked by likely impact on HERALD and by cost to implement, then placed in an execution order. Each proposal becomes an experiment under `experiments/`, with versions tracked there (see `experiments/README.md`).

**Scales.** Both run 1 (lowest) to 5 (highest).
- **Impact:** how much it could change the validity or accuracy of HERALD's results.
- **Cost:** engineering plus compute. 1 is a flag or small script; 5 is new pretraining.

## Summary

| ID | Proposal | Impact | Cost | Tier | Experiment | Status |
|---|---|:-:|:-:|:-:|---|---|
| O3 | Correct liver labels (official masks; TotalSegmentator only as QA) | 5 | 1 | 1 | E01 | Done: liver Dice 0.973 (0.969–0.975) |
| O5 | Evaluation standards: shared split, per-case metrics, bootstrap CIs | 5 | 1 | 1 | E02 | v1 done (`common/evaluate.py`) |
| O1 | Mask-weighted patch pooling + tumor-area slice weighting | 4 | 2 | 1 | E03 | Planned |
| G1 | ABMIL per-patient aggregation | 3 | 1 | 1 | E03 | Planned |
| O2 | Joint segment + classify, with the mask as an auxiliary task | 5 | 4 | 1 → 3 | E04 | Planned (largest) |
| O4 | Keep peritumoral context (wider / dual-scale crops) | 3 | 1 | 2 | E05 | Backlog |
| G4 | Report compute (params, FLOPs, latency) with accuracy | 2 | 1 | 2 | folded into E02 | Backlog |
| O7 | CT-pretrained backbone (OrganLens encoder) for Stage 3 | 3 | 3 | 2 | E06 | Backlog |
| G2 | LoRA fine-tuning (classifiers now; FM segmenter later) | 3 | 2–4 | 2 | E07 | Backlog |
| O6 | Phase-identity conditioning of a shared encoder | 3 | 4 | 3 | E08 | Backlog |
| G3 | Distil the Stage 3 ensemble into one model | 2 | 3 | 3 | E09 | Backlog, after the ensemble is final |

**Tier 1** is what we're doing now, in order E01 → E02 → E03 → E04. E04 is much bigger than the first three and starts once they're done. **Tier 2** is cheap follow-ups and backbone swaps. **Tier 3** is larger research directions.

## Execution order and why

1. **E01, liver labels (O3).** Everything downstream depends on correct labels. The notebooks' "liver" was actually the tumor mask, so no DS²Net liver result so far is valid. The dataset turned out to ship official liver masks, which makes this cheap.
2. **E02, evaluation harness (O5 + G4).** Needed before any comparison means anything. After E02, every version is scored the same way:
   - one split
   - per-case metrics
   - bootstrap 95% CIs
   - paired tests between versions
   - compute cost
3. **E03, pooling and aggregation (O1 + G1).** The biggest gain in OrganLens's ablation (0.832 → 0.856 AUROC) for a small, contained change to Stage 3. It's also the first step toward removing the ground-truth mask at inference.
4. **E04, joint model (O2, fine-tuned with G2's LoRA).** The highest-impact direction and the paper's own future work, but also the costliest. It reuses E02's evaluation and E03's pooling.

## Proposals in detail

### O3 · Correct liver labels (E01)

- **Idea.** In OrganLens, TotalSegmentator masks define organ targets.
- **Finding (2026-09-29).** MCT-LTDiag v3.1 already contains `liver_mask_pvp.nii.gz` for each case, alongside the tumor-only `mask_pvp.nii.gz`. `data_prep/` writes both as `liver_mask.nii.gz` / `tumor_mask.nii.gz`, and the DS²Net scripts use them.
- **Plan.**
  - v0: document the notebook labels (tumor used as liver).
  - v1: use the official masks, QA all 517 cases, and retrain Stage 1.
  - v2, only if QA finds bad masks: cross-check against TotalSegmentator.
- **Measure.** Per-case liver Dice on test; QA flags; tumor-inside-liver fraction.
- **Risk.** Low.

### O5 · Evaluation standards (E02)

- **Idea.** OrganLens uses one patient-level split for every model, frozen features with the same small head, and patient-level bootstrap CIs (1,000 resamples). It also flags baselines that saw the evaluation data during pretraining.
- **HERALD today.** Three pipelines on three different splits, 78 test cases, no CIs, and slice-level and per-case metrics mixed together in the paper.
- **Plan.**
  - Already done: the shared split in `common/`, and per-case metrics in `ds2net/train.py`.
  - Next: `common/evaluate.py`, which takes any run's `per_case_test.csv` and reports bootstrap 95% CIs plus a paired bootstrap between two runs.
  - Add per-case output to `unet_hybrid/`, and rerun the baselines on the shared split.
  - Include G4's compute columns (parameters, FLOPs, inference time per case).
- **Measure.** CI width; which differences between models are significant.
- **Cost.** Low: scripts only, plus baseline reruns.

### O1 · Mask-weighted pooling (E03)

- **Idea.** OrganLens pools patch features weighted by a predicted organ mask, then weights slices by predicted organ area. Its ablation: mean CLS 0.802 → uniform patch pooling 0.832 → mask-weighted 0.851 → plus area weighting 0.856.
- **HERALD today.**
  - `stage3_paper.py` feeds the tumor mask as a 4th input channel and averages slice probabilities equally.
  - The DS²Net Stage 3 uses a bounding-box crop plus a background ring.
- **Plan.** Take the backbone's patch-feature map, pool it weighted by the tumor mask (ground truth first, then Stage 2's predicted probabilities), and weight slices by tumor area. Compare against the current 4-channel input.
- **Measure.** Case-level accuracy, macro-F1 and macro-AUC with CIs from E02, oracle masks and predicted masks.

### G1 · ABMIL aggregation (E03)

- **Idea.** GigaPath-Flash aggregates tile embeddings into a slide-level label with attention-based multiple-instance learning (ABMIL).
- **HERALD.** Aggregating slice embeddings into one tumor type per patient is the same problem. ABMIL learns which slices matter, where the current code takes a plain average.
- **Plan.** Tested in E03 as another aggregation choice, alongside mean and area-weighted.

### O2 · Joint segment + classify with mask supervision (E04)

- **Idea.** In OrganLens, removing the auxiliary mask decoder cost the most (0.853 → 0.809).
- **HERALD.** One shared encoder that segments the tumor and classifies it, with mask-weighted pooling from E03 feeding the classifier. That removes the ground-truth mask from inference, which is the paper's "tighter integration" future work.
- **Extra labels.** `meta_info_tumor` lists per-case radiological features (non-rim APHE, washout, capsule, rim APHE, peripheral nodular enhancement, central scar, lobulated). These could serve as extra auxiliary targets.
- **Encoder.** Fine-tuned with LoRA (G2).
- **Cost.** High: a new model and training loop.

### O4 · Peritumoral context (E05)

Parenchyma around the tumor carries diagnostic signal: enhancement relative to background liver, and cirrhosis for HCC. Ablate the crop margin (`--margin_frac`, `roi_margin_mm`) and try a dual-scale input (tumor crop + liver view). Cheap: config flags plus reruns.

### G4 · Report compute alongside accuracy

Add parameter count, FLOPs, and per-case inference time to every `metrics.json`, and plot accuracy against compute. This matters for the five-model, 4-view-TTA ensemble. Folded into E02.

### O7 · CT-pretrained backbone (E06)

Add the released OrganLens encoder (ViT-L/16, CT-pretrained DINOv2, liver representation) as a Stage 3 backbone option, compared against ImageNet weights.
- **Risks.**
  - OrganLens was trained on non-contrast chest CT with a [−1000, 1000] HU window, while HERALD uses contrast abdominal CT.
  - ViT-L is about 300M parameters.
  - Its liver representation is untested on liver disease.

### G2 · LoRA fine-tuning (E07)

GigaTIME-Flash puts LoRA (rank 8, α 16, on attention `qkv`/`proj`) on a ViT and trains a light conv decoder that takes skip features from blocks 4, 6, 9 and 12.
- **First:** replace "unfreeze the last blocks" with LoRA in the ViT/Swin classifiers. Low cost.
- **Later:** a foundation-model encoder with LoRA and a light decoder as a Stage 2 segmenter. High cost.

### O6 · Phase-identity conditioning (E08)

OrganLens adds a learned organ embedding to the CLS token, scaled by a factor that starts at 0. HERALD could do the same with a phase embedding (nc/art/pvp/delay) in one shared encoder, instead of stacking the phases as channels. That's a principled fix for the PhaseNorm problem in `ds2net/PORTING.md`. Cost is high: it needs a per-phase encoder pass and a way to fuse the phases.

### G3 · Ensemble distillation (E09)

GigaPath-Flash distils a 1B-parameter teacher into a 22M student. For HERALD, distil the Stage 3 ensemble (5 backbones × 4 TTA views = 20 passes per slice) into one student. The paper's note that the KoLeo loss term destabilised distillation into a small model is worth remembering. Only worth doing once the ensemble is final and deployment speed matters.
