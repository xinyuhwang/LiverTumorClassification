# HERALD research roadmap

These are 11 proposals from two papers: OrganLens (O1–O7) and GigaPath-Flash / GigaTIME-Flash (G1–G4). Each is ranked by likely impact on HERALD and by cost to implement, then placed in an execution order. Each proposal becomes an experiment under `experiments/`, with versions tracked there (see `experiments/README.md`).

**Scales.** Both run 1 (lowest) to 5 (highest).
- **Impact:** how much it could change the validity or accuracy of HERALD's results.
- **Cost:** engineering plus compute. 1 is a flag or small script; 5 is new pretraining.

## Current status (2026-10-02)

- **Done:**
  - E01: liver labels, liver Dice 0.973.
  - E02: evaluation harness (CIs, paired tests, validation output, global Dice, size bins).
  - E05: tumor baselines (DS²Net 0.726 cascade; UNet-Hybrid 0.657 oracle, test).
  - E06 v1: nnU-Net 3D, tumor 0.796 per case / 0.895 global on validation.
- **Goal:** tumor Dice > 0.90, with the measure still to be defined (per case / global / by size). nnU-Net reaches 0.90 for total tumor volume ≥ ~150 ml and for hemangiomas; small tumors (< 10 ml) are ~0.65–0.71.
- **Next:** E06 v2 (5-fold nnU-Net ensemble; scripts ready). E12: nnU-Net's recipe in DS²Net Stage 2, one change per version (v1 native 5 mm slices: code ready). Small-tumor work after that.

## Summary

E05 (tumor segmentation baselines) was added on 2026-09-30, outside the 11 proposals, because Stage 2 had no valid numbers under the new pipeline. The backlog items formerly numbered E05–E09 are now E07–E11: E06 (nnU-Net 3D baseline, "what's achievable") was added on 2026-10-01 after the project goal of tumor Dice > 0.90 was stated. E12 (nnU-Net's recipe in DS²Net) was added on 2026-10-02; it follows the reserved backlog numbers E07–E11. R1–R3 and R6 come from a second batch of related papers (2026-10-02); sources are cited in brackets and listed under [Ideas from related papers](#ideas-from-related-papers-2026-10-02).

| ID | Proposal | Impact | Cost | Tier | Experiment | Status |
|---|---|:-:|:-:|:-:|---|---|
| O3 | Correct liver labels (official masks; TotalSegmentator only as QA) | 5 | 1 | 1 | E01 | Done: liver Dice 0.973 (0.969–0.975) |
| O5 | Evaluation standards: shared split, per-case metrics, bootstrap CIs | 5 | 1 | 1 | E02 | v1 done (`common/evaluate.py`) |
| O1 | Mask-weighted patch pooling + tumor-area slice weighting | 4 | 2 | 1 | E03 | Planned |
| G1 | ABMIL per-patient aggregation | 3 | 1 | 1 | E03 | Planned |
| O2 | Joint segment + classify, with the mask as an auxiliary task | 5 | 4 | 1 → 3 | E04 | Planned (largest) |
| O4 | Keep peritumoral context (wider / dual-scale crops) | 3 | 1 | 2 | E07 | Backlog |
| G4 | Report compute (params, FLOPs, latency) with accuracy | 2 | 1 | 2 | folded into E02 | Backlog |
| O7 | CT-pretrained backbone (OrganLens encoder) for Stage 3 | 3 | 3 | 2 | E08 | Backlog |
| G2 | LoRA fine-tuning (classifiers now; FM segmenter later) | 3 | 2–4 | 2 | E09 | Backlog |
| O6 | Phase-identity conditioning of a shared encoder | 3 | 4 | 3 | E10 | Backlog |
| G3 | Distil the Stage 3 ensemble into one model | 2 | 3 | 3 | E11 | Backlog, after the ensemble is final |
| R1 | Failure rate (Dice < 0.5) and per-size-group tests next to mean Dice [2, 6] | 2 | 1 | 2 | E02 v6 | Code done, with per-lesion detection |
| R2 | Cross-phase consistency loss for tumor masks [3] | 3 | 3 | 3 | E13 | Backlog |
| R3 | Slice-interaction module in DS²Net's 2.5D input [8] | 3 | 2 | 2 | E12, after v1 | Backlog, only if E12 v1 shows slice context helps |
| R6 | Label QC with a learned quality judge (SegAE) [9] | 3 | 1 | 2 | E14 | Backlog, after E06 v2 and E12 v1 |

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

### O4 · Peritumoral context (E07)

Parenchyma around the tumor carries diagnostic signal: enhancement relative to background liver, and cirrhosis for HCC. Ablate the crop margin (`--margin_frac`, `roi_margin_mm`) and try a dual-scale input (tumor crop + liver view). Cheap: config flags plus reruns.

### G4 · Report compute alongside accuracy

Add parameter count, FLOPs, and per-case inference time to every `metrics.json`, and plot accuracy against compute. This matters for the five-model, 4-view-TTA ensemble. Folded into E02.

### O7 · CT-pretrained backbone (E08)

Add the released OrganLens encoder (ViT-L/16, CT-pretrained DINOv2, liver representation) as a Stage 3 backbone option, compared against ImageNet weights.
- **Risks.**
  - OrganLens was trained on non-contrast chest CT with a [−1000, 1000] HU window, while HERALD uses contrast abdominal CT.
  - ViT-L is about 300M parameters.
  - Its liver representation is untested on liver disease.

### G2 · LoRA fine-tuning (E09)

GigaTIME-Flash puts LoRA (rank 8, α 16, on attention `qkv`/`proj`) on a ViT and trains a light conv decoder that takes skip features from blocks 4, 6, 9 and 12.
- **First:** replace "unfreeze the last blocks" with LoRA in the ViT/Swin classifiers. Low cost.
- **Later:** a foundation-model encoder with LoRA and a light decoder as a Stage 2 segmenter. High cost.

### O6 · Phase-identity conditioning (E10)

OrganLens adds a learned organ embedding to the CLS token, scaled by a factor that starts at 0. HERALD could do the same with a phase embedding (nc/art/pvp/delay) in one shared encoder, instead of stacking the phases as channels. That's a principled fix for the PhaseNorm problem in `ds2net/PORTING.md`. Cost is high: it needs a per-phase encoder pass and a way to fuse the phases.

### G3 · Ensemble distillation (E11)

GigaPath-Flash distils a 1B-parameter teacher into a 22M student. For HERALD, distil the Stage 3 ensemble (5 backbones × 4 TTA views = 20 passes per slice) into one student. The paper's note that the KoLeo loss term destabilised distillation into a small model is worth remembering. Only worth doing once the ensemble is final and deployment speed matters.

## Ideas from related papers (2026-10-02)

Nine papers were reviewed on 2026-10-02. None changes the plan; four ideas went into the backlog and two give context. Numbers in brackets refer to the source list below; verify quoted numbers against the papers before citing. Author lists are complete for all nine. All nine include Zongwei Zhou as an author, so they come from one group and its collaborators rather than independent teams.

### R1 · Failure rate and per-size tests (E02)

Report the share of cases with tumor Dice < 0.5 next to the mean, overall and per size group, and test differences between versions per size group. Sources: per-size-bucket results with t-tests [2]; reporting failed cases [6]. Cheap: an addition to `common/evaluate.py`. It makes small-tumor progress visible, which a mean hides.

### R2 · Cross-phase consistency loss (E13)

The four phases show the same tumor, so masks predicted from each phase should agree. A loss term can enforce it. RT-Super [3] uses a related constraint over time: a tumor in an earlier scan must lie inside the same tumor in a later, larger one. Our phases are already on the PVP grid, so no extra registration is needed, though residual misregistration (`liver_air_frac` QA) limits how strict the loss can be. A candidate for DS²Net or the joint model (E04).

### R3 · Slice-interaction module (E12)

A module that mixes features across neighbouring slices (attention + depthwise convolutions) in a 2.5D network [8]. It's only worth trying if E12 v1 shows that real 5 mm slice context helps DS²Net.

### R6 · Label QC with a learned quality judge (E14)

SegAE [9] predicts a mask's Dice from the image, the mask and the structure's name, with no reference label. It correlates with real Dice at r = 0.902, takes 0.06 s per 3D mask, and found 8–13% poor masks in large public datasets. Weights are released. Two uses for HERALD:
- **Screen MCT-LTDiag's labels.** E01 found problems in the official liver masks, but the tumor masks have only been checked for overlap with the liver. Flag low-scoring masks for a visual check, and exclude or fix confirmed bad ones in a new version.
- **Flag likely failed predictions** per case, alongside R1's failure rate.

**Release check (2026-10-03):** the public repository (github.com/Schuture/SegAE, commit 027fa66) does not contain the paper's vision-language judge. It ships an earlier SegAE: a ResNet-50 regressor on a 2-channel input (CT window [−200, 200] + mask, 256 × 256 crops) conditioned on fixed per-class embeddings for 145 DAP Atlas classes. **Liver is class 13; there is no tumor class**, so tumor masks can't be scored without retraining. The inference script also needs small fixes to run. Usable now for liver masks only.

**First step (cheap, inference only):** run SegAE on nnU-Net's 78 validation predictions and check that its predicted Dice tracks the real per-case Dice we already have. It was trained on PET/CT with a narrow HU window and never learned to predict tumor Dice, so this check decides whether it's usable on our 4-phase contrast CT at all.

### Context for existing items

- **R4, the 0.90 goal.** Two radiologists agreed to Dice 0.742 on pancreatic tumors [6], and nnU-Net reached only 0.17–0.43 Dice on small (≤ 2 cm) tumors of other organs on external data [3]. MCT-LTDiag has one annotation per case, so our own inter-rater agreement can't be measured. Use these when defining the 0.90 target with the team.
- **R5, supports O7 (E08).** Initial weights pretrained on the same organ held up better on an unseen centre than generic foundation-model weights [6].

### Sources

1. Zhou Z, Rahman Siddiquee MM, Tajbakhsh N, Liang J. UNet++: A Nested U-Net Architecture for Medical Image Segmentation. DLMIA/ML-CDS 2018, LNCS 11045:3–11. doi:10.1007/978-3-030-00889-5_1
2. Zhou Z, Rahman Siddiquee MM, Tajbakhsh N, Liang J. UNet++: Redesigning Skip Connections to Exploit Multiscale Features in Image Segmentation. IEEE TMI 2020;39(6):1856–1867. doi:10.1109/TMI.2019.2959609
3. Bassi PRAS, Li W, Gu H, Chen J, Zhou X, Zhu Z, Er S, Hamamci IE, Menze BH, Akan GE, Wang K, Yang Y, Yuille AL, Zhou Z. RT-Super: Learning Tumor Segmentation from Longitudinal Images and Reports. arXiv:2609.35637, 2026 (preprint)
4. Myronenko A, Yang D, Tang Y, Turkbey B, Simon B, Harmon S, Makwana R, Aboian M, Azamat S, Hamamci IE, Er S, Menze B, Zhou Z, Li W, Edgar M, He Y, Guo P, Xu D. NV-Reason-CT: 3D Visual Language Model for CT Analysis. arXiv:2609.27511, 2026 (preprint)
5. Luo Y, Guo Y, Li W, Zhou Z, Zhang R, Ding K. LeCor: Learning to Be Corrected by Meta-Learned Test-Time Training for Interactive 3D Lung-Tumour Segmentation. arXiv:2609.09477, 2026 (preprint)
6. Aktas HE, Sen Tasci E, Peng L, Tasci ME, Taktak YB, Taflan SS, Tutun B, Bol F, Iren M, Ekici M, Bejar AM, Keles E, Pan H, Dou W, Gultekin B, Akin A, Ikizgul O, Cetin O, Uysal E, Mureva M, Nalbant MO, Kaya N, Medetalibeyoglu A, Atakir K, Akkus Yildirim B, Dagoglu Kartal G, Zhou Z, Erturk SM, Miller FH, Durak G, Bagci U. Multicenter Validation of Foundation Model Adaptation for Automated Pancreatic Tumor Delineation on CT Scans. Cancers 2026;18(17):2836. PMC13564768
7. Li J, Xie H, Ji W, Zhou Z, Yu S, Wu J, Bi Q, Zheng Y. Taking a Deep Look at Multi-rater Agreement for Calibrated Medical Image Segmentation (MRNet+). IEEE, early access 2026, IEEE Xplore 11614067
8. Luo Y, Guo Y, Hooshangnejad H, Zhang R, Feng X, Chen Q, Ngwa W, Zhou Z, Ding K. Multimodal Slice Interaction Network Enhanced by Transfer Learning for Precise Segmentation of Internal Gross Tumor Volume in Lung Cancer PET/CT Imaging. IEEE ICHI 2026. doi:10.1109/ICHI69079.2026.00072
9. Chen Y, Zhou Z, Li W, Yuille A. Large-Scale Label Quality Assessment for Medical Segmentation via a Vision-Language Judge and Synthetic Data. arXiv:2601.14406, 2026 (preprint). Code and weights: github.com/Schuture/SegAE
