# Stage 3 design: two implementations

Stage 3 (tumor subtype classification: BCLM, CRLM, HCC, HH, ICC) has two separate implementations in `unet_hybrid/`:

| | Original | Paper version |
|---|---|---|
| Entry point | `train.py --stage 3` | `stage3_paper.py` |
| Model code | `models.py` (`ResNetClassifier`, `EfficientNetClassifier`, `EnsembleClassifier`, `SwinClassifier`) | self-contained in `stage3_paper.py` |
| Matches the paper's Methods | No | Yes, where the paper is specific |

The original is kept because it's the code that existed when the paper was written. The paper version implements Stage 3 as the paper's Methods section describes it, so the reported setup can be reproduced and checked. The two share the split loader (`train.load_splits`), the HU window, and the resize helpers in `datasets.py`. Nothing else is shared, so changes to one don't affect the other.

## How they differ

| | Original (`train.py --stage 3`) | Paper version (`stage3_paper.py`) |
|---|---|---|
| Input | 12 ch: 4 phases × 3 slices | 4 ch: PVP slices s−1, s, s+1 + ground-truth tumor mask of slice s |
| Region | Whole slice, resized | Square crop around the tumor, resized to 224 × 224 |
| Slices used | 1 liver slice (random when training, middle slice at eval) + 12 evenly spaced liver slices. These may miss the tumor. | Up to 8 slices with the largest tumor area |
| Backbones | Two-stream ResNet-50, two-stream EfficientNet-B3, Swin-T/B (single stream) | EfficientNet-B3, ViT-B/16, Swin-T, Swin-B, Stage 2 UNet encoder (+ ResNet-50, from Table 7) |
| First-layer adaptation | 3 → 12 ch, all channels = mean filter | 3 → 4 ch: pretrained filters kept, 4th (mask) = their mean |
| Frozen layers | Everything trainable, except Swin (last stage only) | Per the paper (see below) |
| Head | 2 × `_StreamHead` + fusion MLP (Swin: `{256,128,C}`) | `FrozenHead`: MLP `{256, 128, C}`, BatchNorm, dropout 0.5 / 0.3 |
| Ensemble | ResNet + EfficientNet, learned logit weights, trained jointly | Separately trained backbones, equal-weight softmax averaging |
| TTA | None | 4 views: original, h-flip, v-flip, h+v flip |
| Optimiser | AdamW, lr 5e-5, batch 4, ≤30 epochs, patience 7 | AdamW, lr 1.5e-4, wd 1e-4, batch 16, cosine, ≤40 epochs, patience 10 |
| Prediction unit | 1 prediction per case | Mean of slice probabilities per case |

## Paper specification → code

| Paper (Methods, "Stage 3 Classification") | `stage3_paper.py` |
|---|---|
| 4-ch input: 2.5D PVP triplet + GT tumor mask (oracle upper bound) | `extract_case_crops` |
| EfficientNet-B3, ImageNet, first conv 3 → 4 by weight copying, final two feature blocks unfrozen | `PaperClassifier`, `efficientnet_b3`: `features[-2:]` trainable |
| ViT-B/16, patch embedding 3 → 4, fully frozen except final two encoder blocks, 224 input | `vit_b16`: `blocks[-2:]` trainable |
| Swin-Tiny / Swin-Base, patch embedding 3 → 4, frozen except final stage, 768 / 1024-d pooled features | `swin_tiny`, `swin_base`: `layers[-1]` trainable |
| UNetTransformerClassifier: Stage 2 encoder (enc1–enc4 + bottleneck), GAP on bottleneck, fully frozen, first conv 3 → 4 | `unet`: `UNetEncoder`, head only trainable |
| Shared head: MLP `{256, 128, C}`, BN + dropout 0.5 / 0.3 | `FrozenHead` |
| Cross-entropy, class weights ∝ 1 / frequency | `run_train` (counts over training cases) |
| AdamW lr 1.5e-4, batch 16, wd 1e-4, cosine, ≤40 epochs, patience 10 on val accuracy | `parse_args` defaults, `run_train` |
| Equal-weight softmax averaging | `run_ensemble` |
| TTA over 4 flip views | `predict_cases` (`TTA_VIEWS`) |

## Choices the paper doesn't specify

Each of these is a default and can be changed from the command line.

- **Slices per case** (`--max_slices 8`): the slices with the largest tumor area.
- **Crop** (`--margin_frac 0.25`): a square around all tumor pixels on the slice, padded by 25% of its side on each side, minimum 32 px, zero-padded where it runs past the image. Slices with more than one lesion get one box covering all of them.
- **Case-level prediction**: TTA-averaged softmax, averaged over the case's slices. The paper reports results per patient (78 test cases) but doesn't say how slice predictions are combined.
- **Training augmentation**: random horizontal and vertical flips, the same views used for TTA.
- **Frozen BatchNorm**: kept in eval mode, so pretrained running statistics aren't overwritten.
- **Input normalisation**: HU window [−150, 250] → [0, 1], with no ImageNet mean/std, the same as the rest of the pipeline.
- **ResNet-50**: it isn't in Methods, but it appears in Table 7. It's included, with `layer4` trainable.

## Things to be aware of

- **The UNet encoder takes 12 channels, not 3.** The paper says the encoder's first conv is adapted "from 3 to 4 channels". The Stage 2 encoder actually takes 12 channels (4 phases × 3 slices). The paper version keeps the Stage 2 filters for the PVP triplet (channels 6–8, `datasets.PHASE_NAMES`), and the mask channel gets their mean.
- **EfficientNet's adapted first layer stays frozen.** Following the paper literally, only the last two feature blocks are trainable. That means the mask channel's filter stays at its mean initialisation. Unfreezing `features[0]` is an obvious variant to try.
- **This is an oracle setting.** It uses ground-truth tumor masks, so accuracy is an upper bound. End-to-end inference with Stage 2's predicted masks isn't implemented.
- **The paper describes the ensemble three different ways:**
  - Methods: all five backbones
  - Table 7: EfficientNet + ViT + UNet, 0.69
  - Figure 1 caption: ResNet + EfficientNet + ViT

  `--members` accepts any combination. The default is the five from Methods.
- **Cases with no tumor voxels are skipped**, with a warning, and don't count in the accuracy.

## Usage

Run from `unet_hybrid/`. The `--data_dir`, `--ckpt_dir`, `--log_dir` and `--splits_csv` defaults match `train.py`.

```bash
# 1. Train each backbone. The UNet member needs Stage 2 first:
#    python train.py --stage 2
python stage3_paper.py --backbone efficientnet_b3
python stage3_paper.py --backbone vit_b16
python stage3_paper.py --backbone swin_tiny
python stage3_paper.py --backbone swin_base
python stage3_paper.py --backbone unet

# 2. Ensemble their saved probabilities (no retraining needed)
python stage3_paper.py --mode ensemble --members efficientnet_b3,vit_b16,swin_tiny,swin_base,unet
python stage3_paper.py --mode ensemble --members efficientnet_b3,vit_b16,unet   # Table 7

# Quick check: 6 cases, 2 epochs
python stage3_paper.py --backbone vit_b16 --smoke_test
```

Tumor crops are cached in `<ckpt_dir>/stage3_crops/` on the first run.

Outputs:
- **Checkpoint**: `<ckpt_dir>/stage3_paper_<backbone>_best.pth`
- **Per-case probabilities and metrics**: `<log_dir>/stage3_paper_<backbone>_probs.json`
- **Ensemble results**: `<log_dir>/stage3_paper_ensemble.json`
- **Printed**: accuracy, a per-class report, and a row-normalised confusion matrix
