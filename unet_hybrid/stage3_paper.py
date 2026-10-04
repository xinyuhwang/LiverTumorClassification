"""
stage3_paper.py — HERALD Stage 3 (tumor subtype classification), implemented
as described in the paper (Methods: "Stage 3 Classification").

This is separate from the original Stage 3 in train.py / models.py, which is
kept unchanged. See STAGE3_DESIGN.md for how the two differ.

Paper specification implemented here
  Input      4 channels = 2.5D PVP triplet (slices s-1, s, s+1) + GT tumor mask
             of slice s, cropped around the tumor and resized to 224x224
  Backbones  (each trained separately, same head)
               efficientnet_b3  first conv 3→4 ch; final two feature blocks
                                unfrozen
               vit_b16          patch embedding 3→4 ch; final two encoder
                                blocks unfrozen
               swin_tiny /      patch embedding 3→4 ch; final Swin stage
               swin_base        unfrozen
               unet             Stage 2 UNetTransformer encoder (enc1-4 +
                                bottleneck), fully frozen; GAP on bottleneck
               resnet50         not in Methods, but reported in Table 7;
                                layer4 unfrozen
             3→4 ch adaptation: RGB filters kept, 4th = mean of the RGB filters
  Head       FrozenHead: MLP {256, 128, C}, BatchNorm + dropout 0.5 / 0.3
  Loss       cross-entropy, class weights ∝ 1 / class frequency (train cases)
  Training   AdamW lr 1.5e-4, wd 1e-4, batch 16, cosine schedule, ≤40 epochs,
             early stopping patience 10 on validation accuracy
  Inference  TTA over 4 views (original, h-flip, v-flip, h+v flip);
             ensemble = equal-weight softmax probability averaging

Choices the paper does not specify (defaults, all overridable)
  - Slices per case: the --max_slices (8) slices with the largest tumor area
  - Crop: square box around the slice's tumor pixels, padded by
    --margin_frac (0.25) of its side, minimum 32 px, zero-padded at edges
  - Case prediction: mean of TTA-averaged softmax over the case's slices
  - Train augmentation: random h/v flips (the same views used for TTA)
  - Frozen BatchNorm layers are kept in eval mode
  - Inputs are HU-windowed to [0,1] (datasets.HU_MIN/HU_MAX), no ImageNet
    mean/std normalisation — matching the original pipeline

Extensions (E03; off by default, so the paper version above is unchanged)
  --pooling mask   (v1, OrganLens O1) pool the backbone's spatial features
                   weighted by the tumor-mask channel instead of global
                   average / CLS pooling, and weight slices by tumor area
                   when averaging a case's slice predictions
  --mil abmil      (v2, GigaPath-Flash G1) train on whole cases: slice
                   embeddings → gated attention (Ilse et al. 2018) → one
                   prediction per case
  --tag NAME       suffix for checkpoint / output names (stage3_paper_<backbone>_<tag>);
                   ensemble members are named the same way
  --cv_folds K     (v4) cross-validated evaluation over train + val: K folds
                   stratified by type (common.splits.cv_folds); each fold trains
                   on the other folds minus 1/7 of them for early stopping, and
                   predicts its held-out cases. "val" outputs are then the
                   out-of-fold predictions for all train + val cases; "test"
                   is the average of the K fold models. --ensemble works as usual

Usage
  python stage3_paper.py --mode train --backbone efficientnet_b3
  python stage3_paper.py --mode train --backbone unet     # needs Stage 2 ckpt
  python stage3_paper.py --mode ensemble \
      --members efficientnet_b3,vit_b16,swin_tiny,swin_base,unet
"""

import os, json, argparse, time, random
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import timm
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

import datasets as DS
import models   as M
from train import load_splits, DEFAULT_SPLITS_CSV

SEED = 42
BACKBONES = ("efficientnet_b3", "vit_b16", "swin_tiny", "swin_base",
             "unet", "resnet50")
PAPER_MEMBERS = ("efficientnet_b3", "vit_b16", "swin_tiny", "swin_base",
                 "unet")


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def parse_args():
    p = argparse.ArgumentParser(description="HERALD Stage 3 (paper version)")
    p.add_argument("--mode", choices=["train", "ensemble"], default="train")
    p.add_argument("--backbone", choices=BACKBONES, default="efficientnet_b3")
    p.add_argument("--members", default=",".join(PAPER_MEMBERS),
                   help="Comma-separated backbones to ensemble "
                        "(each must have been trained first)")
    p.add_argument("--data_dir",   default="/scratch/teh.c/mct_training/mct_local")
    p.add_argument("--ckpt_dir",   default="/scratch/teh.c/mct_training/checkpoints")
    p.add_argument("--log_dir",    default="/scratch/teh.c/mct_training/logs")
    p.add_argument("--splits_csv", default=DEFAULT_SPLITS_CSV)
    p.add_argument("--cache_dir",  default=None,
                   help="Where to cache tumor crops (default: <ckpt_dir>/stage3_crops)")
    p.add_argument("--stage2_ckpt", default=None,
                   help="Stage 2 checkpoint for --backbone unet "
                        "(default: <ckpt_dir>/unet_v2_tumor_best.pth)")
    # paper hyper-parameters
    p.add_argument("--epochs",     type=int,   default=40)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--lr",         type=float, default=1.5e-4)
    p.add_argument("--patience",   type=int,   default=10)
    p.add_argument("--img_size",   type=int,   default=224)
    # choices the paper leaves open
    p.add_argument("--max_slices",  type=int,   default=8)
    p.add_argument("--margin_frac", type=float, default=0.25)
    # UNetTransformer shape (must match the Stage 2 run)
    p.add_argument("--base_channels", type=int, default=32)
    p.add_argument("--tf_heads",      type=int, default=8)
    p.add_argument("--tf_win",        type=int, default=4)
    p.add_argument("--no_pretrain", action="store_true")
    p.add_argument("--pooling", choices=["avg", "mask"], default="avg",
                   help="avg = paper (global average / CLS); mask = tumor-mask-weighted "
                        "features + tumor-area slice weighting (E03 v1)")
    p.add_argument("--mil", choices=["none", "abmil"], default="none",
                   help="abmil = attention-based MIL over a case's slices (E03 v2)")
    p.add_argument("--bags_per_batch", type=int, default=4, help="cases per batch with --mil abmil")
    p.add_argument("--tag", default="", help="suffix for output names, e.g. v1")
    p.add_argument("--cv_folds", type=int, default=0,
                   help="K-fold cross-validated evaluation over train + val (E03 v4)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--smoke_test",  action="store_true",
                   help="6 train cases, 2 epochs")
    return p.parse_args()


# ── Tumor-crop extraction ─────────────────────────────────────────────────────

def _crop_square(img, cy, cx, side):
    """Crop a side x side window centred at (cy, cx), zero-padding at edges."""
    H, W = img.shape
    r0, c0 = cy - side // 2, cx - side // 2
    out = np.zeros((side, side), dtype=np.float32)
    rs, re = max(0, r0), min(H, r0 + side)
    cs, ce = max(0, c0), min(W, c0 + side)
    out[rs - r0:re - r0, cs - c0:ce - c0] = img[rs:re, cs:ce]
    return out


def extract_case_crops(cid, data_dir, max_slices, size, margin_frac):
    """
    Returns (K, 4, size, size) float16: [PVP s-1, PVP s, PVP s+1, tumor mask s]
    for the K ≤ max_slices slices with the largest tumor area.
    """
    case_dir = os.path.join(data_dir, cid)
    pvp   = nib.load(os.path.join(case_dir, "pvp.nii.gz")).get_fdata(dtype=np.float32)
    tumor = nib.load(os.path.join(case_dir, "tumor_mask.nii.gz")).get_fdata() > 0
    pvp   = (np.clip(pvp, DS.HU_MIN, DS.HU_MAX) - DS.HU_MIN) / (DS.HU_MAX - DS.HU_MIN)
    D     = pvp.shape[2]

    area   = tumor.sum(axis=(0, 1))
    slices = [int(s) for s in np.argsort(area)[::-1] if area[s] > 0][:max_slices]

    crops = []
    for s in sorted(slices):
        rows, cols = np.where(tumor[:, :, s])
        r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        side = max(r1 - r0, c1 - c0)
        side = max(32, int(round(side * (1 + 2 * margin_frac))))
        cy, cx = (r0 + r1) // 2, (c0 + c1) // 2
        chans = [DS._resize_slice(
                     _crop_square(pvp[:, :, int(np.clip(z, 0, D - 1))], cy, cx, side),
                     size)
                 for z in (s - 1, s, s + 1)]
        chans.append(DS._resize_mask(
            _crop_square(tumor[:, :, s].astype(np.float32), cy, cx, side), size))
        crops.append(np.stack(chans))
    if not crops:
        return np.zeros((0, 4, size, size), dtype=np.float16)
    return np.stack(crops).astype(np.float16)


def load_crops(case_ids, args):
    """Build (or load cached) crops for each case. Cases with no tumor are skipped."""
    os.makedirs(args.cache_dir, exist_ok=True)
    out, t0 = {}, time.time()
    for cid in case_ids:
        path = os.path.join(args.cache_dir,
                            f"{cid}_k{args.max_slices}_s{args.img_size}"
                            f"_m{args.margin_frac}.npy")
        if os.path.exists(path):
            arr = np.load(path)
        else:
            arr = extract_case_crops(cid, args.data_dir, args.max_slices,
                                     args.img_size, args.margin_frac)
            # write-then-rename: parallel jobs never read a half-written file
            tmp = f"{path}.{os.getpid()}.tmp.npy"
            np.save(tmp, arr)
            os.replace(tmp, path)
        if len(arr) == 0:
            print(f"  WARNING: {cid} has no tumor voxels — skipped")
            continue
        out[cid] = arr
    print(f"  Crops ready for {len(out)}/{len(case_ids)} cases "
          f"({time.time() - t0:.0f}s)")
    return out


class TumorCropDataset(Dataset):
    """One sample per (case, tumor slice)."""
    def __init__(self, crops, labels, augment=False):
        self.crops, self.labels, self.augment = crops, labels, augment
        self.items = [(cid, k) for cid, arr in crops.items()
                      for k in range(len(arr))]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        cid, k = self.items[idx]
        x = self.crops[cid][k].astype(np.float32)
        if self.augment:
            if random.random() < 0.5: x = x[:, :, ::-1]
            if random.random() < 0.5: x = x[:, ::-1, :]
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.tensor(self.labels[cid], dtype=torch.long))


class TumorBagDataset(Dataset):
    """One sample per case: all its crops (a bag), for --mil abmil."""
    def __init__(self, crops, labels, augment=False):
        self.crops, self.labels, self.augment = crops, labels, augment
        self.ids = sorted(crops)

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        cid = self.ids[idx]
        x = self.crops[cid].astype(np.float32)
        if self.augment:
            if random.random() < 0.5: x = x[:, :, :, ::-1]
            if random.random() < 0.5: x = x[:, :, ::-1, :]
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.tensor(self.labels[cid], dtype=torch.long))


# ── Models ────────────────────────────────────────────────────────────────────

def adapt_conv_4ch(conv, src=slice(0, 3)):
    """
    Return a copy of `conv` taking 4 input channels: channels 0-2 keep the
    pretrained filters conv.weight[:, src]; channel 3 (tumor mask) gets their
    mean.
    """
    new = nn.Conv2d(4, conv.out_channels, conv.kernel_size, stride=conv.stride,
                    padding=conv.padding, dilation=conv.dilation,
                    groups=conv.groups, bias=conv.bias is not None)
    with torch.no_grad():
        w3 = conv.weight[:, src]
        new.weight[:, :3] = w3
        new.weight[:, 3:] = w3.mean(dim=1, keepdim=True)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


class FrozenHead(nn.Module):
    """Shared classification head: MLP {256, 128, C} with BN + dropout."""
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes))

    def forward(self, x): return self.net(x)


class UNetEncoder(nn.Module):
    """Stage 2 UNetTransformer encoder (enc1-4 + bottleneck) → GAP."""
    def __init__(self, unet):
        super().__init__()
        self.unet = unet
        self.num_features = unet.bottleneck[0].out_channels   # 16 * base

    def forward(self, x):
        return self.unet.encode(x)[-1].mean(dim=(2, 3))


class GatedAttention(nn.Module):
    """ABMIL pooling (Ilse et al. 2018): a_k ∝ exp(wᵀ(tanh(V h_k) ⊙ σ(U h_k)))."""
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.V, self.U = nn.Linear(dim, hidden), nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, h):                                   # h: (K, dim)
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))
        a = torch.softmax(a.float(), dim=0).to(h.dtype)
        return (a * h).sum(0), a.squeeze(1)


class PaperClassifier(nn.Module):
    def __init__(self, backbone, num_classes, args):
        super().__init__()
        pretrained = not args.no_pretrain
        self.backbone_name = backbone
        self.pooling = getattr(args, "pooling", "avg")

        if backbone == "efficientnet_b3":
            net = tvm.efficientnet_b3(
                weights=tvm.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)
            net.features[0][0] = adapt_conv_4ch(net.features[0][0])
            self.features = nn.Sequential(net.features, net.avgpool, nn.Flatten(1))
            feat_dim, trainable = 1536, [net.features[-2], net.features[-1]]

        elif backbone == "resnet50":
            net = tvm.resnet50(
                weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            net.conv1 = adapt_conv_4ch(net.conv1)
            self.features = nn.Sequential(*list(net.children())[:-1], nn.Flatten(1))
            feat_dim, trainable = 2048, [net.layer4]

        elif backbone == "vit_b16":
            net = timm.create_model("vit_base_patch16_224",
                                    pretrained=pretrained, num_classes=0)
            net.patch_embed.proj = adapt_conv_4ch(net.patch_embed.proj)
            self.features = net
            feat_dim, trainable = net.num_features, list(net.blocks[-2:])

        elif backbone in ("swin_tiny", "swin_base"):
            name = {"swin_tiny": "swin_tiny_patch4_window7_224",
                    "swin_base": "swin_base_patch4_window7_224"}[backbone]
            net = timm.create_model(name, pretrained=pretrained,
                                    num_classes=0, global_pool="avg")
            net.patch_embed.proj = adapt_conv_4ch(net.patch_embed.proj)
            self.features = net
            feat_dim, trainable = net.num_features, [net.layers[-1]]

        elif backbone == "unet":
            unet = M.UNetTransformer(in_ch=12, out_ch=1, img_size=args.img_size,
                                     tf_heads=args.tf_heads, tf_win=args.tf_win,
                                     base=args.base_channels)
            if os.path.exists(args.stage2_ckpt):
                unet.load_state_dict(torch.load(args.stage2_ckpt,
                                                map_location="cpu",
                                                weights_only=True))
                print(f"  Loaded Stage 2 encoder from {args.stage2_ckpt}")
            elif not args.smoke_test:
                raise FileNotFoundError(
                    f"Stage 2 checkpoint not found: {args.stage2_ckpt}")
            else:
                print("  SMOKE TEST: Stage 2 checkpoint missing, random encoder")
            # Stage 2 input is 4 phases x 3 slices; keep the PVP triplet's filters
            pvp = DS.PHASE_NAMES.index("pvp") * 3
            unet.enc1[0].block[0] = adapt_conv_4ch(unet.enc1[0].block[0],
                                                   src=slice(pvp, pvp + 3))
            self.features = UNetEncoder(unet)
            feat_dim, trainable = self.features.num_features, []   # fully frozen

        else:
            raise ValueError(backbone)

        for p in self.features.parameters():
            p.requires_grad_(False)
        for m in trainable:
            for p in m.parameters():
                p.requires_grad_(True)

        self.head = FrozenHead(feat_dim, num_classes)
        self.attn = GatedAttention(feat_dim) if getattr(args, "mil", "none") == "abmil" else None

    def spatial(self, x):
        """Backbone features before pooling, as (B, C, h, w)."""
        b = self.backbone_name
        if b == "efficientnet_b3":
            return self.features[0](x)
        if b == "resnet50":
            return self.features[:-2](x)
        if b == "vit_b16":
            t = self.features.forward_features(x)[:, self.features.num_prefix_tokens:]
            n = int(t.shape[1] ** 0.5)
            return t.transpose(1, 2).reshape(t.shape[0], t.shape[2], n, n)
        if b in ("swin_tiny", "swin_base"):
            return self.features.forward_features(x).permute(0, 3, 1, 2)
        return self.features.unet.encode(x)[-1]

    def embed(self, x):
        """One feature vector per crop: the paper's pooling, or mask-weighted (O1)."""
        if self.pooling == "avg":
            return self.features(x)
        f = self.spatial(x)
        w = F.adaptive_avg_pool2d(x[:, 3:4].to(f.dtype), f.shape[-2:])   # tumor share per cell
        ws = w.sum(dim=(2, 3))
        pooled = (f * w).sum(dim=(2, 3)) / ws.clamp(min=1e-6)
        return torch.where(ws > 1e-6, pooled, f.mean(dim=(2, 3)))      # empty mask → average

    def forward_bags(self, x, sizes):
        """ABMIL: crops of several cases stacked in x, sizes = crops per case."""
        h = self.embed(x)
        z = torch.stack([self.attn(hb)[0] for hb in torch.split(h, sizes)])
        return self.head(z)

    def train(self, mode=True):
        super().train(mode)
        # frozen BatchNorm layers keep their pretrained running statistics
        for m in self.features.modules():
            if (isinstance(m, nn.modules.batchnorm._BatchNorm)
                    and not any(p.requires_grad for p in m.parameters())):
                m.eval()
        return self

    def forward(self, x):
        return self.head(self.embed(x))


# ── Training / evaluation ─────────────────────────────────────────────────────

TTA_VIEWS = ((), (-1,), (-2,), (-2, -1))   # original, h-flip, v-flip, both


@torch.no_grad()
def predict_cases(model, crops, device, batch=32):
    """Case-level probabilities: TTA-averaged softmax, averaged over slices
    (tumor-area-weighted with --pooling mask; one attention-pooled bag with
    --mil abmil)."""
    model.eval()
    out = {}
    for cid, arr in crops.items():
        x = torch.from_numpy(arr.astype(np.float32)).to(device)
        if model.attn is not None:
            with autocast(device.type, enabled=device.type == "cuda"):
                p = sum(F.softmax(model.forward_bags(torch.flip(x, v) if v else x,
                                                     [len(x)]).float(), 1)
                        for v in TTA_VIEWS) / len(TTA_VIEWS)
            out[cid] = p[0].cpu().numpy()
            continue
        probs = []
        for i in range(0, len(x), batch):
            xb = x[i:i + batch]
            with autocast(device.type, enabled=device.type == "cuda"):
                p = sum(F.softmax(model(torch.flip(xb, v) if v else xb).float(), 1)
                        for v in TTA_VIEWS) / len(TTA_VIEWS)
            probs.append(p)
        probs = torch.cat(probs)
        if model.pooling == "mask":            # O1: slices weighted by tumor area
            area = x[:, 3].sum(dim=(1, 2)).float()
            out[cid] = ((probs * area[:, None]).sum(0) / area.sum().clamp(min=1e-6)).cpu().numpy()
        else:
            out[cid] = probs.mean(0).cpu().numpy()
    return out


def case_accuracy(probs, labels):
    return accuracy_score([labels[c] for c in probs],
                          [int(np.argmax(p)) for p in probs.values()])


def write_per_case(out_dir, split, probs, labels, class_names):
    """per_case_<split>.csv (case_id, true, pred, p_<class>) for common/evaluate.py."""
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    rows = [{"case_id": c, "true": class_names[labels[c]],
             "pred": class_names[int(np.argmax(p))],
             **{f"p_{n}": float(v) for n, v in zip(class_names, p)}}
            for c, p in sorted(probs.items())]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"per_case_{split}.csv"), index=False)


def report(title, probs, labels, class_names):
    y_true = [labels[c] for c in probs]
    y_pred = [int(np.argmax(p)) for p in probs.values()]
    acc = accuracy_score(y_true, y_pred)
    idx = list(range(len(class_names)))
    cm  = confusion_matrix(y_true, y_pred, labels=idx)
    print(f"\n── {title} — {len(y_true)} cases, accuracy {acc:.4f} ──")
    print(classification_report(y_true, y_pred, labels=idx,
                                target_names=class_names, zero_division=0))
    print("Confusion matrix (rows = true, row-normalised):")
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    print("        " + " ".join(f"{n:>6}" for n in class_names))
    for n, row in zip(class_names, norm):
        print(f"  {n:<6}" + " ".join(f"{v:6.2f}" for v in row))
    return {"accuracy": acc, "confusion_matrix": cm.tolist()}


def run_train(args, device, splits, labels, class_names, save=True, es_ids=None):
    """Train one model; returns case probabilities {"val": …, "test": …}.
    Early stopping uses val_ids, or es_ids if given (CV folds: an inner split,
    so the held-out fold in val_ids never influences training).
    save=False (CV folds) skips the per-run output files."""
    train_ids, val_ids, test_ids = splits
    tr_crops = load_crops(train_ids, args)
    va_crops = load_crops(val_ids, args)
    te_crops = load_crops(test_ids, args)
    es_crops = load_crops(es_ids, args) if es_ids is not None else va_crops

    # class weights ∝ 1 / frequency (train cases)
    counts = np.bincount([labels[c] for c in tr_crops], minlength=len(class_names))
    w = 1.0 / np.maximum(counts, 1)
    w = torch.tensor(w / w.sum() * len(w), dtype=torch.float32, device=device)

    if args.mil == "abmil":     # batches of whole cases (bags)
        loader = DataLoader(TumorBagDataset(tr_crops, labels, augment=True),
                            batch_size=args.bags_per_batch, shuffle=True, drop_last=True,
                            num_workers=args.num_workers, collate_fn=lambda b: b)
    else:
        loader = DataLoader(TumorCropDataset(tr_crops, labels, augment=True),
                            batch_size=args.batch_size, shuffle=True, drop_last=True,
                            num_workers=args.num_workers, pin_memory=True)

    model = PaperClassifier(args.backbone, len(class_names), args).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"  {args.backbone}: {sum(p.numel() for p in params):,} trainable / "
          f"{sum(p.numel() for p in model.parameters()):,} total params")

    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.epochs)
    scaler    = GradScaler(device.type, enabled=device.type == "cuda")
    ckpt      = os.path.join(args.ckpt_dir, f"stage3_paper_{args.name}_best.pth")

    best_acc, no_imp, history = -1.0, 0, []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            if args.mil == "abmil":
                sizes = [len(xb) for xb, _ in batch]
                x = torch.cat([xb for xb, _ in batch]).to(device)
                y = torch.stack([yb for _, yb in batch]).to(device)
                with autocast(device.type, enabled=device.type == "cuda"):
                    loss = criterion(model.forward_bags(x, sizes).float(), y)
            else:
                x, y = batch[0].to(device), batch[1].to(device)
                with autocast(device.type, enabled=device.type == "cuda"):
                    loss = criterion(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            losses.append(loss.item())
        scheduler.step()

        vl_acc = case_accuracy(predict_cases(model, es_crops, device), labels)
        history.append({"epoch": ep, "loss": float(np.mean(losses)),
                        "val_acc": vl_acc})
        print(f"Ep {ep:3d}/{args.epochs} | loss {np.mean(losses):.4f} | "
              f"val_acc {vl_acc:.4f} | {time.time() - t0:.0f}s", flush=True)

        if vl_acc > best_acc:
            best_acc, no_imp = vl_acc, 0
            torch.save(model.state_dict(), ckpt)
            print(f"  ✓ Saved  best val acc: {best_acc:.4f}")
        else:
            no_imp += 1
            if no_imp >= args.patience:
                print(f"  Early stopping at epoch {ep}")
                break

    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    va_probs = predict_cases(model, va_crops, device)
    te_probs = predict_cases(model, te_crops, device)
    if not save:
        return {"val": va_probs, "test": te_probs, "best_val_acc": best_acc}
    report(f"{args.name} — validation", va_probs, labels, class_names)
    result = report(f"{args.name} — test", te_probs, labels, class_names)
    save_outputs(args, labels, class_names, va_probs, te_probs,
                 {"best_val_acc": best_acc, "history": history, **result})


def save_outputs(args, labels, class_names, va_probs, te_probs, extra):
    # per-case probabilities, so ensembles can be formed without re-running models
    probs_path = os.path.join(args.log_dir, f"stage3_paper_{args.name}_probs.json")
    with open(probs_path, "w") as f:
        json.dump({"backbone": args.backbone, "name": args.name, "pooling": args.pooling,
                   "mil": args.mil, "class_names": class_names,
                   "labels": {c: labels[c] for c in {**va_probs, **te_probs}},
                   "val":  {c: p.tolist() for c, p in va_probs.items()},
                   "test": {c: p.tolist() for c, p in te_probs.items()}, **extra},
                  f, indent=1)
    out_dir = os.path.join(args.log_dir, f"stage3_paper_{args.name}")
    for split, probs in (("val", va_probs), ("test", te_probs)):
        write_per_case(out_dir, split, probs, labels, class_names)
    print(f"\nSaved → {probs_path}, {out_dir}/per_case_{{val,test}}.csv")


def run_cv(args, device, splits, labels, labels_df, class_names):
    """E03 v4: out-of-fold predictions for all train + val cases."""
    from common.splits import cv_folds
    train_ids, val_ids, test_ids = splits
    base, oof, tests, fold_acc = args.name, {}, [], []
    for k, fold in enumerate(cv_folds(train_ids + val_ids, labels_df, k=args.cv_folds, seed=0)):
        # early stopping on ~1/7 of this fold's training cases, never on the held-out fold
        inner = cv_folds(fold["train"], labels_df, k=7, seed=1)[0]
        args.name = f"{base}_cv{k}"
        print(f"\n══ fold {k}: train {len(inner['train'])}, early-stop {len(inner['val'])}, "
              f"held out {len(fold['val'])} ══")
        r = run_train(args, device, (inner["train"], fold["val"], test_ids),
                      labels, class_names, save=False, es_ids=inner["val"])
        held = r["val"]
        oof.update(held); tests.append(r["test"])
        fold_acc.append(case_accuracy(held, labels))
        print(f"  fold {k}: held-out accuracy {fold_acc[-1]:.4f}")
    args.name = base
    test = {c: np.mean([t[c] for t in tests], axis=0) for c in tests[0]}
    report(f"{base} — out-of-fold (train + val)", oof, labels, class_names)
    result = report(f"{base} — test (mean of {args.cv_folds} fold models)", test, labels, class_names)
    save_outputs(args, labels, class_names, oof, test,
                 {"cv_folds": args.cv_folds, "fold_accuracy": fold_acc, **result})


def run_ensemble(args, class_names):
    members = [m.strip() for m in args.members.split(",") if m.strip()]
    runs = []
    for m in members:
        path = os.path.join(args.log_dir, f"stage3_paper_{m}_probs.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} — train it first: "
                                    f"python stage3_paper.py --backbone {m}")
        with open(path) as f:
            runs.append(json.load(f))
        assert runs[-1]["class_names"] == class_names, f"class order differs in {m}"

    labels = runs[0]["labels"]
    out = {"members": members}
    for split in ("val", "test"):
        cases = set.intersection(*(set(r[split]) for r in runs))
        # equal-weight softmax probability averaging
        probs = {c: np.mean([r[split][c] for r in runs], axis=0) for c in sorted(cases)}
        for r, m in zip(runs, members):
            print(f"  {m:<16} {split} acc "
                  f"{case_accuracy({c: np.array(r[split][c]) for c in cases}, labels):.4f}")
        out[split] = report(f"Ensemble ({'+'.join(members)}) — {split}",
                            probs, labels, class_names)
        write_per_case(os.path.join(args.log_dir, "stage3_paper_ensemble_" + "+".join(members)),
                       split, probs, {c: int(v) for c, v in labels.items()}, class_names)
    path = os.path.join(args.log_dir, "stage3_paper_ensemble.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nSaved → {path}")


def main():
    args = parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    args.cache_dir   = args.cache_dir   or os.path.join(args.ckpt_dir, "stage3_crops")
    args.stage2_ckpt = args.stage2_ckpt or os.path.join(args.ckpt_dir,
                                                        "unet_v2_tumor_best.pth")
    args.name = args.backbone + (f"_{args.tag}" if args.tag else "")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir,  exist_ok=True)

    train_ids, val_ids, test_ids, labels_df = load_splits(args.splits_csv)
    class_names = sorted(labels_df["type"].astype(str).unique())
    labels = {c: class_names.index(t)
              for c, t in zip(labels_df["case_id"], labels_df["type"].astype(str))}
    print(f"  Classes: {class_names}")

    if args.mode == "ensemble":
        run_ensemble(args, class_names)
        return

    if args.smoke_test and args.cv_folds:   # stratified folds need a few cases per class
        train_ids, val_ids, test_ids = train_ids[:60], val_ids[:10], test_ids[:2]
        args.cv_folds, args.epochs, args.patience = 2, 2, 2
        print("SMOKE TEST — CV: 70 cases, 2 folds, 2 epochs")
    elif args.smoke_test:
        train_ids, val_ids, test_ids = train_ids[:6], val_ids[:2], test_ids[:2]
        args.epochs, args.patience = 2, 2
        args.batch_size = min(args.batch_size, 4)
        print("SMOKE TEST — 6 train cases, 2 epochs")
    if args.cv_folds:
        run_cv(args, device, (train_ids, val_ids, test_ids), labels, labels_df, class_names)
    else:
        run_train(args, device, (train_ids, val_ids, test_ids), labels, class_names)


if __name__ == "__main__":
    main()
