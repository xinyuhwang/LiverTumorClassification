"""
train.py — HERALD entry point
  Stage 1: liver segmentation   (UNetTransformer)
  Stage 2: tumor segmentation   (UNetTransformer, GT liver crops)
  Stage 3: tumor classification (EnsembleClassifier: ResNet-50 + EfficientNet-B3)
Usage:
  python train.py --stage 1 [--smoke_test] [--max_cases 50]
  python train.py --stage 2
  python train.py --stage 3
  python train.py --stage 0   # run all stages sequentially
"""
import os, sys, json, argparse, time, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
# ── local imports ─────────────────────────────────────────────────────────────
import datasets as DS
import models   as M
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
from common.splits import load_splits, BAD_CASES, DEFAULT_SPLITS_CSV
# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
# ── argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="HERALD training pipeline")
    p.add_argument("--stage",      type=int, default=0,
                   help="1=liver, 2=tumor, 3=classify, 0=all")
    p.add_argument("--data_dir",   default="/scratch/teh.c/mct_training/mct_local")
    p.add_argument("--ckpt_dir",   default="/scratch/teh.c/mct_training/checkpoints")
    p.add_argument("--log_dir",    default="/scratch/teh.c/mct_training/logs")
    p.add_argument("--splits_csv", default=DEFAULT_SPLITS_CSV)
    p.add_argument("--max_cases",  type=int, default=None,
                   help="Cap dataset size (debug / smoke test)")
    p.add_argument("--smoke_test", action="store_true",
                   help="Run 2 cases, 2 epochs only")
    # training hyper-params
    p.add_argument("--img_size",   type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--epochs_seg", type=int, default=40,
                   help="Max epochs for Stage 1 & 2")
    p.add_argument("--epochs_cls", type=int, default=30,
                   help="Max epochs for Stage 3")
    p.add_argument("--patience",   type=int, default=7)
    p.add_argument("--max_cached", type=int, default=30)
    p.add_argument("--tf_heads",   type=int, default=8,
                   help="Transformer attention heads in UNetTransformer")
    p.add_argument("--tf_win",     type=int, default=4,
                   help="Window size for transformer bottleneck")
    # ── Run stages ─────────────────────────────────────────────────────────
    p.add_argument("--resume",      action="store_true",
                   help="Resume from latest checkpoint if available")
    p.add_argument("--base_channels", type=int, default=32,
                   help="Base channel width for UNetTransformer")
    p.add_argument("--no_pretrain", action="store_true",
                   help="Disable ImageNet pretraining for Stage 3 backbones")
    p.add_argument("--cls_backbone", default="ensemble",
                   choices=["resnet", "efficientnet", "ensemble",
                            "swin_tiny", "swin_base"],
                   help="Stage 3 classifier: resnet | efficientnet | ensemble"
                        " | swin_tiny | swin_base")

    return p.parse_args()


# ── metric helpers ────────────────────────────────────────────────────────────

def dice_score(pred_logits, target, thresh=0.5):
    pred = (torch.sigmoid(pred_logits) > thresh).float()
    inter = (pred * target).sum()
    return (2 * inter / (pred.sum() + target.sum() + 1e-6)).item()


def class_weights(labels_df, cls2idx, device):
    counts = np.zeros(len(cls2idx))
    for _, row in labels_df.iterrows():
        lbl = str(row["type"] if "type" in labels_df.columns else row.iloc[1])
        if lbl in cls2idx:
            counts[cls2idx[lbl]] += 1
    w = 1.0 / (counts + 1e-6)
    w = w / w.sum() * len(counts)
    return torch.tensor(w, dtype=torch.float32).to(device)


# ── Stage 1 & 2 training loop ─────────────────────────────────────────────────

def train_seg_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    losses, dices = [], []
    for imgs, msks in loader:
        imgs, msks = imgs.to(device), msks.to(device)
        optimizer.zero_grad()
        with autocast(device_type="cuda"):
            main, dem_aux, sem_aux = model(imgs.float())
            loss = criterion(main, dem_aux, sem_aux, msks.float())
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        losses.append(loss.item())
        with torch.no_grad():
            dices.append(dice_score(main, msks.float()))
    return float(np.mean(losses)), float(np.mean(dices))


@torch.no_grad()
def val_seg_epoch(model, loader, criterion, device):
    model.eval()
    losses, dices = [], []
    for imgs, msks in loader:
        imgs, msks = imgs.to(device), msks.to(device)
        with autocast(device_type="cuda"):
            main, dem_aux, sem_aux = model(imgs.float())
            loss = criterion(main, dem_aux, sem_aux, msks.float())
        losses.append(loss.item())
        dices.append(dice_score(main, msks.float()))
    return float(np.mean(losses)), float(np.mean(dices))


def run_seg_stage(stage_num, train_ids, val_ids, args, device, log):
    tag      = "liver" if stage_num == 1 else "tumor"
    ckpt     = os.path.join(args.ckpt_dir, f"unet_v2_{tag}_best.pth")
    ckpt_latest = os.path.join(args.ckpt_dir, f"unet_v2_{tag}_latest.pth")
    ckpt_meta   = os.path.join(args.ckpt_dir, f"unet_v2_{tag}_meta.json")
    DatasetCls = DS.LiverDataset if stage_num == 1 else DS.TumorDataset

    print(f"\n{'='*60}")
    print(f"  Stage {stage_num} — {tag.capitalize()} Segmentation")
    print(f"{'='*60}")

    train_ds = DatasetCls(train_ids, augment=True)
    val_ds   = DatasetCls(val_ids,   augment=False)

    sampler     = DS.CaseGroupedSampler(train_ds)
    train_loader= DataLoader(train_ds, batch_size=args.batch_size,
                             sampler=sampler, num_workers=0, pin_memory=True)
    val_loader  = DataLoader(val_ds,   batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    model   = M.UNetTransformer(in_ch=12, out_ch=1, img_size=args.img_size,
                                tf_heads=args.tf_heads, tf_win=args.tf_win,
                                base=args.base_channels).to(device)

    criterion = M.DeepSupervisionLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_seg)
    scaler    = GradScaler("cuda")

    # ── Resume from latest checkpoint if requested ──
    start_epoch = 1
    best_dice   = -1.0
    no_imp      = 0
    if args.resume and os.path.exists(ckpt_latest):
        ck = torch.load(ckpt_latest, map_location=device, weights_only=True)
        if os.path.exists(ckpt_meta):
            with open(ckpt_meta) as f:
                meta        = json.load(f)
                start_epoch = meta.get("epoch", 0) + 1
                best_dice   = meta.get("best_dice", -1.0)
                no_imp      = meta.get("no_imp", 0)
        if "model" in ck:
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            scaler.load_state_dict(ck["scaler"])
        else:
            # legacy checkpoint: weights only — fast-forward the LR schedule
            model.load_state_dict(ck)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(start_epoch - 1):
                    scheduler.step()
            print("  WARNING: legacy checkpoint — optimizer/scaler state "
                  "not restored, LR schedule fast-forwarded")
        print(f"  Resumed from epoch {start_epoch - 1} "
              f"(best dice so far: {best_dice:.4f}, "
              f"lr {optimizer.param_groups[0]['lr']:.2e})")
    elif stage_num == 2:
        s1_ckpt = os.path.join(args.ckpt_dir, "unet_v2_liver_best.pth")
        if os.path.exists(s1_ckpt):
            sd = torch.load(s1_ckpt, map_location=device, weights_only=True)
            # load only encoder weights (enc1-4 + bottleneck) — skip head
            enc_sd = {k: v for k, v in sd.items()
                      if any(k.startswith(p) for p in
                             ("enc1","enc2","enc3","enc4","bottleneck"))}
            # both stages take the same 12-ch (4 phases x 3 slices) input
            missing, unexpected = model.load_state_dict(enc_sd, strict=False)
            print(f"  Warm-started encoder from Stage 1 "
                  f"({len(enc_sd)} tensors loaded, "
                  f"{len(missing)} missing, {len(unexpected)} unexpected)")
        else:
            print("  WARNING: Stage 1 checkpoint not found; training from scratch")

    history = []
    for ep in range(start_epoch, args.epochs_seg + 1):
        t0                   = time.time()
        tr_loss, tr_dice     = train_seg_epoch(model, train_loader, optimizer,
                                               scaler, criterion, device)
        vl_loss, vl_dice     = val_seg_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        row = dict(stage=stage_num, epoch=ep,
                   tr_loss=round(tr_loss,4), tr_dice=round(tr_dice,4),
                   vl_loss=round(vl_loss,4), vl_dice=round(vl_dice,4))
        history.append(row)
        print(f"Ep {ep:3d}/{args.epochs_seg} | "
              f"tr_loss {tr_loss:.4f} tr_dice {tr_dice:.4f} | "
              f"vl_loss {vl_loss:.4f} vl_dice {vl_dice:.4f} | {elapsed:.0f}s",
              flush=True)

        if vl_dice > best_dice:
            best_dice = vl_dice
            no_imp    = 0
            torch.save(model.state_dict(), ckpt)
            print(f"  ✓ Saved  best val {tag} Dice: {best_dice:.4f}")
        else:
            no_imp += 1

        # save latest every epoch (after the best/no_imp update, so the meta
        # matches this epoch) so we can resume after a SLURM timeout
        torch.save({"model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler":    scaler.state_dict()}, ckpt_latest)
        with open(ckpt_meta, "w") as f:
            json.dump({"epoch": ep, "best_dice": best_dice,
                       "no_imp": no_imp}, f)

        if no_imp >= args.patience:
            print(f"  Early stopping at epoch {ep} "
                  f"(no improvement for {args.patience} epochs)")
            break

    log[f"stage{stage_num}"] = {"best_val_dice": best_dice, "history": history}
    print(f"\nStage {stage_num} complete — best {tag} Dice: {best_dice:.4f}")
    return best_dice


# ── Stage 3 training loop ─────────────────────────────────────────────────────

def train_cls_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    losses, preds_all, labels_all = [], [], []
    for slice_x, vol_x, labels in loader:
        slice_x = slice_x.to(device)
        vol_x   = vol_x.to(device)
        labels  = labels.to(device)
        optimizer.zero_grad()
        with autocast("cuda"):
            logits = model(slice_x.float(), vol_x.float())
            loss   = criterion(logits, labels)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        losses.append(loss.item())
        preds_all.extend(logits.argmax(1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())
    acc = accuracy_score(labels_all, preds_all)
    return float(np.mean(losses)), acc


@torch.no_grad()
def val_cls_epoch(model, loader, device):
    model.eval()
    preds_all, labels_all = [], []
    for slice_x, vol_x, labels in loader:
        slice_x = slice_x.to(device)
        vol_x   = vol_x.to(device)
        logits  = model(slice_x.float(), vol_x.float())
        preds_all.extend(logits.argmax(1).cpu().tolist())
        labels_all.extend(labels.cpu().tolist())
    return accuracy_score(labels_all, preds_all), preds_all, labels_all


def run_cls_stage(train_ids, val_ids, test_ids, labels_df,
                  args, device, log):
    print(f"\n{'='*60}")
    print(f"  Stage 3 — Tumor Classification")
    print(f"{'='*60}")

    ckpt_map = {
        "resnet":      "resnet_cls_best.pth",
        "efficientnet":"efficientnet_cls_best.pth",
        "ensemble":    "ensemble_cls_best.pth",
        "swin_tiny":   "swin_tiny_cls_best.pth",
        "swin_base":   "swin_base_cls_best.pth",
    }
    ckpt = os.path.join(args.ckpt_dir, ckpt_map[args.cls_backbone])

    print(f"  Backbone: {args.cls_backbone.upper()}")

    train_ds = DS.ClassificationDataset(train_ids, labels_df, augment=True)
    val_ds   = DS.ClassificationDataset(val_ids,   labels_df, augment=False)
    test_ds  = DS.ClassificationDataset(test_ids,  labels_df, augment=False)

    cls2idx    = train_ds.cls2idx
    num_classes= len(cls2idx)
    idx2cls    = {v: k for k, v in cls2idx.items()}

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=4, shuffle=False,
                              num_workers=0, pin_memory=True)

    backbone_cls = {
        "resnet":       lambda **kw: M.ResNetClassifier(**kw),
        "efficientnet": lambda **kw: M.EfficientNetClassifier(**kw),
        "ensemble":     lambda **kw: M.EnsembleClassifier(**kw),
        "swin_tiny":    lambda **kw: M.SwinClassifier("swin_tiny", **kw),
        "swin_base":    lambda **kw: M.SwinClassifier("swin_base", **kw),
    }[args.cls_backbone]
    model = backbone_cls(
        num_classes=num_classes, proj_dim=256,
        pretrained=not args.no_pretrain).to(device)

    cw       = class_weights(labels_df, cls2idx, device)
    criterion= nn.CrossEntropyLoss(weight=cw)
    optimizer= torch.optim.AdamW(model.parameters(),
                                 lr=args.lr * 0.5, weight_decay=1e-4)
    scheduler= torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_cls)
    scaler   = GradScaler("cuda")

    best_acc, no_imp = -1.0, 0
    history = []
    for ep in range(1, args.epochs_cls + 1):
        t0                   = time.time()
        tr_loss, tr_acc      = train_cls_epoch(model, train_loader, optimizer,
                                               scaler, criterion, device)
        vl_acc, _, _         = val_cls_epoch(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        row = dict(epoch=ep, tr_loss=round(tr_loss,4),
                   tr_acc=round(tr_acc,4), vl_acc=round(vl_acc,4))
        history.append(row)
        print(f"Ep {ep:3d}/{args.epochs_cls} | "
              f"loss {tr_loss:.4f} | tr_acc {tr_acc:.3f} | "
              f"val_acc {vl_acc:.3f} | {elapsed:.0f}s", flush=True)

        if vl_acc > best_acc:
            best_acc = vl_acc
            no_imp   = 0
            torch.save(model.state_dict(), ckpt)
            print(f"  ✓ Saved  best val acc: {best_acc:.4f}")
        else:
            no_imp += 1
            if no_imp >= args.patience:
                print(f"  Early stopping at epoch {ep}")
                break

    # ── Test evaluation ────────────────────────────────────────────────────
    print("\n── Test set evaluation ──")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    test_acc, test_preds, test_labels = val_cls_epoch(model, test_loader, device)
    class_names = [idx2cls[i] for i in range(num_classes)]
    print(f"Test accuracy: {test_acc:.4f}")
    present_labels = sorted(set(test_labels))
    present_names  = [idx2cls[i] for i in present_labels]
    print(classification_report(test_labels, test_preds,
                                labels=present_labels,
                                target_names=present_names,
                                zero_division=0))
    cm = confusion_matrix(test_labels, test_preds).tolist()

    log["stage3"] = {
        "best_val_acc": best_acc,
        "test_acc":     test_acc,
        "class_names":  class_names,
        "confusion_matrix": cm,
        "history":      history,
    }
    print(f"\nStage 3 complete — best val acc: {best_acc:.4f} | "
          f"test acc: {test_acc:.4f}")
    return test_acc


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # propagate args into datasets module globals
    DS.LOCAL_DATA_DIR = args.data_dir
    DS.SPLITS_CSV     = args.splits_csv
    DS.IMG_SIZE       = args.img_size
    DS.MAX_CACHED     = args.max_cached

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir,  exist_ok=True)

    train_ids, val_ids, test_ids, labels_df = load_splits(args.splits_csv)

    # ── Smoke test / max_cases cap ─────────────────────────────────────────
    if args.smoke_test:
        args.max_cases  = 6
        args.epochs_seg = 2
        args.epochs_cls = 2
        args.patience   = 2
        print("SMOKE TEST — 6 cases, 2 epochs")

    if args.max_cases:
        train_ids = train_ids[:args.max_cases]
        val_ids   = val_ids[:max(2, args.max_cases // 5)]
        test_ids  = test_ids[:max(2, args.max_cases // 5)]
        print(f"Capped — train: {len(train_ids)}, val: {len(val_ids)}, "
              f"test: {len(test_ids)}")

    # ── Run stages ─────────────────────────────────────────────────────────
    log      = {}
    run_all  = (args.stage == 0)

    if run_all or args.stage == 1:
        run_seg_stage(1, train_ids, val_ids, args, device, log)

    if run_all or args.stage == 2:
        run_seg_stage(2, train_ids, val_ids, args, device, log)

    if run_all or args.stage == 3:
        run_cls_stage(train_ids, val_ids, test_ids, labels_df,
                      args, device, log)

    # ── Save run log ───────────────────────────────────────────────────────
    log_path = os.path.join(args.log_dir, "herald_results.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nResults saved → {log_path}")


if __name__ == "__main__":
    main()