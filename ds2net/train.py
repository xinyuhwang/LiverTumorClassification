"""
train.py — DS²Net entry point
  Stage 1: liver segmentation   (DS2NetLiver, 20-ch 2.5D, 4 phases × 5 slices)
  Stage 2: tumor segmentation   (DS2NetUNet, 12-ch 2.5D, 4 phases × 3 slices)
  Stage 3: tumor classification (EfficientNet4Phase on GT tumor ROIs)

Usage
  python train.py --stage 1 --run_name liver_v1
  python train.py --stage 2 --run_name tumor_v1 --set slices=liver
  python train.py --stage 2 --run_name tumor_z5 --set z_spacing=5.0 samples_per_epoch=59314
  python train.py --stage 1 --run_name liver_v1 --eval_only --save_liver_masks
  python train.py --stage 3 --smoke_test

Outputs go to <work_dir>/runs/ds2net/<run_name>/:
  config.json, history.json, best.pth, last.pth (resume state),
  metrics.json (slice-level + per-case test metrics), per_case_test.csv
"""

import argparse, ast, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from scipy.ndimage import label as cc_label, zoom
from skimage.transform import resize as sk_resize
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             cohen_kappa_score, confusion_matrix)
from sklearn.preprocessing import label_binarize

import config as C
import datasets as DS
import models as M
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root
from common.splits import load_splits, DEFAULT_SPLITS_CSV
from common.metrics import volume_metrics, liver_extras, largest_component

SEED = 42
METRIC_KEYS = ["Dice", "IoU", "Precision", "Recall", "F1", "MAE"]


def parse_args():
    p = argparse.ArgumentParser(description="DS²Net training pipeline")
    p.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--run_name", default=None, help="default: stage<N>")
    p.add_argument("--data_dir", default=os.environ.get("HERALD_DATA", "data/mct_ltdiag"))
    p.add_argument("--work_dir", default=os.environ.get("HERALD_WORK", "work"),
                   help="Caches and run outputs")
    p.add_argument("--splits_csv", default=DEFAULT_SPLITS_CSV)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Override config values, e.g. --set slices=liver epochs=30")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training; evaluate best.pth of this run")
    p.add_argument("--save_liver_masks", action="store_true",
                   help="Stage 1: write predicted liver masks for every case")
    p.add_argument("--no_pretrain", action="store_true")
    p.add_argument("--workers", type=int, default=8, help="Preprocessing processes")
    p.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    p.add_argument("--max_cases", type=int, default=None)
    p.add_argument("--smoke_test", action="store_true",
                   help="4 train / 2 val / 2 test cases, 2 epochs")
    return p.parse_args()


def resolve_config(args):
    cfg = dict(C.STAGES[args.stage])
    for kv in args.set:
        k, v = kv.split("=", 1)
        if k not in cfg:
            raise KeyError(f"Unknown config key '{k}' for stage {args.stage}")
        try:
            cfg[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            cfg[k] = v
    if args.smoke_test:
        cfg.update(epochs=2, val_every=1, batch_size=2)
        if "grad_accum_steps" in cfg:
            cfg["grad_accum_steps"] = 1
        if cfg.get("samples_per_epoch"):
            cfg["samples_per_epoch"] = None
    return cfg


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def amp_setup(device):
    if device.type != "cuda":
        return False, torch.float32, None
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda") if dtype == torch.float16 else None
    return True, dtype, scaler


# ── Segmentation metrics ──────────────────────────────────────────────────────

def slice_metrics(probs, gt, thr):
    """Batch-mean of per-slice metrics (as in the notebooks)."""
    pred = (probs > thr).float()
    tp = (pred * gt).sum((-1, -2)); fp = (pred * (1 - gt)).sum((-1, -2))
    fn = ((1 - pred) * gt).sum((-1, -2))
    pr = ((tp + 1e-6) / (tp + fp + 1e-6)).mean().item()
    rc = ((tp + 1e-6) / (tp + fn + 1e-6)).mean().item()
    return {"Dice": ((2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)).mean().item(),
            "IoU": ((tp + 1e-6) / (tp + fp + fn + 1e-6)).mean().item(),
            "Precision": pr, "Recall": rc, "F1": 2 * pr * rc / (pr + rc + 1e-6),
            "MAE": (probs - gt).abs().mean().item()}


def cc_filter(mask, min_voxels):
    if min_voxels <= 0 or not mask.any():
        return mask
    lab, _ = cc_label(mask)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return (sizes >= min_voxels)[lab]


# ── Stage 1 / 2 ───────────────────────────────────────────────────────────────

def build_seg_model(stage, cfg, pretrained):
    if stage == 1:
        return M.DS2NetLiver(n_input_channels=4 * cfg["n_context_slices"],
                             mha_heads=cfg["mha_heads"], mha_dropout=cfg["mha_dropout"],
                             pretrained=pretrained)
    return M.DS2NetUNet(n_input_channels=4 * cfg["n_context_slices"], pretrained=pretrained)


def seg_forward(model, x, stage):
    """Returns (list of head logits, stb logits or None)."""
    out = model(x)
    return (out, None) if stage == 1 else out


def seg_loss(stage, cfg, preds, stb, masks, pos_weight):
    return M.ds2_adaptive_loss([p.float() for p in preds], masks, pos_weight,
                               cfg["boundary_weight"],
                               stb_logits=stb, stb_weight=cfg.get("stb_weight", 0.4))


def seg_probs(stage, cfg, preds, stb):
    return M.fuse_probs(preds, stb, cfg.get("stb_infer_weight", 0.3))


@torch.no_grad()
def evaluate_slices(model, loader, stage, cfg, device, amp, amp_dtype):
    model.eval()
    agg, vloss, nb = {k: 0. for k in METRIC_KEYS}, 0., 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
            preds, stb = seg_forward(model, imgs, stage)
        loss = seg_loss(stage, cfg, preds, stb, masks, cfg["pos_weight"]).item()
        vloss += loss if math.isfinite(loss) else 0.
        for k, v in slice_metrics(seg_probs(stage, cfg, preds, stb), masks,
                                  cfg["seg_threshold"]).items():
            agg[k] += v
        nb += 1
    return vloss / max(nb, 1), {k: v / max(nb, 1) for k, v in agg.items()}


TTA = [(lambda x: x, lambda p: p),
       (lambda x: torch.flip(x, [-1]), lambda p: torch.flip(p, [-1])),
       (lambda x: torch.flip(x, [-2]), lambda p: torch.flip(p, [-2])),
       (lambda x: torch.rot90(x, 1, [-2, -1]), lambda p: torch.rot90(p, -1, [-2, -1])),
       (lambda x: torch.rot90(x, 2, [-2, -1]), lambda p: torch.rot90(p, -2, [-2, -1])),
       (lambda x: torch.rot90(x, 3, [-2, -1]), lambda p: torch.rot90(p, -3, [-2, -1])),
       (lambda x: torch.rot90(torch.flip(x, [-1]), 1, [-2, -1]),
        lambda p: torch.flip(torch.rot90(p, -1, [-2, -1]), [-1])),
       (lambda x: torch.rot90(torch.flip(x, [-2]), 1, [-2, -1]),
        lambda p: torch.flip(torch.rot90(p, -1, [-2, -1]), [-2]))]


def tta_probs(model, x, stage, cfg, amp, amp_dtype):
    acc = 0
    for fwd, inv in TTA:
        with torch.autocast(x.device.type, dtype=amp_dtype, enabled=amp):
            preds, stb = seg_forward(model, fwd(x), stage)
        acc = acc + inv(seg_probs(stage, cfg, preds, stb))
    return (acc / len(TTA))[:, 0].float().cpu().numpy()


def predicted_liver(liver_dir, case_id, cfg):
    """Stage 1 liver mask as adopted in E01: threshold 0.5, small-component
    filter, largest component. On the case's original grid."""
    prob = np.load(Path(liver_dir) / f"{case_id}_liver_prob.npy").astype(np.float32)
    return largest_component(cc_filter(prob > 0.5, 1000))


@torch.no_grad()
def predict_case(model, case_dir, stage, cfg, device, amp, amp_dtype, batch=16,
                 roi_mask=None):
    """8-fold TTA → probability volume on the original grid.
    Whole-slice mode (default): every slice.
    Liver-ROI mode (cfg roi="liver"): only slices where roi_mask has liver,
    each cropped to its square liver box; 0 outside the box. roi_mask is on
    the original grid; None means the GT liver(+tumor) mask."""
    model.eval()
    S, n_ctx = cfg["img_size"], cfg["n_context_slices"]
    if cfg.get("roi", "none") != "liver":
        phases_rs, _, rs_shape, orig_shape, _ = DS.build_volume(case_dir, cfg)
        D = phases_rs[0].shape[0]
        probs = np.zeros((D, S, S), np.float32)
        for s in range(0, D, batch):
            x = torch.from_numpy(np.stack([DS.stack_context(phases_rs, z, n_ctx)
                                           for z in range(s, min(s + batch, D))])).to(device)
            probs[s:s + len(x)] = tta_probs(model, x, stage, cfg, amp, amp_dtype)
        return DS.to_original_grid(probs, rs_shape, orig_shape)

    vols_rs, masks_rs, orig_shape, _, factors = DS.load_resampled(case_dir, cfg)
    if roi_mask is None:
        roi_rs = masks_rs["liver"].astype(bool)
    else:
        roi_rs = (zoom(roi_mask.astype(np.uint8), factors, order=0) if factors
                  else roi_mask.astype(np.uint8)).astype(bool)
        roi_rs = roi_rs[:masks_rs["liver"].shape[0], :masks_rs["liver"].shape[1],
                        :masks_rs["liver"].shape[2]]
    canvas = np.zeros(masks_rs["liver"].shape, np.float32)
    zs = [z for z in range(canvas.shape[2]) if roi_rs[:, :, z].any()]
    margin = DS.roi_margin_px(cfg)
    for i in range(0, len(zs), batch):
        chunk = zs[i:i + batch]
        boxes = [DS.roi_box(roi_rs[:, :, z], margin) for z in chunk]
        x = torch.from_numpy(np.stack([DS.roi_stack(vols_rs, z, n_ctx, b, S)
                                       for z, b in zip(chunk, boxes)])).to(device)
        for z, b, p in zip(chunk, boxes, tta_probs(model, x, stage, cfg, amp, amp_dtype)):
            side = b[1] - b[0]
            DS.paste_square(canvas[:, :, z],
                            sk_resize(p, (side, side), order=1, preserve_range=True), b)
    if canvas.shape != tuple(orig_shape):
        canvas = zoom(canvas, tuple(o / r for o, r in zip(orig_shape, canvas.shape)), order=1)
    return canvas


def run_seg_stage(args, cfg, splits, device, run_dir, case_types):
    stage = args.stage
    train_ids, val_ids, test_ids = splits
    amp, amp_dtype, scaler = amp_setup(device)
    print(f"AMP: {amp} ({amp_dtype}) | GradScaler: {scaler is not None}")

    cache_dir, ok = DS.build_cache(train_ids + val_ids + test_ids, args.data_dir,
                                   Path(args.work_dir) / "cache" / "ds2net", cfg, args.workers)
    ok = set(ok)
    tr = [c for c in train_ids if c in ok]
    va = [c for c in val_ids if c in ok]
    te = [c for c in test_ids if c in ok]
    train_ds = DS.SliceDataset(cache_dir, tr, cfg, augment=True, case_types=case_types)
    val_ds   = DS.SliceDataset(cache_dir, va, cfg)
    test_ds  = DS.SliceDataset(cache_dir, te, cfg)
    kw = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda",
              persistent_workers=args.num_workers > 0)
    sampler = WeightedRandomSampler(torch.tensor(train_ds.slice_weights, dtype=torch.double),
                                    num_samples=cfg.get("samples_per_epoch") or len(train_ds),
                                    replacement=True)
    train_loader = DataLoader(train_ds, cfg["batch_size"], sampler=sampler, **kw)
    val_loader   = DataLoader(val_ds, cfg["batch_size"], shuffle=True, **kw)
    test_loader  = DataLoader(test_ds, cfg["batch_size"], shuffle=False, **kw)

    model = build_seg_model(stage, cfg, not args.no_pretrain).to(device)
    print(f"Model: {type(model).__name__} "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    accum  = cfg["grad_accum_steps"]
    spe    = math.ceil(len(train_loader) / accum)     # optimizer steps per epoch
    optim  = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched  = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=cfg["lr"], epochs=cfg["epochs"], steps_per_epoch=spe,
        pct_start=cfg["pct_start"], div_factor=10., final_div_factor=1e4,
        anneal_strategy="cos")

    best_path, last_path = run_dir / "best.pth", run_dir / "last.pth"
    state = {"epoch": 0, "best": -1.0, "no_improve": 0, "history": []}
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); optim.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        if scaler and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        print(f"Resumed at epoch {state['epoch']} (best val Dice {state['best']:.4f})")

    warmup_epochs = max(1, int(cfg["epochs"] * cfg["pct_start"]))
    if not args.eval_only:
        for epoch in range(state["epoch"] + 1, cfg["epochs"] + 1):
            pos_w = cfg["pos_weight"] + (max(0., 1 - epoch / warmup_epochs)
                                         * cfg.get("pos_weight_warmup_extra", 0.))
            model.train(); optim.zero_grad(set_to_none=True)
            t0, ep_loss, ep_dice, n_steps = time.time(), 0., 0., 0
            for step, (imgs, masks) in enumerate(train_loader):
                imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
                    preds, stb = seg_forward(model, imgs, stage)
                loss = seg_loss(stage, cfg, preds, stb, masks, pos_w) / accum
                if not torch.isfinite(loss) or loss.item() * accum > 1000:
                    print(f"  skipping step {step}: loss {loss.item() * accum:.3e}")
                    optim.zero_grad(set_to_none=True)
                    continue
                (scaler.scale(loss) if scaler else loss).backward()
                if (step + 1) % accum == 0 or step + 1 == len(train_loader):
                    if scaler:
                        scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                    if scaler:
                        scaler.step(optim); scaler.update()
                    else:
                        optim.step()
                    if sched.last_epoch + 1 < sched.total_steps:
                        sched.step()
                    optim.zero_grad(set_to_none=True)
                ep_loss += loss.item() * accum; n_steps += 1
                with torch.no_grad():
                    ep_dice += slice_metrics(seg_probs(stage, cfg, preds, stb).float(),
                                             masks, cfg["seg_threshold"])["Dice"]
            row = {"epoch": epoch, "loss": ep_loss / max(n_steps, 1),
                   "train_dice": ep_dice / max(n_steps, 1), "lr": optim.param_groups[0]["lr"],
                   "pos_weight": pos_w}
            msg = ""
            if epoch % cfg["val_every"] == 0 or epoch == cfg["epochs"]:
                vloss, vm = evaluate_slices(model, val_loader, stage, cfg, device, amp, amp_dtype)
                row.update(val_loss=vloss, **{f"val_{k}": v for k, v in vm.items()})
                if vm["Dice"] > state["best"]:
                    state["best"], state["no_improve"] = vm["Dice"], 0
                    torch.save(model.state_dict(), best_path)
                    msg = f"  ✓ best val Dice {vm['Dice']:.4f}"
                else:
                    state["no_improve"] += 1
                msg = f" | val_loss {vloss:.4f} val_dice {vm['Dice']:.4f}" + msg
            state["history"].append(row); state["epoch"] = epoch
            print(f"Ep {epoch:3d}/{cfg['epochs']} | loss {row['loss']:.4f} | "
                  f"train_dice {row['train_dice']:.4f} | lr {row['lr']:.2e}{msg} | "
                  f"{time.time() - t0:.0f}s", flush=True)
            torch.save({"model": model.state_dict(), "optimizer": optim.state_dict(),
                        "scheduler": sched.state_dict(),
                        "scaler": scaler.state_dict() if scaler else None,
                        "state": state}, last_path)
            (run_dir / "history.json").write_text(json.dumps(state["history"], indent=1))
            if state["no_improve"] >= cfg["early_stop_patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    # ── Test evaluation ──
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    _, slice_m = evaluate_slices(model, test_loader, stage, cfg, device, amp, amp_dtype)
    print("\nTest (slice-level, cached slices only): "
          + " ".join(f"{k} {v:.4f}" for k, v in slice_m.items()))

    task = cfg["task"]
    roi_mode = cfg.get("roi", "none") == "liver"
    liver_run = cfg.get("roi_liver_run")
    liver_dir = (Path(args.work_dir) / "runs" / "ds2net" / liver_run / "liver_masks"
                 if liver_run else None)
    # Evaluation modes: Stage 1 and whole-slice Stage 2 have one. Liver-ROI
    # Stage 2 has "oracle" (GT liver+tumor box) and, with roi_liver_run,
    # "cascade" (box from that Stage 1 run's adopted liver masks).
    modes = ["full"] if not roi_mode else (["oracle"] + (["cascade"] if liver_dir else []))
    primary = modes[-1]
    rows = {(sp, m): [] for sp in ("val", "test") for m in modes}
    liver_out = run_dir / "liver_masks"
    save_liver = stage == 1 and args.save_liver_masks
    predict_ids = (tr + va + te) if save_liver else (va + te)
    if save_liver:
        liver_out.mkdir(exist_ok=True)
    for i, cid in enumerate(predict_ids, 1):
        split = "test" if cid in te else "val" if cid in va else None
        _, masks, _ = DS.load_case(Path(args.data_dir) / cid,
                                   cfg.get("liver_includes_tumor", False))
        for mode in modes:
            roi_mask = predicted_liver(liver_dir, cid, cfg) if mode == "cascade" else None
            prob = predict_case(model, Path(args.data_dir) / cid, stage, cfg, device,
                                amp, amp_dtype, roi_mask=roi_mask)
            pred = cc_filter(prob > cfg["seg_threshold"], cfg["min_component_voxels"])
            if cfg.get("keep_largest_component", False):
                pred = largest_component(pred)
            if save_liver:
                np.save(liver_out / f"{cid}_liver_prob.npy", prob.astype(np.float16))
            if split:
                row = {"case_id": cid, "tumor_type": case_types.get(cid),
                       **volume_metrics(pred, masks[task].astype(bool))}
                if task == "liver":
                    # common reference for comparing label variants (E01)
                    row.update(liver_extras(pred, masks["liver"], masks["tumor"]))
                rows[(split, mode)].append(row)
        print(f"  [{i}/{len(predict_ids)}] {cid}", flush=True)

    vol_keys = ["Dice", "IoU", "Precision", "Recall"]
    if task == "liver":
        vol_keys += ["Dice_vs_union", "tumor_covered"]
    metrics = {"stage": stage, "task": task, "best_val_dice": state["best"],
               "test_slice_level": slice_m, "eval_modes": modes, "primary_mode": primary,
               "roi_liver_run": liver_run}
    for (split, mode), r in rows.items():
        df = pd.DataFrame(r)
        name = f"per_case_{split}.csv" if mode == primary else f"per_case_{split}_{mode}.csv"
        df.to_csv(run_dir / name, index=False)
        key = f"{split}_{mode}"
        metrics[key] = {"file": name, "n_cases": len(df),
                        "per_case_mean": df[vol_keys].mean().to_dict(),
                        "per_case_std": df[vol_keys].std().to_dict(),
                        "per_case_by_type": df.groupby("tumor_type")[vol_keys].mean()
                                              .round(4).to_dict(orient="index")}
        print(f"\n{split} per-case ({mode}, full volume, 8-fold TTA) → {name}:")
        print(df[vol_keys].agg(["mean", "std"]).round(4).to_string())
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))


# ── Stage 3 ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def case_predictions(model, loader, device, amp, amp_dtype):
    model.eval()
    probs, labels = {}, {}
    for x, bg, y, ids in loader:
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
            logits = model(x.to(device), bg.to(device))
        for p, yi, cid in zip(torch.softmax(logits.float(), 1).cpu().numpy(), y.tolist(), ids):
            probs.setdefault(cid, []).append(p); labels[cid] = yi
    cids = sorted(probs)
    return (np.array([labels[c] for c in cids]),
            np.array([np.mean(probs[c], axis=0) for c in cids]), cids)


def cls_report(y_true, y_prob, class_names):
    y_pred = y_prob.argmax(1)
    idx = list(range(len(class_names)))
    try:
        auc = roc_auc_score(label_binarize(y_true, classes=idx), y_prob,
                            average="macro", multi_class="ovr")
    except ValueError:
        auc = float("nan")
    return {"accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "macro_auc": auc, "kappa": cohen_kappa_score(y_true, y_pred),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=idx).tolist(),
            "class_names": class_names}


def run_cls_stage(args, cfg, splits, device, run_dir, labels_df):
    train_ids, val_ids, test_ids = splits
    amp, amp_dtype, _ = amp_setup(device)
    class_names = sorted(labels_df["type"].astype(str).unique())
    labels = {c: class_names.index(t) for c, t in
              zip(labels_df["case_id"], labels_df["type"].astype(str))}
    roi_dir = DS.build_roi_cache(train_ids + val_ids + test_ids, args.data_dir,
                                 Path(args.work_dir) / "cache" / "ds2net", cfg, args.workers)
    train_ds = DS.TumorPatchDataset(roi_dir, train_ids, labels, augment=True)
    val_ds   = DS.TumorPatchDataset(roi_dir, val_ids, labels)
    test_ds  = DS.TumorPatchDataset(roi_dir, test_ids, labels)
    kw = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_ds, cfg["batch_size"], shuffle=True, **kw)
    val_loader   = DataLoader(val_ds, cfg["batch_size"], **kw)
    test_loader  = DataLoader(test_ds, cfg["batch_size"], **kw)

    # class weights ∝ 1 / patches per class (as in the notebook)
    counts = np.bincount([it[2] for it in train_ds.items], minlength=len(class_names))
    w = 1.0 / np.maximum(counts, 1); w = w / w.sum() * len(w)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device),
                                    label_smoothing=cfg["label_smoothing"])
    model = M.EfficientNet4Phase(len(class_names), dropout=cfg["dropout"],
                                 use_curve=cfg["use_curve_features"],
                                 curve_embed_dim=cfg["curve_embed_dim"],
                                 pretrained=not args.no_pretrain).to(device)
    best_path = run_dir / "best.pth"

    def make_opt(unfrozen):
        head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
        if not unfrozen:
            o = torch.optim.AdamW(head, lr=cfg["lr"], weight_decay=1e-4)
            s = torch.optim.lr_scheduler.OneCycleLR(
                o, max_lr=cfg["lr"], epochs=cfg["epochs"], steps_per_epoch=len(train_loader),
                pct_start=0.20, div_factor=10.0, final_div_factor=1e4, anneal_strategy="cos")
        else:
            o = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": cfg["lr"] / 10},
                                   {"params": head, "lr": cfg["lr"] / 5}], weight_decay=1e-4)
            s = torch.optim.lr_scheduler.OneCycleLR(
                o, max_lr=[cfg["lr"] / 10, cfg["lr"] / 5],
                epochs=cfg["epochs"] - cfg["unfreeze_epoch"] + 1,
                steps_per_epoch=len(train_loader), pct_start=0.10, div_factor=5.0,
                final_div_factor=1e3, anneal_strategy="cos")
        return o, s

    optim, sched = make_opt(False)
    scaler = torch.amp.GradScaler("cuda") if amp and amp_dtype == torch.float16 else None
    best, no_imp, history = -1.0, 0, []
    for epoch in range(1, (0 if args.eval_only else cfg["epochs"]) + 1):
        if epoch == cfg["unfreeze_epoch"]:
            model.freeze_backbone(False)
            optim, sched = make_opt(True)
            print(f"  Epoch {epoch}: backbone unfrozen")
        model.train(); t0, losses = time.time(), []
        for x, bg, y, _ in train_loader:
            x, bg, y = x.to(device), bg.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
                loss = criterion(model(x, bg).float(), y)
            (scaler.scale(loss) if scaler else loss).backward()
            if scaler:
                scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler:
                scaler.step(optim); scaler.update()
            else:
                optim.step()
            if sched.last_epoch + 1 < sched.total_steps:
                sched.step()
            losses.append(loss.item())
        yv, pv, _ = case_predictions(model, val_loader, device, amp, amp_dtype)
        acc = accuracy_score(yv, pv.argmax(1))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_acc": acc,
                        "val_macro_f1": f1_score(yv, pv.argmax(1), average="macro")})
        flag = ""
        if acc > best:
            best, no_imp, flag = acc, 0, f"  ✓ best val acc {acc:.4f}"
            torch.save(model.state_dict(), best_path)
        else:
            no_imp += 1
        print(f"Ep {epoch:3d}/{cfg['epochs']} | loss {np.mean(losses):.4f} | "
              f"val_acc {acc:.4f}{flag} | {time.time() - t0:.0f}s", flush=True)
        (run_dir / "history.json").write_text(json.dumps(history, indent=1))
        if no_imp >= cfg["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    metrics = {"stage": 3, "best_val_acc": best}
    rows = []
    for split, loader in (("val", val_loader), ("test", test_loader)):
        y, prob, cids = case_predictions(model, loader, device, amp, amp_dtype)
        metrics[split] = cls_report(y, prob, class_names)
        print(f"\n{split}: acc {metrics[split]['accuracy']:.4f} | "
              f"macro-F1 {metrics[split]['macro_f1']:.4f} | "
              f"macro-AUC {metrics[split]['macro_auc']:.4f}")
        if split == "test":
            rows = [{"case_id": c, "true": class_names[t], "pred": class_names[p.argmax()],
                     **{f"p_{n}": float(v) for n, v in zip(class_names, p)}}
                    for c, t, p in zip(cids, y, prob)]
    pd.DataFrame(rows).to_csv(run_dir / "per_case_test.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))


def main():
    args = parse_args()
    set_seed(SEED)
    cfg = resolve_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    run_dir = Path(args.work_dir) / "runs" / "ds2net" / (args.run_name or f"stage{args.stage}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | run: {run_dir}")
    (run_dir / "config.json").write_text(json.dumps(
        {"args": vars(args), "config": cfg}, indent=1, default=str))

    train_ids, val_ids, test_ids, labels_df = load_splits(args.splits_csv)
    if args.smoke_test:
        train_ids, val_ids, test_ids = train_ids[:4], val_ids[:2], test_ids[:2]
    elif args.max_cases:
        train_ids = train_ids[:args.max_cases]
        val_ids, test_ids = val_ids[:max(2, args.max_cases // 5)], test_ids[:max(2, args.max_cases // 5)]
    case_types = dict(zip(labels_df["case_id"], labels_df["type"].astype(str)))
    splits = (train_ids, val_ids, test_ids)

    if args.stage in (1, 2):
        run_seg_stage(args, cfg, splits, device, run_dir, case_types)
    else:
        run_cls_stage(args, cfg, splits, device, run_dir, labels_df)
    print(f"\nOutputs → {run_dir}")


if __name__ == "__main__":
    main()
