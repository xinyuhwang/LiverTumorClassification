"""
metrics.py — per-case volume metrics shared by every pipeline, so that
per_case_test.csv files from ds2net/ and unet_hybrid/ are directly comparable
(see common/evaluate.py).
"""

import numpy as np
from scipy.ndimage import label


def largest_component(mask):
    """Keep only the largest 3-D connected component (the liver is one organ).
    E01 v3: removes spurious blobs far from the liver, but also drops tumor
    predicted as a separate piece (231025c23)."""
    lab, n = label(mask)
    if n <= 1:
        return mask.astype(bool)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return lab == sizes.argmax()


def volume_metrics(pred, gt):
    """Dice, IoU, precision, recall of two boolean volumes. Empty-vs-empty
    counts as perfect agreement."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    pr = tp / (tp + fp) if tp + fp else float(fn == 0)
    rc = tp / (tp + fn) if tp + fn else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
    return {"Dice": dice, "IoU": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            "Precision": pr, "Recall": rc, "pred_voxels": int(pred.sum()),
            "gt_voxels": int(gt.sum())}


def liver_extras(pred, liver, tumor):
    """E01's common reference for liver-label variants: Dice against
    liver ∪ tumor, and the fraction of GT tumor inside the predicted liver."""
    pred, tumor = pred.astype(bool), tumor.astype(bool)
    union = liver.astype(bool) | tumor
    return {"Dice_vs_union": volume_metrics(pred, union)["Dice"],
            "tumor_covered": float((pred & tumor).sum()) / max(int(tumor.sum()), 1)}
