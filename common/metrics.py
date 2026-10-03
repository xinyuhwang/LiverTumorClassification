"""
metrics.py — per-case volume metrics shared by every pipeline, so that
per_case_test.csv files from ds2net/ and unet_hybrid/ are directly comparable
(see common/evaluate.py).
"""

import numpy as np
from scipy.ndimage import label

CONN26 = np.ones((3, 3, 3), bool)      # lesions: 26-connected components


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


def lesion_metrics(pred, gt, voxel_ml):
    """Per-lesion detection (R1). Lesions are 26-connected components.
    A GT lesion is detected if any predicted voxel overlaps it; a predicted
    component that overlaps no GT lesion is a false positive. Returns
    (case summary dict, list of per-GT-lesion dicts with volume, overlap
    fraction and lesion Dice against the predicted components touching it)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    gl, ng = label(gt, CONN26)
    pl, npred = label(pred, CONN26)
    g_size = np.bincount(gl.ravel(), minlength=ng + 1)
    p_size = np.bincount(pl.ravel(), minlength=npred + 1)
    both = gt & pred
    pairs = np.zeros((ng + 1, npred + 1), np.int64)          # overlap voxels per (gt, pred)
    np.add.at(pairs, (gl[both], pl[both]), 1)
    lesions = []
    for k in range(1, ng + 1):
        hit = np.nonzero(pairs[k])[0]
        tp = int(pairs[k].sum())
        lesions.append({"lesion": k, "gt_ml": float(g_size[k]) * voxel_ml,
                        "detected": bool(len(hit)), "overlap": tp / max(int(g_size[k]), 1),
                        "lesion_Dice": 2 * tp / max(int(g_size[k] + p_size[hit].sum()), 1)})
    fp = int((pairs[1:, 1:].sum(0) == 0).sum()) if npred else 0
    n_det = sum(l["detected"] for l in lesions)
    summary = {"n_gt_lesions": ng, "n_detected": n_det, "n_pred_lesions": npred,
               "n_fp_lesions": fp, "lesion_recall": n_det / ng if ng else float("nan")}
    return summary, lesions
