"""
review_lesion_fragments.py — contact sheets of small GT tumor components for a
visual check (E02 v6 follow-up): are the tiny "lesions" real satellite tumors
or annotation specks?

For each case, every 26-connected GT tumor component below --max_ml gets one
panel: the PVP slice (liver window) where the component is largest, cropped to
a --crop_mm square around it, with
  red     the small component
  yellow  the rest of the GT tumor
  cyan    the predicted tumor (label 2), if --pred_dir is given
Title: component volume (ml / voxels), slices it spans, distance to the
nearest other GT tumor voxel (mm), and whether the prediction touches it.

python review_lesion_fragments.py --cases 230525a4 231025c16 --out review/ \
       [--pred_dir <nnU-Net label maps>] [--max_ml 0.1]
Writes <out>/<case>.png and <out>/fragments.csv.
"""
import argparse, os
from pathlib import Path

import numpy as np, pandas as pd, nibabel as nib
from scipy.ndimage import label, distance_transform_edt, find_objects
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONN26 = np.ones((3, 3, 3), bool)
LIVER_WINDOW = (-150, 250)


def contour(ax, mask, color):
    if mask.any():
        ax.contour(mask.T, levels=[0.5], colors=[color], linewidths=1.0)


def review_case(cid, data_dir, pred_dir, max_ml, crop_mm, max_panels):
    case = Path(data_dir) / cid
    img = nib.load(case / "pvp.nii.gz")
    pvp = img.get_fdata(dtype=np.float32)
    tumor = np.asanyarray(nib.load(case / "tumor_mask.nii.gz").dataobj) > 0
    pred = (np.asanyarray(nib.load(Path(pred_dir) / f"{cid}.nii.gz").dataobj) == 2
            if pred_dir else None)
    sp = np.array(img.header.get_zooms()[:3], float)
    vox_ml = float(np.prod(sp)) / 1000
    lab, n = label(tumor, CONN26)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    small = [k for k in range(1, n + 1) if sizes[k] * vox_ml < max_ml]
    rows = []
    half = [int(round(crop_mm / 2 / s)) for s in sp[:2]]
    panels = []
    for k in small:
        comp = lab == k
        rest = tumor & ~comp
        dist = (distance_transform_edt(~rest, sampling=sp)[comp].min()
                if rest.any() else float("nan"))
        zs = np.nonzero(comp.any(axis=(0, 1)))[0]
        z = int(zs[np.argmax([comp[:, :, z].sum() for z in zs])])
        cy, cx = [int(round(c.mean())) for c in np.nonzero(comp[:, :, z])]
        hit = bool(pred is not None and (pred & comp).any())
        rows.append({"case_id": cid, "component": k, "voxels": int(sizes[k]),
                     "ml": sizes[k] * vox_ml, "n_slices": len(zs), "slice": z,
                     "dist_to_other_tumor_mm": dist, "predicted": hit})
        panels.append((k, z, cy, cx))
    big = [k for k in range(1, n + 1) if k not in small]
    if panels:
        panels = panels[:max_panels]
        cols = min(6, len(panels)); r = int(np.ceil(len(panels) / cols))
        fig, axes = plt.subplots(r, cols, figsize=(3 * cols, 3.2 * r), squeeze=False)
        lo, hi = LIVER_WINDOW
        for ax, (k, z, cy, cx), row in zip(axes.ravel(), panels, rows):
            sl = (slice(max(cy - half[0], 0), cy + half[0]), slice(max(cx - half[1], 0), cx + half[1]), z)
            ax.imshow(np.clip(pvp[sl], lo, hi).T, cmap="gray", origin="lower")
            contour(ax, (lab == k)[sl], "red")
            contour(ax, (tumor & (lab != k))[sl], "yellow")
            if pred is not None:
                contour(ax, pred[sl], "cyan")
            d = row["dist_to_other_tumor_mm"]
            ax.set_title(f"#{k}: {row['ml']:.3f} ml ({row['voxels']} vox), {row['n_slices']} sl\n"
                         f"z={z}, {d:.0f} mm to tumor, pred {'yes' if row['predicted'] else 'no'}",
                         fontsize=7)
            ax.axis("off")
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        fig.suptitle(f"{cid}: {len(small)} GT components < {max_ml} ml, "
                     f"{len(big)} larger\nred = small component, yellow = other GT tumor"
                     f"{', cyan = prediction' if pred is not None else ''}", fontsize=9)
        fig.tight_layout()
    else:
        fig = None
    return rows, fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--data_dir", default=os.environ.get("HERALD_DATA"))
    p.add_argument("--pred_dir", default=None)
    p.add_argument("--max_ml", type=float, default=0.1)
    p.add_argument("--crop_mm", type=float, default=60)
    p.add_argument("--max_panels", type=int, default=24)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cid in args.cases:
        r, fig = review_case(cid, args.data_dir, args.pred_dir, args.max_ml, args.crop_mm, args.max_panels)
        rows += r
        if fig is not None:
            fig.savefig(out / f"{cid}.png", dpi=110); plt.close(fig)
        print(f"{cid}: {len(r)} small components")
    pd.DataFrame(rows).to_csv(out / "fragments.csv", index=False)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
