"""
prepare_mct_ltdiag.py — turn the raw MCT-LTDiag archives into the per-case
layout read by ds2net/ and unet_hybrid/, and QA every case.

Raw archive (<raw>/<case>.tar)          Prepared (<out>/<case>/)
  NIFTI/nc.nii.gz                   →    phase_0.nii.gz   non-contrast
  NIFTI/art.nii.gz                  →    phase_1.nii.gz   arterial
  NIFTI/pvp.nii.gz                  →    pvp.nii.gz  (+ phase_2.nii.gz symlink)
  NIFTI/delay.nii.gz                →    phase_3.nii.gz   delayed
  mask_pvp.nii.gz                   →    tumor_mask.nii.gz  (tumor only, {0,1})
  liver_mask_pvp.nii.gz             →    liver_mask.nii.gz  (official liver mask)
  DICOM/                                 not extracted

Also copies the metadata tables and writes <out>/manifest.csv with, per case:
shapes, spacing, grid agreement across files, mask labels, liver/tumor volume,
the fraction of tumor inside the liver mask, and a list of QA flags.

Usage
  python prepare_mct_ltdiag.py --raw <raw_dir> --out <data_dir> [--workers 8]
"""

import argparse, os, shutil, tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

MEMBERS = {
    "NIFTI/nc.nii.gz":       "phase_0.nii.gz",
    "NIFTI/art.nii.gz":      "phase_1.nii.gz",
    "NIFTI/pvp.nii.gz":      "pvp.nii.gz",
    "NIFTI/delay.nii.gz":    "phase_3.nii.gz",
    "mask_pvp.nii.gz":       "tumor_mask.nii.gz",
    "liver_mask_pvp.nii.gz": "liver_mask.nii.gz",
}
OUTPUTS = list(MEMBERS.values()) + ["phase_2.nii.gz"]


def extract_case(tar_path, case_dir):
    case_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    with tarfile.open(tar_path) as tf:
        members = {m.name.lstrip("./"): m for m in tf.getmembers() if m.isfile()}
        for src, dst in MEMBERS.items():
            m = members.get(src)
            if m is None:
                missing.append(src)
                continue
            with tf.extractfile(m) as fin, open(case_dir / (dst + ".part"), "wb") as fout:
                shutil.copyfileobj(fin, fout, 1 << 20)
            os.replace(case_dir / (dst + ".part"), case_dir / dst)
    link = case_dir / "phase_2.nii.gz"
    if not link.exists() and (case_dir / "pvp.nii.gz").exists():
        link.symlink_to("pvp.nii.gz")
    return missing


def qa_case(case_dir):
    row, flags = {}, []
    pvp = nib.load(case_dir / "pvp.nii.gz")
    row["shape"]   = "x".join(map(str, pvp.shape))
    row["spacing"] = "x".join(f"{z:.3f}" for z in pvp.header.get_zooms()[:3])
    slice_mm = float(pvp.header.get_zooms()[2])
    max_shift = 0.0
    for name in ("phase_0", "phase_1", "phase_3", "tumor_mask", "liver_mask"):
        path = case_dir / f"{name}.nii.gz"
        if not path.exists():
            flags.append(f"missing_{name}")
            continue
        img = nib.load(path)
        if img.shape != pvp.shape:
            flags.append(f"shape_{name}")
        elif not np.allclose(img.affine[:3, :3], pvp.affine[:3, :3], atol=1e-3):
            flags.append(f"orientation_{name}")
        else:
            # Phases are stored on the PVP voxel grid; header origins can differ
            # by a few mm. Record it, and flag only shifts of a slice or more.
            shift = float(np.abs(img.affine[:3, 3] - pvp.affine[:3, 3]).max())
            max_shift = max(max_shift, shift)
            if shift >= slice_mm:
                flags.append(f"origin_shift_{name}")
    row["max_origin_shift_mm"] = round(max_shift, 2)

    vox_ml = float(np.prod(pvp.header.get_zooms()[:3])) / 1000.0
    masks = {}
    for name in ("tumor_mask", "liver_mask"):
        path = case_dir / f"{name}.nii.gz"
        if path.exists():
            a = np.asanyarray(nib.load(path).dataobj)
            labels = np.unique(a).tolist()
            row[f"{name}_labels"] = " ".join(map(str, labels))
            if not set(labels) <= {0, 1}:
                flags.append(f"labels_{name}")
            masks[name] = a > 0
            row[f"{name.split('_')[0]}_ml"] = round(masks[name].sum() * vox_ml, 2)
    t, l = masks.get("tumor_mask"), masks.get("liver_mask")
    if t is not None and l is not None and t.shape == l.shape:
        row["tumor_in_liver"] = round(float((t & l).sum() / max(t.sum(), 1)), 4)
        if t.sum() == 0:
            flags.append("empty_tumor")
        if l.sum() == 0:
            flags.append("empty_liver")
        if row["tumor_in_liver"] < 0.9:
            flags.append("tumor_outside_liver")
    row["flags"] = " ".join(flags)
    return row


def process(tar_path, out_dir, overwrite):
    case_id  = Path(tar_path).stem
    case_dir = Path(out_dir) / case_id
    row = {"case_id": case_id}
    try:
        if overwrite or not all((case_dir / o).exists() for o in OUTPUTS):
            missing = extract_case(tar_path, case_dir)
            if missing:
                row["flags"] = "missing_in_tar:" + ",".join(missing)
                return row
        row.update(qa_case(case_dir))
    except Exception as e:                                   # noqa: BLE001
        row["flags"] = f"ERROR:{type(e).__name__}:{e}"
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True, help="Directory with <case>.tar archives")
    p.add_argument("--out", required=True, help="Prepared data directory")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for meta in ("meta_info_patient", "meta_info_tumor"):
        for src in raw.glob(f"{meta}.*"):
            shutil.copy2(src, out / src.name)

    tars = sorted(raw.glob("*.tar"))
    print(f"Preparing {len(tars)} cases → {out}")
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [ex.submit(process, t, out, args.overwrite) for t in tars]
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            print(f"[{i}/{len(tars)}] {row['case_id']} {row.get('flags') or 'ok'}",
                  flush=True)

    manifest = pd.DataFrame(rows).sort_values("case_id")
    manifest.to_csv(out / "manifest.csv", index=False)
    flagged = manifest[manifest["flags"].fillna("") != ""]
    print(f"\nWrote {out / 'manifest.csv'} — {len(manifest)} cases, "
          f"{len(flagged)} flagged")
    if len(flagged):
        print(flagged[["case_id", "flags"]].to_string(index=False))


if __name__ == "__main__":
    main()
