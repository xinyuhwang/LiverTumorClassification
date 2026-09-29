"""
prepare_mct_ltdiag.py — turn the raw MCT-LTDiag archives into the per-case
layout read by ds2net/ and unet_hybrid/, put every file on the PVP voxel grid,
and QA every case.

Raw archive (<raw>/<case>.tar)          Prepared (<out>/<case>/)
  NIFTI/nc.nii.gz                   →    phase_0.nii.gz   non-contrast
  NIFTI/art.nii.gz                  →    phase_1.nii.gz   arterial
  NIFTI/pvp.nii.gz                  →    pvp.nii.gz  (+ phase_2.nii.gz symlink)
  NIFTI/delay.nii.gz                →    phase_3.nii.gz   delayed
  mask_pvp.nii.gz                   →    tumor_mask.nii.gz  (tumor only, {0,1})
  liver_mask_pvp.nii.gz             →    liver_mask.nii.gz  (official liver mask)
  DICOM/                                 not extracted

Alignment: both pipelines stack phases voxel-for-voxel, so every phase and mask
must share the PVP grid. Files whose shape or affine differ from the PVP are
resampled into PVP space using their NIfTI headers (linear for images, nearest
for masks). In 11 cases the phases differ from the PVP by an in-plane scale of
up to 4.5% and origin shifts of up to 9 mm; header-based resampling raised
body-mask overlap with the PVP from 0.87–0.95 to 0.92–0.98 (checked 2026-09-29).
Breathing motion between phases is NOT corrected — it is measured instead
(liver_air_frac_*).

manifest.csv, one row per case:
  shape, spacing              PVP grid
  max_origin_shift_mm, max_scale_diff   original geometry vs PVP (before alignment)
  resampled                   files resampled onto the PVP grid
  *_mask_labels, liver_ml, tumor_ml, tumor_in_liver
  liver_air_frac_<phase>      fraction of the PVP liver mask below −200 HU in
                              that phase: ≈0 when aligned; high = misregistration
  flags                       problems left after alignment

Usage
  python prepare_mct_ltdiag.py --raw <raw_dir> --out <data_dir> [--workers 8] [--overwrite]
"""

import argparse, os, shutil, tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to

MEMBERS = {
    "NIFTI/nc.nii.gz":       "phase_0.nii.gz",
    "NIFTI/art.nii.gz":      "phase_1.nii.gz",
    "NIFTI/pvp.nii.gz":      "pvp.nii.gz",
    "NIFTI/delay.nii.gz":    "phase_3.nii.gz",
    "mask_pvp.nii.gz":       "tumor_mask.nii.gz",
    "liver_mask_pvp.nii.gz": "liver_mask.nii.gz",
}
OUTPUTS = list(MEMBERS.values()) + ["phase_2.nii.gz"]
PHASES  = {"nc": "phase_0", "art": "phase_1", "pvp": "pvp", "delay": "phase_3"}
IMAGES  = ("phase_0", "phase_1", "phase_3")
MASKS   = ("tumor_mask", "liver_mask")

AIR_HU            = -200    # below this inside the liver = not liver tissue
MISALIGNED_FRAC   = 0.10    # flag a phase if >10% of the liver mask looks like air/fat
TUMOR_INSIDE_FLAG = 0.90    # flag if <90% of the tumor is inside the liver mask


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


def align_to_pvp(case_dir):
    """Resample any file not on the PVP grid into PVP space. Returns the
    original geometry differences and the list of resampled files."""
    pvp = nib.load(case_dir / "pvp.nii.gz")
    ref_scale = np.linalg.norm(pvp.affine[:3, :3], axis=0)
    info = {"max_origin_shift_mm": 0.0, "max_scale_diff": 0.0, "resampled": []}
    for name in IMAGES + MASKS:
        path = case_dir / f"{name}.nii.gz"
        img = nib.load(path)
        shift = float(np.abs(img.affine[:3, 3] - pvp.affine[:3, 3]).max())
        scale = float(np.abs(np.linalg.norm(img.affine[:3, :3], axis=0) / ref_scale - 1).max())
        info["max_origin_shift_mm"] = max(info["max_origin_shift_mm"], shift)
        info["max_scale_diff"] = max(info["max_scale_diff"], scale)
        if img.shape == pvp.shape and np.allclose(img.affine, pvp.affine, atol=1e-3):
            continue
        is_mask = name in MASKS
        out = resample_from_to(img, pvp, order=0 if is_mask else 1,
                               cval=0 if is_mask else -1024)
        data = out.get_fdata()
        data = (data > 0.5).astype(np.uint8) if is_mask else np.rint(data).astype(np.int16)
        tmp = case_dir / f"{name}.part.nii.gz"
        nib.save(nib.Nifti1Image(data, pvp.affine), tmp)
        os.replace(tmp, path)
        info["resampled"].append(name)
    info["max_origin_shift_mm"] = round(info["max_origin_shift_mm"], 2)
    info["max_scale_diff"] = round(info["max_scale_diff"], 4)
    info["resampled"] = " ".join(info["resampled"])
    return info


def qa_case(case_dir):
    row, flags = {}, []
    pvp = nib.load(case_dir / "pvp.nii.gz")
    row["shape"]   = "x".join(map(str, pvp.shape))
    row["spacing"] = "x".join(f"{z:.3f}" for z in pvp.header.get_zooms()[:3])
    vox_ml = float(np.prod(pvp.header.get_zooms()[:3])) / 1000.0

    masks = {}
    for name in MASKS:
        img = nib.load(case_dir / f"{name}.nii.gz")
        if img.shape != pvp.shape:
            flags.append(f"shape_{name}")
            continue
        a = np.asanyarray(img.dataobj)
        labels = np.unique(a).tolist()
        row[f"{name}_labels"] = " ".join(map(str, labels))
        if not set(labels) <= {0, 1}:
            flags.append(f"labels_{name}")
        masks[name] = a > 0
        row[f"{name.split('_')[0]}_ml"] = round(masks[name].sum() * vox_ml, 2)

    t, l = masks.get("tumor_mask"), masks.get("liver_mask")
    if t is not None and l is not None:
        row["tumor_in_liver"] = round(float((t & l).sum() / max(t.sum(), 1)), 4)
        if t.sum() == 0:
            flags.append("empty_tumor")
        if l.sum() == 0:
            flags.append("empty_liver")
        if row["tumor_in_liver"] < TUMOR_INSIDE_FLAG:
            flags.append("tumor_outside_liver")
    if l is not None and l.any():
        for ph, name in PHASES.items():
            vol = np.asanyarray(nib.load(case_dir / f"{name}.nii.gz").dataobj)
            if vol.shape != l.shape:
                flags.append(f"shape_{name}")
                continue
            frac = float((vol[l] < AIR_HU).mean())
            row[f"liver_air_frac_{ph}"] = round(frac, 4)
            if frac > MISALIGNED_FRAC:
                flags.append(f"misaligned_{ph}")
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
        row.update(align_to_pvp(case_dir))
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
    p.add_argument("--overwrite", action="store_true",
                   help="Re-extract every case from its archive")
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
            note = f" [resampled: {row['resampled']}]" if row.get("resampled") else ""
            print(f"[{i}/{len(tars)}] {row['case_id']} {row.get('flags') or 'ok'}{note}",
                  flush=True)

    manifest = pd.DataFrame(rows).sort_values("case_id")
    manifest.to_csv(out / "manifest.csv", index=False)
    flagged = manifest[manifest["flags"].fillna("") != ""]
    print(f"\nWrote {out / 'manifest.csv'} — {len(manifest)} cases, "
          f"{(manifest['resampled'].fillna('') != '').sum()} with resampled files, "
          f"{len(flagged)} flagged")


if __name__ == "__main__":
    main()
