"""
prepare_dataset.py — convert the prepared MCT-LTDiag data into an nnU-Net v2
raw dataset, using HERALD's shared split (E06).

  <nnUNet_raw>/Dataset<ID>_MCTLTDiag/
    imagesTr/<case>_000{0..3}.nii.gz   train + val cases (symlinks): nc, art, pvp, delay
    labelsTr/<case>.nii.gz             0 background, 1 liver (official liver ∪ tumor), 2 tumor
    imagesTs/<case>_000{0..3}.nii.gz   test cases (symlinks), predicted once at the end
    labelsTs/<case>.nii.gz             test labels, for our own evaluation only
    imagesVal/<case>_000{0..3}.nii.gz  val cases again (symlinks), for ensemble prediction
    dataset.json
  <nnUNet_preprocessed>/Dataset<ID>_MCTLTDiag/splits_final.json
    fold 0: our 360 train / 78 val cases (E06 v1; nnU-Net keeps an existing file)
    folds 1..K (--cv_folds K): inner CV over the 360 train cases (E06 v2),
    from common.splits.cv_folds; val and test stay out of every fold

All files share the PVP voxel grid (data_prep/prepare_mct_ltdiag.py). Labels
are written with the PVP header so nnU-Net's geometry checks pass.

python prepare_dataset.py --data_dir $HERALD_DATA --raw $nnUNet_raw \
                          --preprocessed $nnUNet_preprocessed [--dataset_id 501]
python prepare_dataset.py --cv_folds 5 --splits_only      # E06 v2: splits + imagesVal
"""
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.splits import load_splits, cv_folds, DEFAULT_SPLITS_CSV

CHANNELS = ["phase_0", "phase_1", "pvp", "phase_3"]          # nc, art, pvp, delay
CHANNEL_NAMES = {"0": "CT", "1": "CT", "2": "CT", "3": "CT"}  # CT normalisation per channel
LABELS = {"background": 0, "liver": 1, "tumor": 2}


def link(src, dst):
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(Path(src).resolve())


def make_case(args):
    cid, data_dir, img_dir, lab_dir = args
    case = Path(data_dir) / cid
    for i, ch in enumerate(CHANNELS):
        link(case / f"{ch}.nii.gz", Path(img_dir) / f"{cid}_{i:04d}.nii.gz")
    pvp = nib.load(case / "pvp.nii.gz")
    liver = np.asanyarray(nib.load(case / "liver_mask.nii.gz").dataobj) > 0
    tumor = np.asanyarray(nib.load(case / "tumor_mask.nii.gz").dataobj) > 0
    lab = np.zeros(liver.shape, np.uint8)
    lab[liver | tumor] = 1          # E01: liver label includes the tumor
    lab[tumor] = 2
    hdr = pvp.header.copy(); hdr.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(lab, pvp.affine, hdr), Path(lab_dir) / f"{cid}.nii.gz")
    return cid


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=os.environ.get("HERALD_DATA"))
    p.add_argument("--raw", default=os.environ.get("nnUNet_raw"))
    p.add_argument("--preprocessed", default=os.environ.get("nnUNet_preprocessed"))
    p.add_argument("--splits_csv", default=DEFAULT_SPLITS_CSV)
    p.add_argument("--dataset_id", type=int, default=501)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cv_folds", type=int, default=0,
                   help="Add K inner CV folds over the train cases as folds 1..K (E06 v2)")
    p.add_argument("--splits_only", action="store_true",
                   help="Only write splits_final.json and imagesVal/ (raw data already built)")
    args = p.parse_args()

    name = f"Dataset{args.dataset_id:03d}_MCTLTDiag"
    root = Path(args.raw) / name
    dirs = {k: root / k for k in ("imagesTr", "labelsTr", "imagesTs", "labelsTs", "imagesVal")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    train, val, test, labels = load_splits(args.splits_csv)
    if not args.splits_only:
        jobs = ([(c, args.data_dir, dirs["imagesTr"], dirs["labelsTr"]) for c in train + val]
                + [(c, args.data_dir, dirs["imagesTs"], dirs["labelsTs"]) for c in test])
        with ProcessPoolExecutor(args.workers) as ex:
            done = list(ex.map(make_case, jobs))
        print(f"{name}: {len(train) + len(val)} training cases, {len(test)} test cases "
              f"({len(done)} written)")

        (root / "dataset.json").write_text(json.dumps({
            "channel_names": CHANNEL_NAMES, "labels": LABELS,
            "numTraining": len(train) + len(val), "file_ending": ".nii.gz",
            "name": "MCT-LTDiag", "description": "HERALD E06: 4-phase CT, liver (incl. tumor) + tumor",
            "reference": "doi:10.7910/DVN/S3RW15", "licence": "CC0 1.0"}, indent=1))
    for c in val:
        for i, ch in enumerate(CHANNELS):
            link(Path(args.data_dir) / c / f"{ch}.nii.gz", dirs["imagesVal"] / f"{c}_{i:04d}.nii.gz")
    print(f"imagesVal: {len(val)} val cases (symlinks)")

    pre = Path(args.preprocessed) / name
    pre.mkdir(parents=True, exist_ok=True)
    splits = [{"train": train, "val": val}] + (cv_folds(train, labels, args.cv_folds)
                                               if args.cv_folds else [])
    (pre / "splits_final.json").write_text(json.dumps(splits, indent=1))
    print(f"splits_final.json: fold 0 = {len(train)} train / {len(val)} val"
          + "".join(f"; fold {i} = {len(s['train'])} / {len(s['val'])}"
                    for i, s in enumerate(splits[1:], 1)) + f" → {pre}")


if __name__ == "__main__":
    main()
