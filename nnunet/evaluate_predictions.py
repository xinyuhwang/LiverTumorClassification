"""
evaluate_predictions.py — score nnU-Net label-map predictions with HERALD's
shared metric code (E06), so they compare directly with ds2net/unet_hybrid.

Writes, for the given split:
  <out>/tumor/per_case_<split>.csv   tumor (label 2) vs GT tumor
  <out>/liver/per_case_<split>.csv   liver (labels 1+2) vs GT liver ∪ tumor,
                                      plus Dice_vs_union and tumor_covered
Both include gt_ml / pred_ml (volumes in ml) for size-stratified summaries.

python evaluate_predictions.py --pred <dir of <case>.nii.gz> --split val --out <dir>
"""
import argparse, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np, pandas as pd, nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.splits import load_splits, DEFAULT_SPLITS_CSV
from common.metrics import volume_metrics, liver_extras


def score(args):
    cid, pred_dir, data_dir, ctype = args
    case = Path(data_dir) / cid
    img = nib.load(case / "liver_mask.nii.gz")
    liver = np.asanyarray(img.dataobj) > 0
    tumor = np.asanyarray(nib.load(case / "tumor_mask.nii.gz").dataobj) > 0
    pred = np.asanyarray(nib.load(Path(pred_dir) / f"{cid}.nii.gz").dataobj)
    if pred.shape != liver.shape:
        raise ValueError(f"{cid}: prediction {pred.shape} vs label {liver.shape}")
    vox_ml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
    p_t, p_l = pred == 2, pred >= 1
    ref_l = liver | tumor
    t = {"case_id": cid, "tumor_type": ctype, **volume_metrics(p_t, tumor),
         "gt_ml": tumor.sum() * vox_ml, "pred_ml": p_t.sum() * vox_ml}
    l = {"case_id": cid, "tumor_type": ctype, **volume_metrics(p_l, ref_l),
         **liver_extras(p_l, liver, tumor), "gt_ml": ref_l.sum() * vox_ml,
         "pred_ml": p_l.sum() * vox_ml}
    return t, l


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", required=True, help="Directory of <case>.nii.gz label maps")
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data_dir", default=os.environ.get("HERALD_DATA"))
    p.add_argument("--splits_csv", default=DEFAULT_SPLITS_CSV)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    _, val, test, labels = load_splits(args.splits_csv)
    types = dict(zip(labels["case_id"], labels["type"].astype(str)))
    ids = {"val": val, "test": test}[args.split]
    missing = [c for c in ids if not (Path(args.pred) / f"{c}.nii.gz").exists()]
    if missing:
        sys.exit(f"{len(missing)} predictions missing in {args.pred}, e.g. {missing[:3]}")
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(score, [(c, args.pred, args.data_dir, types.get(c)) for c in ids]))
    for i, task in enumerate(("tumor", "liver")):
        out = Path(args.out) / task
        out.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([r[i] for r in res])
        df.to_csv(out / f"per_case_{args.split}.csv", index=False)
        print(f"{task} {args.split}: {len(df)} cases, mean Dice {df['Dice'].mean():.4f} "
              f"→ {out / f'per_case_{args.split}.csv'}")


if __name__ == "__main__":
    main()
