"""E01 v4 — which predicted liver components to keep?

Re-scores E01 v2's saved liver probabilities (no retraining) under several
component rules, for every case of every split:

  all          small-component filter only (≥ 1000 voxels) = v2 as evaluated
  largest      largest component only                       = v3 (adopted)
  size_f       also keep components ≥ f × largest's volume  (f = 0.05, 0.1, 0.2)
  near_Dmm     also keep components within D mm of the largest (D = 5, 10, 20)

Reference: liver ∪ tumor (v2's label). Output: one per_case_<split>.csv per
rule under <out_dir>/<rule>/, readable by common/evaluate.py.

python rescore_components.py <run_name> <out_dir> [workers]
"""
import os, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np, pandas as pd, nibabel as nib
from scipy.ndimage import label, distance_transform_edt

sys.path.insert(0, os.path.expanduser("~/LiverTumorClassification"))
from common.splits import load_splits
from common.metrics import volume_metrics, liver_extras

RULES = ["all", "largest", "size_0.05", "size_0.1", "size_0.2",
         "near_5mm", "near_10mm", "near_20mm"]
D = os.environ["HERALD_DATA"]


def apply_rules(pred, spacing):
    lab, n = label(pred)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    keep_cc = sizes >= 1000                      # the evaluation's small-component filter
    keep_cc[0] = False
    base = keep_cc[lab]
    out = {"all": base}
    if not base.any():
        return {r: base for r in RULES}
    big = sizes.argmax()
    out["largest"] = lab == big
    for f in (0.05, 0.1, 0.2):
        out[f"size_{f}"] = (keep_cc & (sizes >= f * sizes[big]))[lab]
    # distance (mm) from every voxel to the largest component
    dist = distance_transform_edt(lab != big, sampling=spacing)
    comps = np.unique(lab[base])
    min_d = {c: dist[lab == c].min() for c in comps}
    for d in (5, 10, 20):
        ok = np.zeros(len(sizes), bool)
        ok[[c for c in comps if min_d[c] <= d]] = True
        out[f"near_{d}mm"] = ok[lab]
    return out


def score_case(args):
    cid, run_dir, ctype = args
    prob = np.load(f"{run_dir}/liver_masks/{cid}_liver_prob.npy").astype(np.float32)
    img = nib.load(f"{D}/{cid}/liver_mask.nii.gz")
    liv = np.asanyarray(img.dataobj) > 0
    tum = np.asanyarray(nib.load(f"{D}/{cid}/tumor_mask.nii.gz").dataobj) > 0
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    ref = liv | tum
    rows = {}
    for rule, pred in apply_rules(prob > 0.5, spacing).items():
        rows[rule] = {"case_id": cid, "tumor_type": ctype,
                      **volume_metrics(pred, ref), **liver_extras(pred, liv, tum)}
    return rows


def main():
    run, out_dir = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    run_dir = os.path.join(os.environ["HERALD_RESULTS"], "runs", "ds2net", run)
    train, val, test, labels = load_splits()
    types = dict(zip(labels["case_id"], labels["type"].astype(str)))
    for split, ids in (("val", val), ("train", train), ("test", test)):
        with ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(score_case, [(c, run_dir, types.get(c)) for c in ids]))
        for rule in RULES:
            os.makedirs(f"{out_dir}/{rule}", exist_ok=True)
            pd.DataFrame([r[rule] for r in results]).to_csv(
                f"{out_dir}/{rule}/per_case_{split}.csv", index=False)
        print(f"{split}: {len(ids)} cases scored under {len(RULES)} rules", flush=True)


if __name__ == "__main__":
    main()
