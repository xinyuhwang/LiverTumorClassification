"""Score a saved Stage 1 run on any split, with the evaluation's small-component
filter (as trained) and with largest-component post-processing (E01 v3).
Used to confirm the v3 decision on VALIDATION cases (the decision was first
made on test, which should only report).

python rescore_split.py <run_name> <split: val|test> <out_dir>
Writes <out_dir>/cc1000/per_case_test.csv and <out_dir>/largest/per_case_test.csv
(file name kept so common/evaluate.py reads them).
"""
import os, sys
import numpy as np, pandas as pd, nibabel as nib
from scipy.ndimage import label

sys.path.insert(0, os.path.expanduser("~/LiverTumorClassification"))
from common.splits import load_splits
from common.metrics import volume_metrics, liver_extras, largest_component

run, split, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
train, val, test, labels = load_splits()
cases = {"val": val, "test": test}[split]
types = dict(zip(labels["case_id"], labels["type"].astype(str)))
D = os.environ["HERALD_DATA"]
R = os.path.join(os.environ["HERALD_RESULTS"], "runs", "ds2net", run)


def cc_filter(m, n=1000):               # ds2net/train.py min_component_voxels
    lab, _ = label(m)
    s = np.bincount(lab.ravel()); s[0] = 0
    return (s >= n)[lab]


rows = {"cc1000": [], "largest": []}
for cid in cases:
    prob = np.load(f"{R}/liver_masks/{cid}_liver_prob.npy").astype(np.float32)
    liv = np.asanyarray(nib.load(f"{D}/{cid}/liver_mask.nii.gz").dataobj) > 0
    tum = np.asanyarray(nib.load(f"{D}/{cid}/tumor_mask.nii.gz").dataobj) > 0
    ref = liv | tum
    base = cc_filter(prob > 0.5)
    for name, pred in (("cc1000", base), ("largest", largest_component(base))):
        rows[name].append({"case_id": cid, "tumor_type": types.get(cid),
                           **volume_metrics(pred, ref), **liver_extras(pred, liv, tum)})
for name, r in rows.items():
    os.makedirs(f"{out_dir}/{name}", exist_ok=True)
    pd.DataFrame(r).to_csv(f"{out_dir}/{name}/per_case_test.csv", index=False)
print(f"{run} {split}: {len(cases)} cases → {out_dir}/{{cc1000,largest}}")
