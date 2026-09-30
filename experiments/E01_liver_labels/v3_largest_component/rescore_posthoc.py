"""Re-score saved Stage 1 predictions with largest-connected-component post-processing."""
import os, sys
import numpy as np, pandas as pd, nibabel as nib
from scipy.ndimage import label

sys.path.insert(0, os.path.expanduser("~/LiverTumorClassification"))
from common.splits import load_splits
from common.metrics import volume_metrics, liver_extras

run, out_dir = sys.argv[1], sys.argv[2]
_, _, test, labels = load_splits()
types = dict(zip(labels["case_id"], labels["type"].astype(str)))
D = os.environ["HERALD_DATA"]
R = os.path.join(os.environ["HERALD_RESULTS"], "runs", "ds2net", run)


def largest_cc(m):
    lab, n = label(m)
    if n <= 1:
        return m
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return lab == sizes.argmax()


rows = []
for cid in test:
    prob = np.load(f"{R}/liver_masks/{cid}_liver_prob.npy").astype(np.float32)
    pred = largest_cc(prob > 0.5)
    liv = np.asanyarray(nib.load(f"{D}/{cid}/liver_mask.nii.gz").dataobj) > 0
    tum = np.asanyarray(nib.load(f"{D}/{cid}/tumor_mask.nii.gz").dataobj) > 0
    ref = liv | tum        # this run's label (liver_includes_tumor=True)
    rows.append({"case_id": cid, "tumor_type": types.get(cid),
                 **volume_metrics(pred, ref), **liver_extras(pred, liv, tum)})
os.makedirs(out_dir, exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(out_dir, "per_case_test.csv"), index=False)
print(f"wrote {len(rows)} cases to {out_dir}/per_case_test.csv")
