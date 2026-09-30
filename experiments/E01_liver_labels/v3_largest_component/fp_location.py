"""Where are Stage 1's false-positive liver voxels? (E01 v1/v2 test cases)"""
import os, sys
import numpy as np, pandas as pd, nibabel as nib
from scipy.ndimage import label

sys.path.insert(0, os.path.expanduser("~/LiverTumorClassification"))
from common.splits import load_splits

_, _, test, _ = load_splits()
D = os.environ["HERALD_DATA"]
R = os.path.join(os.environ["HERALD_RESULTS"], "runs", "ds2net")


def cc(m, n=1000):                      # same filter as ds2net evaluation
    lab, _ = label(m)
    s = np.bincount(lab.ravel()); s[0] = 0
    return (s >= n)[lab]


rows = []
for run in ("e01_v1_liver", "e01_v2_liver_union"):
    for cid in test:
        prob = np.load(f"{R}/{run}/liver_masks/{cid}_liver_prob.npy").astype(np.float32)
        pred = cc(prob > 0.5)
        liv = np.asanyarray(nib.load(f"{D}/{cid}/liver_mask.nii.gz").dataobj) > 0
        tum = np.asanyarray(nib.load(f"{D}/{cid}/tumor_mask.nii.gz").dataobj) > 0
        ref = liv | tum                          # common reference for both runs
        fp = pred & ~ref
        zs = np.where(ref.any(axis=(0, 1)))[0]
        z0, z1 = zs.min(), zs.max()
        per_z = fp.sum(axis=(0, 1)); idx = np.arange(len(per_z))
        in_ext = (idx >= z0) & (idx <= z1)
        near = ((idx >= z0 - 3) & (idx < z0)) | ((idx > z1) & (idx <= z1 + 3))
        rows.append({"run": run, "case": cid, "fp": int(fp.sum()), "ref": int(ref.sum()),
                     "fp_liver_slices": int(per_z[in_ext].sum()),
                     "fp_adjacent_1_3": int(per_z[near].sum()),
                     "fp_far": int(per_z[~in_ext & ~near].sum()),
                     "n_components": int(label(pred)[1])})

df = pd.DataFrame(rows)
for run, g in df.groupby("run"):
    tot = g["fp"].sum()
    print(f"{run}: FP = {tot / g['ref'].sum():.3f} of reference volume | "
          f"on liver slices {g['fp_liver_slices'].sum() / tot:.1%}, "
          f"1-3 slices outside {g['fp_adjacent_1_3'].sum() / tot:.1%}, "
          f"further {g['fp_far'].sum() / tot:.1%} | "
          f"cases with >1 predicted component: {int((g['n_components'] > 1).sum())}")
w = (df[df["run"] == "e01_v2_liver_union"].assign(fp_rel=lambda x: x["fp"] / x["ref"])
     .sort_values("fp_rel", ascending=False).head(6))
print(w[["case", "fp_rel", "fp_liver_slices", "fp_adjacent_1_3", "fp_far", "n_components"]]
      .to_string(index=False))
df.to_csv(os.path.join(os.environ["HERALD_RESULTS"], "e01_fp_location.csv"), index=False)
