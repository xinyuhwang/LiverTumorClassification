"""
summarize_manifest.py — QA summary of a prepared dataset's manifest.csv
(written by prepare_mct_ltdiag.py): flag counts, liver/tumor volume
distributions, tumor-inside-liver fraction, and outlier cases.

Usage
  python summarize_manifest.py <data_dir>/manifest.csv [--out summary.json]
"""

import argparse, json
import pandas as pd

# plausible adult liver volume range on CT (ml); outside it → review the mask
LIVER_ML_RANGE = (700, 3500)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    m = pd.read_csv(args.manifest, dtype={"case_id": str})
    m["flags"] = m["flags"].fillna("")
    flag_counts = (m["flags"].str.split().explode().dropna()
                   .loc[lambda s: s != ""].value_counts().to_dict())
    liver_out = m[(m["liver_ml"] < LIVER_ML_RANGE[0]) | (m["liver_ml"] > LIVER_ML_RANGE[1])]
    tumor_out = m[m["tumor_in_liver"] < 0.99]
    q = [0.0, 0.05, 0.5, 0.95, 1.0]
    summary = {
        "n_cases": len(m),
        "n_flagged": int((m["flags"] != "").sum()),
        "flag_counts": flag_counts,
        "liver_ml_quantiles": m["liver_ml"].quantile(q).round(1).to_dict(),
        "tumor_ml_quantiles": m["tumor_ml"].quantile(q).round(2).to_dict(),
        "tumor_in_liver_quantiles": m["tumor_in_liver"].quantile(q).round(4).to_dict(),
        "max_origin_shift_mm_quantiles": m["max_origin_shift_mm"].quantile(q).round(2).to_dict(),
        "liver_volume_outliers": liver_out[["case_id", "liver_ml"]].to_dict(orient="records"),
        "tumor_outside_liver_lt_99pct": tumor_out[["case_id", "tumor_in_liver"]]
                                           .to_dict(orient="records"),
    }
    text = json.dumps(summary, indent=1, default=str)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
