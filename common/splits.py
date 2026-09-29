"""
splits.py — the single train/val/test split shared by every HERALD pipeline
(ds2net/, unet_hybrid/). Using one split keeps results comparable.
"""

from pathlib import Path
import pandas as pd

DEFAULT_SPLITS_CSV = str(Path(__file__).resolve().parent / "mct_ltdiag_split.csv")

# 4 cases with unexpected NIfTI axis ordering — exclude from all splits.
# 231109b01, 240504b27, 240504e30: liver mask slice count differs from the PVP
# (dataset audit, branch setup/data-and-env); 240504e48: reason not recorded.
BAD_CASES = {'231109b01', '240504b27', '240504e30', '240504e48'}


def load_splits(splits_csv=DEFAULT_SPLITS_CSV):
    """Return (train_ids, val_ids, test_ids, labels_df) from the splits CSV.
    labels_df has columns case_id, type."""
    splits_df  = pd.read_csv(splits_csv)
    # normalise column names (strip whitespace)
    splits_df.columns = splits_df.columns.str.strip()

    # Expected columns: case_id, type, [age_bin], split
    id_col    = "ID"     if "ID"     in splits_df.columns else splits_df.columns[0]
    split_col = "split"  if "split"  in splits_df.columns else splits_df.columns[-1]
    type_col  = "type"   if "type"   in splits_df.columns else splits_df.columns[1]

    n_dup = splits_df[id_col].duplicated().sum()
    if n_dup:
        print(f"  WARNING: dropping {n_dup} duplicate case row(s) from splits")
        splits_df = splits_df.drop_duplicates(subset=id_col, keep="first")

    def ids(split):
        return [i for i in splits_df[splits_df[split_col] == split][id_col]
                                    .astype(str).tolist()
                if i not in BAD_CASES]

    train_ids, val_ids, test_ids = ids("train"), ids("val"), ids("test")
    labels_df = splits_df[[id_col, type_col]].rename(
        columns={id_col: "case_id", type_col: "type"})
    labels_df["case_id"] = labels_df["case_id"].astype(str)

    print(f"Splits — train: {len(train_ids)}, val: {len(val_ids)}, "
          f"test: {len(test_ids)}")
    return train_ids, val_ids, test_ids, labels_df
