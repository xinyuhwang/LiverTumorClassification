"""
splits.py — the single train/val/test split shared by every HERALD pipeline
(ds2net/, unet_hybrid/, nnunet/). Using one split keeps results comparable.
cv_folds() adds inner cross-validation folds over the training cases.
"""

from pathlib import Path
import pandas as pd

DEFAULT_SPLITS_CSV = str(Path(__file__).resolve().parent / "mct_ltdiag_split.csv")

# Cases excluded from all splits. 231109b01: its liver mask stays wrong even
# after resampling onto the PVP grid (18% of the tumor inside it; ≥10% of the
# mask on air/fat in every phase, PVP included).
# Re-included 2026-09-29 (E01 v1 data QA): 240504b27 and 240504e30, whose liver
# masks were on a different slice grid and are now resampled by
# data_prep/prepare_mct_ltdiag.py, and 240504e48, which shows no problem in QA.
# All four are training cases, so val/test are unchanged.
BAD_CASES = {'231109b01'}


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


def cv_folds(train_ids, labels_df, k=5, seed=0):
    """K inner cross-validation folds over the training cases only, stratified
    by tumor type (E06 v2). Returns [{"train": [...], "val": [...]}, ...]; the
    val lists partition train_ids. HERALD's val and test cases are never used,
    so ensembles of these folds can still be compared on val and reported on test."""
    from sklearn.model_selection import StratifiedKFold
    types = dict(zip(labels_df["case_id"], labels_df["type"].astype(str)))
    ids = sorted(train_ids)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return [{"train": [ids[i] for i in tr], "val": [ids[i] for i in va]}
            for tr, va in skf.split(ids, [types[c] for c in ids])]
