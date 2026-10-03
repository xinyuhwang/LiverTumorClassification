"""Tests for common/splits.py. Run from the repository root: pytest tests/"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.splits import load_splits, cv_folds


def test_cv_folds_partition_train_only():
    train, val, test, labels = load_splits()
    folds = cv_folds(train, labels, k=5)
    assert len(folds) == 5
    vals = [c for f in folds for c in f["val"]]
    assert sorted(vals) == sorted(train)                     # val lists partition train
    for f in folds:
        assert not set(f["train"]) & set(f["val"])
        assert sorted(f["train"] + f["val"]) == sorted(train)
        assert not (set(f["train"]) | set(f["val"])) & (set(val) | set(test))


def test_cv_folds_stratified_and_deterministic():
    train, _, _, labels = load_splits()
    types = dict(zip(labels["case_id"], labels["type"].astype(str)))
    folds = cv_folds(train, labels, k=5)
    assert folds == cv_folds(list(reversed(train)), labels, k=5)   # order-independent
    total = Counter(types[c] for c in train)
    for f in folds:
        per = Counter(types[c] for c in f["val"])
        for t, n in total.items():
            assert abs(per[t] - n / 5) <= 1
