"""Tests for common/metrics.py. Run from the repository root: pytest tests/"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.metrics import volume_metrics, liver_extras, largest_component


def test_largest_component_drops_small_blobs():
    m = np.zeros((20, 20, 10), bool)
    m[2:12, 2:12, 2:8] = True          # "liver": 600 voxels
    m[16:19, 16:19, 1:3] = True        # far blob: 18 voxels
    out = largest_component(m)
    assert out.sum() == 600 and not out[17, 17, 1]


def test_largest_component_keeps_single_and_empty():
    empty = np.zeros((5, 5, 5), bool)
    assert not largest_component(empty).any()
    one = empty.copy(); one[1:3, 1:3, 1:3] = True
    assert (largest_component(one) == one).all()


def test_volume_metrics_and_liver_extras():
    pred = np.zeros((4, 4, 2), bool); pred[:2] = True
    liver = pred.copy()
    tumor = np.zeros_like(pred); tumor[1:3, :, 0] = True     # half inside the liver
    assert volume_metrics(pred, liver)["Dice"] == 1.0
    ex = liver_extras(pred, liver, tumor)
    assert round(ex["Dice_vs_union"], 3) == 0.889 and ex["tumor_covered"] == 0.5
    assert volume_metrics(np.zeros_like(pred), np.zeros_like(pred))["Dice"] == 1.0
