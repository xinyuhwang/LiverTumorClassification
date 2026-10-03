"""Tests for common/metrics.py. Run from the repository root: pytest tests/"""

import sys
from pathlib import Path

import numpy as np
import pytest

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


def test_lesion_metrics_detection_and_fp():
    from common.metrics import lesion_metrics
    gt = np.zeros((20, 20, 20), bool); pred = np.zeros_like(gt)
    gt[2:6, 2:6, 2:6] = True            # lesion 1: 64 voxels, half predicted
    gt[12:14, 12:14, 12:14] = True      # lesion 2: 8 voxels, missed
    pred[2:6, 2:6, 2:4] = True          # overlaps lesion 1
    pred[16:18, 2:4, 2:4] = True        # false positive
    s, les = lesion_metrics(pred, gt, voxel_ml=0.001)
    assert s == {"n_gt_lesions": 2, "n_detected": 1, "n_pred_lesions": 2,
                 "n_fp_lesions": 1, "lesion_recall": 0.5}
    assert les[0]["detected"] and les[0]["overlap"] == pytest.approx(0.5)
    assert les[0]["lesion_Dice"] == pytest.approx(2 * 32 / (64 + 32))
    assert not les[1]["detected"] and les[1]["gt_ml"] == pytest.approx(0.008)


def test_lesion_metrics_empty():
    from common.metrics import lesion_metrics
    z = np.zeros((5, 5, 5), bool)
    s, les = lesion_metrics(z, z, 1.0)
    assert s["n_gt_lesions"] == 0 and s["n_fp_lesions"] == 0 and les == []
