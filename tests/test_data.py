from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from liver_tumor.data import create_split_manifest, discover_cases, validate_case


def _write_nifti(path: Path, data: np.ndarray, affine: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def test_discovery_and_validation_accept_binary_pvp_space_masks(tmp_path: Path) -> None:
    case = tmp_path / "case001"
    affine = np.diag([0.8, 0.8, 3.0, 1.0])
    volume = np.zeros((16, 16, 8), dtype=np.int16)
    tumor = np.zeros_like(volume, dtype=np.uint8)
    liver = np.zeros_like(volume, dtype=np.uint8)
    liver[2:14, 2:14, 1:7] = 1
    tumor[6:10, 6:10, 3:5] = 1
    for phase in ["nc", "art", "pvp", "delay"]:
        _write_nifti(case / "NIFTI" / f"{phase}.nii.gz", volume, affine)
    _write_nifti(case / "mask_pvp.nii.gz", tumor, affine)
    _write_nifti(case / "liver_mask_pvp.nii.gz", liver, affine)
    (tmp_path / "meta_info_patient.tab").write_text(
        'ID\ttype\n"case001"\t"HCC"\n', encoding="utf-8"
    )

    records = discover_cases(tmp_path)
    result = validate_case(records[0], load_intensities=True)

    assert len(records) == 1
    assert records[0].tumor_type == "HCC"
    assert result["error_count"] == 0
    assert result["warning_count"] == 0
    assert result["masks"]["tumor"]["connected_components"] == 1
    assert result["masks"]["tumor"]["voxels_outside_liver"] == 0


def test_discovery_keeps_case_with_pvp_but_missing_tumor_mask(tmp_path: Path) -> None:
    case = tmp_path / "case_missing_mask"
    affine = np.eye(4)
    volume = np.zeros((8, 8, 4), dtype=np.int16)
    _write_nifti(case / "NIFTI" / "pvp.nii.gz", volume, affine)

    records = discover_cases(tmp_path)
    result = validate_case(records[0], load_intensities=False)

    assert len(records) == 1
    assert result["error_count"] > 0
    assert "missing_tumor_mask" in {issue["code"] for issue in result["issues"]}


def test_missing_liver_pseudolabel_is_warning_not_error(tmp_path: Path) -> None:
    case = tmp_path / "case001"
    affine = np.eye(4)
    volume = np.zeros((8, 8, 4), dtype=np.int16)
    tumor = np.zeros_like(volume, dtype=np.uint8)
    tumor[2:4, 2:4, 1:3] = 1
    for phase in ["nc", "art", "pvp", "delay"]:
        _write_nifti(case / "NIFTI" / f"{phase}.nii.gz", volume, affine)
    _write_nifti(case / "mask_pvp.nii.gz", tumor, affine)

    result = validate_case(discover_cases(tmp_path)[0], load_intensities=False)

    assert result["error_count"] == 0
    assert result["warning_count"] == 1
    assert result["issues"][0]["code"] == "missing_liver_mask"


def test_create_five_fold_split_has_no_patient_overlap(tmp_path: Path) -> None:
    rows = []
    tumor_types = ["HCC", "ICC", "CRLM", "BCLM", "HH"]
    for index in range(50):
        rows.append(
            {
                "case_id": f"case{index:03d}",
                "tumor_type": tumor_types[index % len(tumor_types)],
                "tumor_volume_mm3": 1000 + index * 100,
                "error_count": 0,
            }
        )
    cases_csv = tmp_path / "cases.csv"
    pd.DataFrame(rows).to_csv(cases_csv, index=False)

    manifest = create_split_manifest(
        cases_csv,
        tmp_path / "split.json",
        protocol="five_fold",
        seed=42,
        n_folds=5,
    )

    assert len(manifest["folds"]) == 5
    all_cases = {row["case_id"] for row in rows}
    for fold in manifest["folds"]:
        train = set(fold["train"])
        val = set(fold["val"])
        assert train.isdisjoint(val)
        assert train | val == all_cases
