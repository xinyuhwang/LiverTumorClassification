"""MCT-LTDiag discovery, validation, exploration, and patient-level splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import generate_binary_structure, label as connected_components
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm

from .experiment import atomic_json_dump


PHASE_FILES = {
    "nc": "nc.nii.gz",
    "art": "art.nii.gz",
    "pvp": "pvp.nii.gz",
    "delay": "delay.nii.gz",
}
TUMOR_MASK_FILENAME = "mask_pvp.nii.gz"
LIVER_MASK_FILENAME = "liver_mask_pvp.nii.gz"


@dataclass(frozen=True)
class CaseRecord:
    case_id: str
    case_dir: Path
    phase_paths: dict[str, Path]
    tumor_mask_path: Path
    liver_mask_path: Path
    tumor_type: str | None = None


def _metadata_file(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.is_file():
        return direct
    candidates = sorted(root.glob(f"**/{name}"))
    return candidates[0] if candidates else None


def read_metadata(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root_path = Path(root).expanduser().resolve()
    patient_path = _metadata_file(root_path, "meta_info_patient.tab")
    tumor_path = _metadata_file(root_path, "meta_info_tumor.tab")
    patients = pd.read_csv(patient_path, sep="\t", dtype=str) if patient_path else pd.DataFrame()
    tumors = pd.read_csv(tumor_path, sep="\t") if tumor_path else pd.DataFrame()
    for frame in [patients, tumors]:
        if "ID" in frame.columns:
            frame["ID"] = frame["ID"].astype(str).str.strip().str.strip('"')
    return patients, tumors


def discover_cases(root: str | Path) -> list[CaseRecord]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root_path}")
    patients, _ = read_metadata(root_path)
    type_map = (
        dict(zip(patients["ID"], patients["type"].astype(str).str.strip().str.upper()))
        if {"ID", "type"}.issubset(patients.columns)
        else {}
    )
    case_dirs = {
        path.parent.parent for path in root_path.glob(f"**/NIFTI/{PHASE_FILES['pvp']}")
    }
    case_dirs.update(path.parent for path in root_path.glob(f"**/{TUMOR_MASK_FILENAME}"))

    records: list[CaseRecord] = []
    for case_dir in sorted(case_dirs, key=str):
        case_id = case_dir.name
        records.append(
            CaseRecord(
                case_id=case_id,
                case_dir=case_dir,
                phase_paths={
                    phase: case_dir / "NIFTI" / filename
                    for phase, filename in PHASE_FILES.items()
                },
                tumor_mask_path=case_dir / TUMOR_MASK_FILENAME,
                liver_mask_path=case_dir / LIVER_MASK_FILENAME,
                tumor_type=type_map.get(case_id),
            )
        )
    return records


def _update_issue_counts(result: dict[str, Any]) -> None:
    issues = result.get("issues", [])
    result["issue_count"] = len(issues)
    result["error_count"] = sum(issue["severity"] == "error" for issue in issues)
    result["warning_count"] = sum(issue["severity"] == "warning" for issue in issues)


def _geometry(image: nib.spatialimages.SpatialImage) -> dict[str, Any]:
    return {
        "shape": [int(value) for value in image.shape],
        "spacing": [float(value) for value in image.header.get_zooms()[:3]],
        "orientation": list(nib.aff2axcodes(image.affine)),
        "affine": np.asarray(image.affine).round(6).tolist(),
    }


def _issue(severity: str, code: str, message: str) -> dict[str, str]:
    return {"severity": severity, "code": code, "message": message}


def _intensity_summary(volume: np.ndarray) -> dict[str, float | int]:
    finite = np.isfinite(volume)
    if not finite.any():
        return {"finite_voxels": 0, "nonfinite_voxels": int(volume.size)}
    values = volume[finite]
    percentiles = np.percentile(values, [0, 1, 50, 99, 100])
    return {
        "finite_voxels": int(finite.sum()),
        "nonfinite_voxels": int((~finite).sum()),
        "min": float(percentiles[0]),
        "p01": float(percentiles[1]),
        "median": float(percentiles[2]),
        "p99": float(percentiles[3]),
        "max": float(percentiles[4]),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def validate_case(record: CaseRecord, load_intensities: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "case_id": record.case_id,
        "case_dir": str(record.case_dir),
        "tumor_type": record.tumor_type,
        "issues": [],
        "phases": {},
    }
    issues: list[dict[str, str]] = result["issues"]

    pvp_path = record.phase_paths["pvp"]
    if not pvp_path.is_file():
        issues.append(_issue("error", "missing_pvp", str(pvp_path)))
        _update_issue_counts(result)
        return result

    try:
        reference = nib.load(str(pvp_path))
        reference_geometry = _geometry(reference)
        result["reference_geometry"] = reference_geometry
    except Exception as exc:
        issues.append(_issue("error", "unreadable_pvp", str(exc)))
        _update_issue_counts(result)
        return result

    for phase, path in record.phase_paths.items():
        phase_result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        result["phases"][phase] = phase_result
        if not path.is_file():
            issues.append(_issue("error", f"missing_phase_{phase}", str(path)))
            continue
        try:
            image = nib.load(str(path))
            geometry = _geometry(image)
            phase_result["geometry"] = geometry
            if tuple(image.shape) != tuple(reference.shape):
                issues.append(
                    _issue(
                        "error",
                        f"shape_mismatch_{phase}",
                        f"{image.shape} != {reference.shape}",
                    )
                )
            if not np.allclose(image.affine, reference.affine, atol=1e-3, rtol=1e-5):
                linear_difference = float(
                    np.max(np.abs(image.affine[:3, :3] - reference.affine[:3, :3]))
                )
                translation_difference = image.affine[:3, 3] - reference.affine[:3, 3]
                try:
                    voxel_shift = np.linalg.solve(
                        reference.affine[:3, :3], translation_difference
                    )
                except np.linalg.LinAlgError:
                    voxel_shift = np.full(3, np.inf)
                phase_result["affine_difference"] = {
                    "max_linear_abs": linear_difference,
                    "translation_mm": translation_difference.round(6).tolist(),
                    "translation_voxels": voxel_shift.round(6).tolist(),
                }
                same_orientation = tuple(nib.aff2axcodes(image.affine)) == tuple(
                    nib.aff2axcodes(reference.affine)
                )
                small_origin_shift = linear_difference < 1e-3 and np.max(np.abs(voxel_shift)) <= 1
                severity = "warning" if same_orientation and small_origin_shift else "error"
                issues.append(
                    _issue(
                        severity,
                        f"affine_mismatch_{phase}",
                        "Affine differs from PVP; "
                        f"translation={translation_difference.round(3).tolist()} mm, "
                        f"voxel_shift={voxel_shift.round(3).tolist()}",
                    )
                )
            if load_intensities:
                summary = _intensity_summary(image.get_fdata(dtype=np.float32))
                phase_result["intensity"] = summary
                if summary.get("nonfinite_voxels", 0):
                    issues.append(
                        _issue(
                            "error",
                            f"nonfinite_{phase}",
                            f"{summary['nonfinite_voxels']} non-finite voxels",
                        )
                    )
        except Exception as exc:
            issues.append(_issue("error", f"unreadable_phase_{phase}", str(exc)))

    masks: dict[str, Any] = {}
    result["masks"] = masks
    loaded_masks: dict[str, np.ndarray] = {}
    for name, path in [
        ("tumor", record.tumor_mask_path),
        ("liver", record.liver_mask_path),
    ]:
        mask_result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
        masks[name] = mask_result
        if not path.is_file():
            issues.append(_issue("error", f"missing_{name}_mask", str(path)))
            continue
        try:
            image = nib.load(str(path))
            mask_result["geometry"] = _geometry(image)
            if tuple(image.shape) != tuple(reference.shape):
                issues.append(
                    _issue(
                        "error",
                        f"shape_mismatch_{name}_mask",
                        f"{image.shape} != {reference.shape}",
                    )
                )
            if not np.allclose(image.affine, reference.affine, atol=1e-3, rtol=1e-5):
                issues.append(
                    _issue("error", f"affine_mismatch_{name}_mask", "Affine differs from PVP")
                )
            values = image.get_fdata(dtype=np.float32)
            if not np.isfinite(values).all():
                issues.append(_issue("error", f"nonfinite_{name}_mask", "Non-finite values"))
            unique = np.unique(values)
            mask_result["unique_values"] = [float(value) for value in unique[:32]]
            if len(unique) > 32:
                mask_result["unique_values_truncated"] = True
            if not np.isin(unique, [0, 1]).all():
                issues.append(
                    _issue(
                        "error",
                        f"unexpected_{name}_labels",
                        f"Expected binary mask, observed {unique[:16].tolist()}",
                    )
                )
            binary = values > 0
            loaded_masks[name] = binary
            mask_result["foreground_voxels"] = int(binary.sum())
            voxel_volume = float(np.prod(reference.header.get_zooms()[:3]))
            mask_result["volume_mm3"] = float(binary.sum() * voxel_volume)
            if not binary.any():
                issues.append(_issue("error", f"empty_{name}_mask", "No foreground voxels"))
        except Exception as exc:
            issues.append(_issue("error", f"unreadable_{name}_mask", str(exc)))

    tumor = loaded_masks.get("tumor")
    liver = loaded_masks.get("liver")
    if tumor is not None:
        _, component_count = connected_components(
            tumor, structure=generate_binary_structure(3, 2)
        )
        masks["tumor"]["connected_components"] = int(component_count)
    if tumor is not None and liver is not None and tumor.shape == liver.shape:
        outside = tumor & ~liver
        count = int(outside.sum())
        masks["tumor"]["voxels_outside_liver"] = count
        masks["tumor"]["outside_liver_fraction"] = float(count / max(int(tumor.sum()), 1))
        if count:
            severity = "warning" if count / max(int(tumor.sum()), 1) < 0.02 else "error"
            issues.append(
                _issue(
                    severity,
                    "tumor_outside_liver",
                    f"{count} tumor voxels are outside the liver mask",
                )
            )

    _update_issue_counts(result)
    return result


def _flatten_case(result: dict[str, Any]) -> dict[str, Any]:
    reference = result.get("reference_geometry", {})
    tumor = result.get("masks", {}).get("tumor", {})
    liver = result.get("masks", {}).get("liver", {})
    row: dict[str, Any] = {
        "case_id": result["case_id"],
        "case_dir": result["case_dir"],
        "tumor_type": result.get("tumor_type"),
        "shape": "x".join(str(value) for value in reference.get("shape", [])),
        "spacing_x": (reference.get("spacing") or [None, None, None])[0],
        "spacing_y": (reference.get("spacing") or [None, None, None])[1],
        "spacing_z": (reference.get("spacing") or [None, None, None])[2],
        "orientation": "".join(reference.get("orientation", [])),
        "tumor_voxels": tumor.get("foreground_voxels"),
        "tumor_volume_mm3": tumor.get("volume_mm3"),
        "liver_voxels": liver.get("foreground_voxels"),
        "liver_volume_mm3": liver.get("volume_mm3"),
        "connected_components": tumor.get("connected_components"),
        "tumor_outside_liver_fraction": tumor.get("outside_liver_fraction"),
        "error_count": result.get("error_count", 0),
        "warning_count": result.get("warning_count", 0),
    }
    for phase in PHASE_FILES:
        intensity = result.get("phases", {}).get(phase, {}).get("intensity", {})
        row[f"{phase}_p01"] = intensity.get("p01")
        row[f"{phase}_median"] = intensity.get("median")
        row[f"{phase}_p99"] = intensity.get("p99")
    return row


def _write_qc_figure(record: CaseRecord, destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tumor = nib.load(str(record.tumor_mask_path)).get_fdata() > 0
    liver = (
        nib.load(str(record.liver_mask_path)).get_fdata() > 0
        if record.liver_mask_path.is_file()
        else None
    )
    z_index = int(np.argmax(tumor.sum(axis=(0, 1))))
    windows = {
        "nc": (-100, 200),
        "art": (-100, 300),
        "pvp": (-100, 250),
        "delay": (-100, 200),
    }
    figure, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for axis, (phase, path) in zip(axes, record.phase_paths.items()):
        if not path.is_file():
            axis.set_title(f"{phase.upper()} missing")
            axis.axis("off")
            continue
        volume = nib.load(str(path)).get_fdata(dtype=np.float32)
        low, high = windows[phase]
        axis.imshow(volume[:, :, z_index].T, cmap="gray", vmin=low, vmax=high, origin="lower")
        axis.contour(tumor[:, :, z_index].T, levels=[0.5], colors=["red"], linewidths=1.0)
        if liver is not None:
            axis.contour(liver[:, :, z_index].T, levels=[0.5], colors=["lime"], linewidths=0.6)
        axis.set_title(phase.upper())
        axis.axis("off")
    figure.suptitle(f"{record.case_id} | {record.tumor_type or 'unknown'} | z={z_index}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=140)
    plt.close(figure)


def audit_dataset(
    root: str | Path,
    output_dir: str | Path,
    max_cases: int | None = None,
    load_intensities: bool = True,
    qc_cases_per_type: int = 1,
) -> dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_records = discover_cases(root_path)
    patients, _ = read_metadata(root_path)
    metadata_ids = (
        set(patients["ID"].dropna().astype(str)) if "ID" in patients.columns else set()
    )
    records = all_records
    if max_cases:
        records = records[:max_cases]
    if not records:
        raise RuntimeError(
            f"No cases containing PVP images or {TUMOR_MASK_FILENAME} found under {root_path}"
        )

    results = [
        validate_case(record, load_intensities=load_intensities)
        for record in tqdm(records, desc="Validating cases")
    ]

    audited_id_counts = pd.Series([record.case_id for record in records]).value_counts()
    duplicate_ids = set(audited_id_counts[audited_id_counts > 1].index)
    for result in results:
        if result["case_id"] in duplicate_ids:
            result["issues"].append(
                _issue(
                    "error",
                    "duplicate_case_id",
                    f"Case ID appears {int(audited_id_counts[result['case_id']])} times",
                )
            )
            _update_issue_counts(result)

    completeness_checked = max_cases is None
    if completeness_checked:
        discovered_ids = {record.case_id for record in all_records}
        if not metadata_ids:
            for result in results:
                result["issues"].append(
                    _issue(
                        "error",
                        "missing_patient_metadata",
                        "meta_info_patient.tab with an ID column is required",
                    )
                )
                _update_issue_counts(result)
        else:
            for result in results:
                if result["case_id"] not in metadata_ids:
                    result["issues"].append(
                        _issue(
                            "error",
                            "case_not_in_metadata",
                            "Discovered case is absent from meta_info_patient.tab",
                        )
                    )
                if not result.get("tumor_type"):
                    result["issues"].append(
                        _issue(
                            "error",
                            "missing_tumor_type_metadata",
                            "No tumor type was found for this case",
                        )
                    )
                _update_issue_counts(result)

            type_map = dict(
                zip(
                    patients["ID"],
                    patients["type"].astype(str).str.strip().str.upper(),
                )
            ) if "type" in patients.columns else {}
            for case_id in sorted(metadata_ids - discovered_ids):
                missing_result: dict[str, Any] = {
                    "case_id": case_id,
                    "case_dir": "",
                    "tumor_type": type_map.get(case_id),
                    "issues": [
                        _issue(
                            "error",
                            "metadata_case_not_discovered",
                            "Metadata case has no PVP image or tumor mask on disk",
                        )
                    ],
                    "phases": {},
                    "masks": {},
                }
                _update_issue_counts(missing_result)
                results.append(missing_result)

    selected: list[tuple[CaseRecord, dict[str, Any]]] = []
    by_type: dict[str, int] = {}
    for record, result in zip(records, results):
        key = record.tumor_type or "UNKNOWN"
        if by_type.get(key, 0) < qc_cases_per_type:
            selected.append((record, result))
            by_type[key] = by_type.get(key, 0) + 1
    for record, result in selected:
        try:
            _write_qc_figure(record, output / "qc" / f"{record.case_id}.png")
        except Exception as exc:
            result["issues"].append(
                _issue("warning", "qc_figure_failed", str(exc))
            )
            _update_issue_counts(result)

    rows = [_flatten_case(result) for result in results]
    cases_frame = pd.DataFrame(rows).sort_values("case_id")
    cases_frame.to_csv(output / "cases.csv", index=False)

    issue_rows: list[dict[str, Any]] = []
    for result in results:
        for issue in result.get("issues", []):
            issue_rows.append({"case_id": result["case_id"], **issue})
    issue_frame = pd.DataFrame(issue_rows, columns=["case_id", "severity", "code", "message"])
    issue_frame.to_csv(output / "issues.csv", index=False)
    atomic_json_dump(results, output / "case_details.json")

    type_counts = {
        str(key): int(value)
        for key, value in cases_frame["tumor_type"].fillna("UNKNOWN").value_counts().items()
    }
    def numeric_stats(column: str) -> dict[str, float | None]:
        values = pd.to_numeric(cases_frame[column], errors="coerce").dropna()
        if values.empty:
            return {"min": None, "median": None, "max": None}
        return {
            "min": float(values.min()),
            "median": float(values.median()),
            "max": float(values.max()),
        }

    summary = {
        "dataset_root": str(root_path),
        "case_count": len(results),
        "audited_discovered_case_count": len(records),
        "total_discovered_case_count": len(all_records),
        "metadata_case_count": len(metadata_ids),
        "metadata_completeness_checked": completeness_checked,
        "tumor_type_counts": type_counts,
        "cases_with_errors": int((cases_frame["error_count"] > 0).sum()),
        "cases_with_warnings": int((cases_frame["warning_count"] > 0).sum()),
        "total_errors": int((issue_frame["severity"] == "error").sum())
        if not issue_frame.empty
        else 0,
        "total_warnings": int((issue_frame["severity"] == "warning").sum())
        if not issue_frame.empty
        else 0,
        "spacing": {axis: numeric_stats(f"spacing_{axis}") for axis in ["x", "y", "z"]},
        "tumor_volume_mm3": numeric_stats("tumor_volume_mm3"),
    }
    atomic_json_dump(summary, output / "summary.json")

    report_lines = [
        "# MCT-LTDiag Dataset Audit",
        "",
        f"- Dataset root: `{root_path}`",
        f"- Cases audited: {summary['case_count']}",
        f"- Cases discovered on disk: {summary['total_discovered_case_count']}",
        f"- Cases listed in metadata: {summary['metadata_case_count']}",
        f"- Full completeness check: {summary['metadata_completeness_checked']}",
        f"- Cases with errors: {summary['cases_with_errors']}",
        f"- Cases with warnings: {summary['cases_with_warnings']}",
        "",
        "## Tumor Types",
        "",
        *[f"- {name}: {count}" for name, count in sorted(type_counts.items())],
        "",
        "## Outputs",
        "",
        "- `cases.csv`: one row per case",
        "- `issues.csv`: machine-readable validation failures",
        "- `case_details.json`: complete geometry and intensity details",
        "- `qc/`: registered four-phase overlays",
        "",
    ]
    (output / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return summary


def _stratification_labels(frame: pd.DataFrame) -> pd.Series:
    tumor_type = frame["tumor_type"].fillna("UNKNOWN").astype(str)
    if "tumor_volume_mm3" not in frame or frame["tumor_volume_mm3"].isna().all():
        return tumor_type
    try:
        size_bin = pd.qcut(
            frame["tumor_volume_mm3"].rank(method="first"),
            q=3,
            labels=["small", "medium", "large"],
        ).astype(str)
        combined = tumor_type + "__" + size_bin
        if combined.value_counts().min() >= 2:
            return combined
    except ValueError:
        pass
    return tumor_type


def create_split_manifest(
    cases_csv: str | Path,
    output_path: str | Path,
    protocol: str,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    n_folds: int = 5,
) -> dict[str, Any]:
    frame = pd.read_csv(cases_csv)
    if frame["case_id"].duplicated().any():
        raise ValueError("Duplicate case IDs in cases.csv")
    if "error_count" not in frame.columns:
        raise ValueError("cases.csv is missing the required error_count column")
    if (frame["error_count"] > 0).any():
        raise ValueError("Cannot create final split while dataset validation errors remain")
    labels = _stratification_labels(frame)
    manifest: dict[str, Any] = {"protocol": protocol, "seed": seed}

    if protocol == "five_fold":
        if labels.value_counts().min() < n_folds:
            labels = frame["tumor_type"].fillna("UNKNOWN").astype(str)
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = []
        for fold, (train_indices, val_indices) in enumerate(splitter.split(frame, labels)):
            folds.append(
                {
                    "fold": fold,
                    "train": sorted(frame.iloc[train_indices]["case_id"].astype(str).tolist()),
                    "val": sorted(frame.iloc[val_indices]["case_id"].astype(str).tolist()),
                }
            )
        manifest["folds"] = folds
    elif protocol == "holdout":
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")
        train, remainder, train_labels, remainder_labels = train_test_split(
            frame,
            labels,
            test_size=val_ratio + test_ratio,
            stratify=labels,
            random_state=seed,
        )
        relative_test = test_ratio / (val_ratio + test_ratio)
        val, test = train_test_split(
            remainder,
            test_size=relative_test,
            stratify=remainder_labels,
            random_state=seed,
        )
        manifest["splits"] = {
            "train": sorted(train["case_id"].astype(str).tolist()),
            "val": sorted(val["case_id"].astype(str).tolist()),
            "test": sorted(test["case_id"].astype(str).tolist()),
        }
    else:
        raise ValueError(f"Unsupported split protocol: {protocol}")

    atomic_json_dump(manifest, Path(output_path))
    return manifest
