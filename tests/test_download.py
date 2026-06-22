from __future__ import annotations

import tarfile
from pathlib import Path

from liver_tumor.download import DataverseDownloader


def test_case_extraction_skips_dicom_by_default(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "NIFTI").mkdir(parents=True)
    (source / "DICOM").mkdir()
    for phase in ["nc", "art", "pvp", "delay"]:
        (source / "NIFTI" / f"{phase}.nii.gz").write_bytes(phase.encode())
    (source / "mask_pvp.nii.gz").write_bytes(b"tumor")
    (source / "liver_mask_pvp.nii.gz").write_bytes(b"liver")
    (source / "DICOM" / "slice.dcm").write_bytes(b"dicom")

    archive = tmp_path / "case001.tar"
    with tarfile.open(archive, "w") as handle:
        for path in source.rglob("*"):
            handle.add(path, arcname=path.relative_to(source), recursive=False)

    downloader = DataverseDownloader(
        persistent_id="doi:test",
        destination=tmp_path / "dataset",
    )
    case_dir, status = downloader.extract_case_archive(archive)

    assert status == "extracted"
    assert (case_dir / "NIFTI" / "pvp.nii.gz").read_bytes() == b"pvp"
    assert (case_dir / "mask_pvp.nii.gz").read_bytes() == b"tumor"
    assert (case_dir / "liver_mask_pvp.nii.gz").read_bytes() == b"liver"
    assert not (case_dir / "DICOM").exists()
