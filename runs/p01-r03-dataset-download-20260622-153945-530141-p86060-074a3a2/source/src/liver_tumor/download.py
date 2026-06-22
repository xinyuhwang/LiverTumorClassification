"""Resumable Harvard Dataverse download and safe case extraction."""

from __future__ import annotations

import json
import os
import re
import shutil
import tarfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import requests

from .experiment import atomic_json_dump


DATAVERSE_BASE_URL = "https://dataverse.harvard.edu"
PHASE_FILENAMES = ("nc.nii.gz", "art.nii.gz", "pvp.nii.gz", "delay.nii.gz")
TUMOR_MASK_FILENAME = "mask_pvp.nii.gz"
LIVER_MASK_FILENAME = "liver_mask_pvp.nii.gz"


@dataclass
class DatasetFile:
    file_id: int
    filename: str
    size_bytes: int
    content_type: str
    directory_label: str | None = None
    restricted: bool = False
    description: str | None = None


class DataverseDownloader:
    def __init__(
        self,
        persistent_id: str,
        destination: str | Path,
        token: str | None = None,
        base_url: str = DATAVERSE_BASE_URL,
        timeout_seconds: int = 120,
        chunk_size_mb: int = 8,
    ) -> None:
        self.persistent_id = persistent_id
        self.destination = Path(destination).expanduser().resolve()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.chunk_size = chunk_size_mb * 1024 * 1024
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "liver-tumor-segmentation/0.1"})
        if token:
            self.session.headers.update({"X-Dataverse-key": token})
        self.archives_dir = self.destination / "archives"
        self.inventory_path = self.destination / "download_inventory.json"

    def fetch_metadata(self) -> dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/api/datasets/:persistentId",
            params={"persistentId": self.persistent_id},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "OK":
            raise RuntimeError(f"Dataverse metadata request failed: {payload}")
        return payload["data"]

    @staticmethod
    def list_files(metadata: dict[str, Any]) -> list[DatasetFile]:
        files: list[DatasetFile] = []
        for entry in metadata.get("latestVersion", {}).get("files", []):
            data = entry.get("dataFile", {})
            if "id" not in data:
                continue
            files.append(
                DatasetFile(
                    file_id=int(data["id"]),
                    filename=str(data.get("filename", data["id"])),
                    size_bytes=int(data.get("filesize", 0)),
                    content_type=str(data.get("contentType", "application/octet-stream")),
                    directory_label=entry.get("directoryLabel"),
                    restricted=bool(entry.get("restricted", False)),
                    description=entry.get("description"),
                )
            )
        return files

    @staticmethod
    def select_files(
        files: Iterable[DatasetFile],
        include_regex: str | None = None,
        max_files: int | None = None,
    ) -> list[DatasetFile]:
        pattern = re.compile(include_regex) if include_regex else None
        selected = [item for item in files if pattern is None or pattern.search(item.filename)]
        selected.sort(key=lambda item: item.filename)
        return selected[:max_files] if max_files else selected

    def check_free_space(self, files: Iterable[DatasetFile], multiplier: float) -> dict[str, int]:
        self.destination.mkdir(parents=True, exist_ok=True)
        total = sum(item.size_bytes for item in files)
        required = int(total * multiplier)
        free = shutil.disk_usage(self.destination).free
        if free < required:
            raise OSError(
                f"Insufficient free space at {self.destination}: "
                f"need approximately {required / 2**30:.1f} GiB, "
                f"have {free / 2**30:.1f} GiB"
            )
        return {"selected_bytes": total, "required_bytes": required, "free_bytes": free}

    def _load_inventory(self) -> dict[str, Any]:
        if self.inventory_path.is_file():
            with self.inventory_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        return {
            "persistent_id": self.persistent_id,
            "files": {},
            "created_at": time.time(),
        }

    def _save_inventory(self, inventory: dict[str, Any]) -> None:
        inventory["updated_at"] = time.time()
        atomic_json_dump(inventory, self.inventory_path)

    def download_file(self, item: DatasetFile) -> tuple[Path, dict[str, Any]]:
        self.archives_dir.mkdir(parents=True, exist_ok=True)
        final_path = self.archives_dir / item.filename
        partial_path = final_path.with_suffix(final_path.suffix + ".part")
        record = asdict(item)

        if final_path.is_file() and final_path.stat().st_size == item.size_bytes:
            record.update({"download_status": "complete", "path": str(final_path)})
            return final_path, record
        if final_path.exists():
            final_path.rename(partial_path)

        existing = partial_path.stat().st_size if partial_path.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        response = self.session.get(
            f"{self.base_url}/api/access/datafile/{item.file_id}",
            headers=headers,
            stream=True,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        resumed = existing > 0 and response.status_code == 206
        if existing and not resumed:
            existing = 0
        mode = "ab" if resumed else "wb"
        with partial_path.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=self.chunk_size):
                if chunk:
                    handle.write(chunk)

        observed = partial_path.stat().st_size
        if item.size_bytes and observed != item.size_bytes:
            raise IOError(
                f"Size mismatch for {item.filename}: expected {item.size_bytes}, got {observed}"
            )
        partial_path.replace(final_path)
        record.update(
            {
                "download_status": "complete",
                "path": str(final_path),
                "resumed": resumed,
                "observed_bytes": observed,
            }
        )
        return final_path, record

    @staticmethod
    def _safe_extract(
        tar: tarfile.TarFile,
        destination: Path,
        include_dicom: bool = False,
    ) -> None:
        destination_resolved = destination.resolve()
        selected_members: list[tarfile.TarInfo] = []
        for member in tar.getmembers():
            parts = PurePosixPath(member.name).parts
            if not parts:
                continue
            top_level = parts[0]
            selected = (
                top_level == "NIFTI"
                or (len(parts) == 1 and top_level in {TUMOR_MASK_FILENAME, LIVER_MASK_FILENAME})
                or (include_dicom and top_level == "DICOM")
            )
            if not selected:
                continue
            target = (destination / member.name).resolve()
            if destination_resolved != target and destination_resolved not in target.parents:
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Links are not allowed in dataset archive: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise RuntimeError(f"Unsupported archive member: {member.name}")
            selected_members.append(member)
        tar.extractall(destination, members=selected_members)

    def extract_case_archive(
        self,
        archive: Path,
        include_dicom: bool = False,
    ) -> tuple[Path, str]:
        if not tarfile.is_tarfile(archive):
            return archive, "not_tar"
        case_id = archive.name
        for suffix in [".tar.gz", ".tgz", ".tar"]:
            if case_id.endswith(suffix):
                case_id = case_id[: -len(suffix)]
                break
        destination = self.destination / case_id
        expected_files = [destination / "NIFTI" / name for name in PHASE_FILENAMES]
        expected_files.extend(
            [destination / TUMOR_MASK_FILENAME, destination / LIVER_MASK_FILENAME]
        )
        if all(path.is_file() for path in expected_files):
            return destination, "already_complete"

        temporary = self.destination / f".{case_id}.extracting"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        try:
            with tarfile.open(archive, "r:*") as handle:
                self._safe_extract(handle, temporary, include_dicom=include_dicom)
            missing = [
                str(path.relative_to(destination))
                for path in expected_files
                if not (temporary / path.relative_to(destination)).is_file()
            ]
            if missing:
                raise RuntimeError(
                    f"Archive is missing expected training files {missing}: {archive}"
                )
            if destination.exists():
                shutil.rmtree(destination)
            temporary.replace(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return destination, "extracted"

    def run(
        self,
        include_regex: str | None = None,
        max_files: int | None = None,
        extract: bool = True,
        extract_dicom: bool = False,
        delete_archives: bool = False,
        free_space_multiplier: float = 2.2,
        metadata_only: bool = False,
    ) -> dict[str, Any]:
        metadata = self.fetch_metadata()
        all_files = self.list_files(metadata)
        selected = self.select_files(all_files, include_regex, max_files)
        inventory = self._load_inventory()
        selected_bytes = sum(item.size_bytes for item in selected)
        inventory["dataset"] = {
            "persistent_id": self.persistent_id,
            "version": metadata.get("latestVersion", {}).get("versionNumber"),
            "version_minor": metadata.get("latestVersion", {}).get("versionMinorNumber"),
            "all_file_count": len(all_files),
            "selected_file_count": len(selected),
            "selected_bytes": selected_bytes,
            "free_space_multiplier": free_space_multiplier,
            "estimated_required_bytes": int(selected_bytes * free_space_multiplier),
        }
        inventory["selected_files"] = [asdict(item) for item in selected]
        self.destination.mkdir(parents=True, exist_ok=True)

        if metadata_only:
            self._save_inventory(inventory)
            return inventory

        inventory["disk"] = self.check_free_space(selected, free_space_multiplier)
        self._save_inventory(inventory)

        for index, item in enumerate(selected, start=1):
            print(f"[{index}/{len(selected)}] {item.filename} ({item.size_bytes / 2**20:.1f} MiB)")
            try:
                archive, record = self.download_file(item)
                if extract and tarfile.is_tarfile(archive):
                    destination, status = self.extract_case_archive(
                        archive,
                        include_dicom=extract_dicom,
                    )
                    record.update({"extract_status": status, "extract_path": str(destination)})
                    if delete_archives and status in {"extracted", "already_complete"}:
                        archive.unlink(missing_ok=True)
                        record["archive_deleted"] = True
                inventory["files"][str(item.file_id)] = record
            except Exception as exc:
                inventory["files"][str(item.file_id)] = {
                    **asdict(item),
                    "download_status": "failed",
                    "error": str(exc),
                }
                self._save_inventory(inventory)
                raise
            self._save_inventory(inventory)
        return inventory


def token_from_environment(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else None
