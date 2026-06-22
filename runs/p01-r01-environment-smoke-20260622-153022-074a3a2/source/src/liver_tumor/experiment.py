"""Reproducible run directories, logging, snapshots, and artifact manifests."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import time
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from .config import dump_config


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_command(command: list[str], cwd: Path, timeout: int = 30) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}


def _git_info(root: Path) -> dict[str, Any]:
    def value(args: list[str]) -> str | None:
        result = _run_command(["git", *args], root)
        if result.get("returncode") == 0:
            return result.get("stdout") or None
        return None

    status = value(["status", "--short"])
    return {
        "root": str(root),
        "commit": value(["rev-parse", "HEAD"]),
        "short_commit": value(["rev-parse", "--short", "HEAD"]),
        "branch": value(["branch", "--show-current"]),
        "status": status or "",
        "dirty": bool(status),
        "remote": value(["remote", "get-url", "origin"]),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_token(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in value)
    return "-".join(part for part in cleaned.split("-") if part) or "run"


def make_run_id(phase: str, round_name: str, task: str, git_sha: str | None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sha = (git_sha or "nogit")[:8]
    return "-".join(
        [_safe_token(phase), _safe_token(round_name), _safe_token(task), timestamp, sha]
    )


def atomic_json_dump(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    temporary.replace(path)


def seed_everything(seed: int) -> dict[str, Any]:
    random.seed(seed)
    state: dict[str, Any] = {"python": seed}
    try:
        import numpy as np

        np.random.seed(seed)
        state["numpy"] = seed
    except ImportError:
        state["numpy"] = None
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        state["torch"] = seed
    except ImportError:
        state["torch"] = None
    return state


def _environment_info(root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "captured_at": _utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        info["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        info["torch"] = {"installed": False}

    info["nvidia_smi"] = _run_command(["nvidia-smi"], root, timeout=20)
    info["conda_list"] = _run_command(["conda", "list", "--json"], root, timeout=60)
    info["pip_freeze"] = _run_command(
        [sys.executable, "-m", "pip", "freeze"], root, timeout=60
    )
    return info


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _Tee:
    def __init__(self, primary: TextIO, log_file: TextIO):
        self.primary = primary
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.primary.write(text)
        self.log_file.write(text)
        self.log_file.flush()
        return len(text)

    def flush(self) -> None:
        self.primary.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.primary.isatty()

    @property
    def encoding(self) -> str:
        return self.primary.encoding


class RunContext(AbstractContextManager["RunContext"]):
    """Context manager that makes every command self-documenting."""

    def __init__(
        self,
        config: dict[str, Any],
        task: str,
        runs_root: str | Path | None = None,
        phase: str | None = None,
        round_name: str | None = None,
    ) -> None:
        self.config = config
        experiment = config.get("experiment", {})
        self.root = repository_root()
        self.git = _git_info(self.root)
        self.phase = phase or str(experiment.get("phase", "phase"))
        self.round_name = round_name or str(experiment.get("round", "round"))
        self.task = task
        configured_root = runs_root or experiment.get("runs_root", self.root / "runs")
        configured_root = Path(configured_root)
        if not configured_root.is_absolute():
            configured_root = self.root / configured_root
        self.runs_root = configured_root.resolve()
        requested_id = experiment.get("run_id")
        self.run_id = str(requested_id) if requested_id else make_run_id(
            self.phase, self.round_name, task, self.git.get("short_commit")
        )
        self.run_dir = self.runs_root / self.run_id
        self.log_dir = self.run_dir / "logs"
        self.report_dir = self.run_dir / "reports"
        self.plot_dir = self.run_dir / "plots"
        self.source_dir = self.run_dir / "source"
        self._start_time = 0.0
        self._stdout: TextIO | None = None
        self._stderr: TextIO | None = None
        self._log_handle: TextIO | None = None
        self._artifacts: list[dict[str, Any]] = []

    def __enter__(self) -> "RunContext":
        if self.run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {self.run_dir}")
        for directory in [
            self.log_dir,
            self.report_dir,
            self.plot_dir,
            self.source_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        matplotlib_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "liver-tumor-matplotlib"
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

        self._start_time = time.monotonic()
        dump_config(self.config, self.run_dir / "config.resolved.yaml")
        atomic_json_dump(self.git, self.run_dir / "git_info.json")
        atomic_json_dump(_environment_info(self.root), self.run_dir / "environment.json")
        (self.run_dir / "command.txt").write_text(
            " ".join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
        )
        (self.run_dir / "git_diff.patch").write_text(
            _run_command(["git", "diff", "--no-ext-diff"], self.root, timeout=60).get(
                "stdout", ""
            ),
            encoding="utf-8",
        )
        self._snapshot_source()
        seed = int(self.config.get("experiment", {}).get("seed", 42))
        atomic_json_dump(seed_everything(seed), self.run_dir / "seeds.json")
        self._write_status("RUNNING")

        self._stdout, self._stderr = sys.stdout, sys.stderr
        self._log_handle = (self.log_dir / "run.log").open("a", encoding="utf-8")
        sys.stdout = _Tee(self._stdout, self._log_handle)
        sys.stderr = _Tee(self._stderr, self._log_handle)
        print(f"RUN_ID={self.run_id}")
        print(f"RUN_DIR={self.run_dir}")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        elapsed = time.monotonic() - self._start_time
        if exc is None:
            print(f"Run completed successfully in {elapsed:.1f}s")
            self._write_status("SUCCESS", elapsed_seconds=elapsed)
        else:
            print(f"Run failed after {elapsed:.1f}s: {exc}", file=sys.stderr)
            self._write_status(
                "FAILED",
                elapsed_seconds=elapsed,
                error_type=getattr(exc_type, "__name__", str(exc_type)),
                error=str(exc),
            )
        self._write_artifact_manifest()
        if self._stdout is not None:
            sys.stdout = self._stdout
        if self._stderr is not None:
            sys.stderr = self._stderr
        if self._log_handle is not None:
            self._log_handle.close()
        return False

    def _write_status(self, status: str, **extra: Any) -> None:
        payload = {
            "run_id": self.run_id,
            "status": status,
            "phase": self.phase,
            "round": self.round_name,
            "task": self.task,
            "updated_at": _utc_now(),
            **extra,
        }
        atomic_json_dump(payload, self.run_dir / "status.json")

    def _snapshot_source(self) -> None:
        patterns = [
            "*.py",
            "*.toml",
            "environment*.yml",
            "environment*.yaml",
            ".gitignore",
            "src/**/*.py",
            "configs/**/*.yaml",
            "configs/**/*.yml",
            "scripts/**/*.sh",
            "docs/**/*.md",
        ]
        copied: set[Path] = set()
        for pattern in patterns:
            for source in self.root.glob(pattern):
                if not source.is_file() or source.name.endswith(".local.md"):
                    continue
                relative = source.relative_to(self.root)
                if relative in copied or "runs" in relative.parts:
                    continue
                destination = self.source_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied.add(relative)

    def register_artifact(
        self,
        path: str | Path,
        kind: str,
        server_only: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        artifact_path = Path(path).expanduser().resolve()
        record: dict[str, Any] = {
            "path": str(artifact_path),
            "kind": kind,
            "server_only": server_only,
            "exists": artifact_path.exists(),
            "metadata": metadata or {},
        }
        if artifact_path.is_file():
            record["size_bytes"] = artifact_path.stat().st_size
            record["sha256"] = _sha256(artifact_path)
        self._artifacts.append(record)
        self._write_artifact_manifest()
        return record

    def _write_artifact_manifest(self) -> None:
        atomic_json_dump(
            {"run_id": self.run_id, "artifacts": self._artifacts},
            self.run_dir / "artifact_manifest.json",
        )
