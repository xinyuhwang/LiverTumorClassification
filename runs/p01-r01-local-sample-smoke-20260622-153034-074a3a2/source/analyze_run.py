#!/usr/bin/env python3
"""Create a compact Markdown review of one or more experiment run directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_run(run_dir: Path) -> str:
    status = read_json(run_dir / "status.json")
    git_info = read_json(run_dir / "git_info.json")
    environment = read_json(run_dir / "environment.json")
    artifacts = read_json(run_dir / "artifact_manifest.json").get("artifacts", [])
    torch_info = environment.get("torch", {})
    lines = [
        f"# Run Review: {run_dir.name}",
        "",
        f"- Status: `{status.get('status', 'UNKNOWN')}`",
        f"- Task: `{status.get('task', 'unknown')}`",
        f"- Phase/Round: `{status.get('phase', '?')}` / `{status.get('round', '?')}`",
        f"- Git: `{git_info.get('branch', '?')}` @ `{git_info.get('short_commit', '?')}`",
        f"- Dirty source: `{git_info.get('dirty', '?')}`",
        f"- Host: `{environment.get('hostname', '?')}`",
        f"- Python: `{str(environment.get('python', '?')).splitlines()[0]}`",
        f"- PyTorch: `{torch_info.get('version', 'not installed')}`",
        f"- CUDA available: `{torch_info.get('cuda_available', False)}`",
        "",
        "## Artifacts",
        "",
    ]
    if artifacts:
        lines.extend(
            f"- `{artifact.get('kind')}`: `{artifact.get('path')}` "
            f"({artifact.get('size_bytes', 'n/a')} bytes)"
            for artifact in artifacts
        )
    else:
        lines.append("- No registered artifacts.")

    audit_summary = read_json(run_dir / "reports" / "dataset_audit" / "summary.json")
    if audit_summary:
        lines.extend(
            [
                "",
                "## Dataset Audit",
                "",
                f"- Cases: {audit_summary.get('case_count')}",
                f"- Cases with errors: {audit_summary.get('cases_with_errors')}",
                f"- Cases with warnings: {audit_summary.get('cases_with_warnings')}",
                f"- Tumor types: `{audit_summary.get('tumor_type_counts')}`",
            ]
        )

    log_path = run_dir / "logs" / "run.log"
    if log_path.is_file():
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        lines.extend(["", "## Log Tail", "", "```text", *tail, "```"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports = [summarize_run(path.expanduser().resolve()) for path in args.run_dirs]
    combined = "\n---\n\n".join(reports)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(combined, encoding="utf-8")
        print(args.output)
    else:
        print(combined)


if __name__ == "__main__":
    main()
