#!/usr/bin/env python3
"""Validate and summarize MCT-LTDiag without modifying source data."""

from __future__ import annotations

import argparse

from liver_tumor.config import load_config
from liver_tumor.data import audit_dataset, create_split_manifest
from liver_tumor.experiment import RunContext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML configuration path")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--skip-intensities", action="store_true")
    parser.add_argument("--qc-cases-per-type", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.set)
    dataset = config["dataset"]
    task = str(config["experiment"].get("task", "dataset-audit"))

    with RunContext(config, task=task) as run:
        audit_dir = run.report_dir / "dataset_audit"
        summary = audit_dataset(
            root=dataset["root"],
            output_dir=audit_dir,
            max_cases=args.max_cases
            if args.max_cases is not None
            else dataset.get("max_cases"),
            load_intensities=not args.skip_intensities
            and bool(dataset.get("load_intensities", True)),
            qc_cases_per_type=args.qc_cases_per_type
            if args.qc_cases_per_type is not None
            else int(dataset.get("qc_cases_per_type", 1)),
        )
        for path in sorted(audit_dir.glob("*")):
            if path.is_file():
                run.register_artifact(path, kind="dataset_audit")
        for path in sorted((audit_dir / "qc").glob("*.png")):
            run.register_artifact(path, kind="dataset_qc_image")

        split = config.get("split", {})
        protocol = str(split.get("protocol", "none"))
        if protocol != "none":
            split_path = run.report_dir / "split_manifest.json"
            create_split_manifest(
                cases_csv=audit_dir / "cases.csv",
                output_path=split_path,
                protocol=protocol,
                seed=int(split.get("seed", config["experiment"].get("seed", 42))),
                train_ratio=float(split.get("train_ratio", 0.70)),
                val_ratio=float(split.get("val_ratio", 0.15)),
                test_ratio=float(split.get("test_ratio", 0.15)),
                n_folds=int(split.get("n_folds", 5)),
            )
            run.register_artifact(split_path, kind="split_manifest")
        print(f"Dataset audit summary: {summary}")


if __name__ == "__main__":
    main()
