#!/usr/bin/env python3
"""Download MCT-LTDiag from Harvard Dataverse with resume and inventory support."""

from __future__ import annotations

import argparse

from liver_tumor.config import load_config
from liver_tumor.download import DataverseDownloader, token_from_environment
from liver_tumor.experiment import RunContext, atomic_json_dump


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML configuration path")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--include-regex")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--delete-archives", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.set)
    dataset = config["dataset"]
    options = config.get("download", {})
    token = token_from_environment(str(dataset.get("token_env", "DATAVERSE_API_TOKEN")))
    if not token:
        print("Warning: no Dataverse API token found; public metadata may still be accessible")

    with RunContext(config, task=str(config["experiment"].get("task", "dataset-download"))) as run:
        downloader = DataverseDownloader(
            persistent_id=str(dataset["persistent_id"]),
            destination=dataset["root"],
            token=token,
            base_url=str(dataset.get("base_url", "https://dataverse.harvard.edu")),
            timeout_seconds=int(options.get("timeout_seconds", 180)),
            chunk_size_mb=int(options.get("chunk_size_mb", 8)),
        )
        inventory = downloader.run(
            include_regex=args.include_regex or options.get("include_regex"),
            max_files=args.max_files or options.get("max_files"),
            extract=not args.no_extract and bool(options.get("extract", True)),
            delete_archives=args.delete_archives or bool(options.get("delete_archives", False)),
            free_space_multiplier=float(options.get("free_space_multiplier", 2.2)),
            metadata_only=args.metadata_only or bool(options.get("metadata_only", False)),
        )
        compact_path = run.report_dir / "download_inventory.json"
        atomic_json_dump(inventory, compact_path)
        run.register_artifact(compact_path, kind="download_inventory")
        if downloader.inventory_path.is_file():
            run.register_artifact(
                downloader.inventory_path,
                kind="server_dataset_inventory",
                server_only=True,
            )
        print(
            f"Selected {inventory['dataset']['selected_file_count']} / "
            f"{inventory['dataset']['all_file_count']} Dataverse files"
        )


if __name__ == "__main__":
    main()
