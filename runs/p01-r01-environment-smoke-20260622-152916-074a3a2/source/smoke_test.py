#!/usr/bin/env python3
"""Check the Conda environment, GPU visibility, run capture, and optional sample data."""

from __future__ import annotations

import argparse
import importlib

from liver_tumor.config import load_config
from liver_tumor.experiment import RunContext, atomic_json_dump


REQUIRED_IMPORTS = [
    "numpy",
    "pandas",
    "scipy",
    "skimage",
    "sklearn",
    "nibabel",
    "matplotlib",
    "yaml",
    "requests",
    "tqdm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_local_sample.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--skip-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.set)
    with RunContext(config, task="environment-smoke") as run:
        versions = {}
        for module_name in REQUIRED_IMPORTS:
            module = importlib.import_module(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
            print(f"import {module_name:<12} OK  {versions[module_name]}")

        try:
            import torch

            versions["torch"] = torch.__version__
            cuda_available = torch.cuda.is_available()
            print(f"CUDA available: {cuda_available}")
            if args.require_cuda and not cuda_available:
                raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")
            device = torch.device("cuda" if cuda_available else "cpu")
            tensor = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
            tensor_check = float((tensor @ tensor.T).sum().cpu())
        except ImportError:
            if args.require_cuda:
                raise RuntimeError("PyTorch is required for the CUDA smoke test") from None
            import numpy as np

            cuda_available = False
            device = "cpu-numpy"
            tensor = np.arange(16, dtype=np.float32).reshape(4, 4)
            tensor_check = float((tensor @ tensor.T).sum())
            versions["torch"] = "not installed (allowed for local Phase 1 smoke)"

        data_check = None
        if not args.skip_data:
            from liver_tumor.data import discover_cases, validate_case

            records = discover_cases(config["dataset"]["root"])
            if not records:
                raise RuntimeError("No sample cases found for data smoke test")
            data_check = validate_case(records[0], load_intensities=False)
            if data_check.get("error_count", 0):
                raise RuntimeError(f"Sample data validation failed: {data_check['issues']}")
            print(f"Validated sample case: {records[0].case_id}")

        output = run.report_dir / "smoke_test.json"
        atomic_json_dump(
            {
                "versions": versions,
                "cuda_available": cuda_available,
                "device": str(device),
                "tensor_check": tensor_check,
                "sample_case": data_check,
            },
            output,
        )
        run.register_artifact(output, kind="smoke_test")


if __name__ == "__main__":
    main()
