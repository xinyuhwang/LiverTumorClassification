from __future__ import annotations

from pathlib import Path

from liver_tumor.config import load_config


def test_load_config_expands_environment_and_applies_override(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("TEST_DATA_ROOT", str(tmp_path / "dataset"))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "dataset:\n  root: ${TEST_DATA_ROOT}\nexperiment:\n  seed: 42\n",
        encoding="utf-8",
    )

    config = load_config(config_path, ["experiment.seed=7", "dataset.enabled=true"])

    assert config["dataset"]["root"] == str(tmp_path / "dataset")
    assert config["dataset"]["enabled"] is True
    assert config["experiment"]["seed"] == 7
    assert config["_meta"]["config_path"] == str(config_path.resolve())
