"""YAML configuration loading with environment expansion and CLI overrides."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Iterable

import yaml


_ENV_PATTERN = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(os.path.expanduser(value))
        unresolved = _ENV_PATTERN.findall(expanded)
        if unresolved:
            names = ", ".join(unresolved)
            raise ValueError(f"Unresolved environment variable(s): {names}")
        return expanded
    return value


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    if any(not key for key in keys):
        raise ValueError(f"Invalid override key: {dotted_key!r}")
    cursor = config
    for key in keys[:-1]:
        current = cursor.get(key)
        if current is None:
            current = {}
            cursor[key] = current
        if not isinstance(current, dict):
            raise ValueError(f"Cannot set {dotted_key!r}: {key!r} is not a mapping")
        cursor = current
    cursor[keys[-1]] = value


def parse_overrides(items: Iterable[str] | None) -> list[tuple[str, Any]]:
    parsed: list[tuple[str, Any]] = []
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Override must use key=value syntax: {item!r}")
        key, raw_value = item.split("=", 1)
        parsed.append((key.strip(), yaml.safe_load(raw_value)))
    return parsed


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Top-level YAML value must be a mapping: {config_path}")

    config = copy.deepcopy(loaded)
    for key, value in parse_overrides(overrides):
        _set_nested(config, key, value)
    config = _expand_environment(config)
    config.setdefault("_meta", {})["config_path"] = str(config_path)
    return config


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=False)
