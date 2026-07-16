"""Configuration loading (YAML)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a nested dict.

    Expected keys: ``data``, ``preprocess``, ``model``, ``ctc``, ``train``, ``eval``.
    """
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Write config dict back to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
