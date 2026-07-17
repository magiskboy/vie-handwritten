"""Shared helpers: config I/O, seeding, paths, GPU runtime."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

_RUNTIME_CONFIGURED = False

# Standard checkpoint directory layout (written by train, read by infer/eval/GUI):
#   <checkpoint>/
#     model.weights.h5   # Keras 3 requires the ``.weights.h5`` suffix
#     config.yaml        # copy of the train config used to build the model
WEIGHTS_NAME = "model.weights.h5"
CHECKPOINT_CONFIG_NAME = "config.yaml"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a nested dict."""
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


def resolve_checkpoint_dir(checkpoint: str | Path) -> Path:
    """Validate a checkpoint directory containing ``model.weights.h5`` + ``config.yaml``."""
    root = Path(checkpoint)
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint must be a directory: {checkpoint}")
    missing = [
        name
        for name in (WEIGHTS_NAME, CHECKPOINT_CONFIG_NAME)
        if not (root / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint {root} is incomplete; missing: {', '.join(missing)}"
        )
    return root.resolve()


def checkpoint_weights_path(checkpoint: str | Path) -> Path:
    """Absolute path to ``model.weights.h5`` inside a checkpoint directory."""
    return resolve_checkpoint_dir(checkpoint) / WEIGHTS_NAME


def load_checkpoint_config(checkpoint: str | Path) -> dict[str, Any]:
    """Load ``config.yaml`` from a checkpoint directory."""
    return load_config(resolve_checkpoint_dir(checkpoint) / CHECKPOINT_CONFIG_NAME)


def save_checkpoint_config(config: dict[str, Any], checkpoint_dir: str | Path) -> Path:
    """Copy the train config into ``checkpoint_dir/config.yaml``."""
    path = Path(checkpoint_dir) / CHECKPOINT_CONFIG_NAME
    save_config(config, path)
    return path


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow RNGs for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if missing; return Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_root() -> Path:
    """Return repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def abs_path(path: str | Path) -> Path:
    """Resolve a possibly-relative path against the project root."""
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def charset_path(config: dict[str, Any]) -> Path:
    """Absolute path to the charset file declared in ``config['data']``."""
    return abs_path(config["data"]["charset_path"])


def configure_runtime(*, memory_growth: bool = True) -> dict:
    """Enable GPU memory growth once before any tensor allocation (CUDA/CPU)."""
    global _RUNTIME_CONFIGURED
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    info = {
        "tensorflow": tf.__version__,
        "gpu_count": len(gpus),
        "gpus": [gpu.name for gpu in gpus],
    }
    if _RUNTIME_CONFIGURED:
        return info

    if gpus and memory_growth:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            logger.warning("Could not set GPU memory growth: %s", exc)

    if gpus:
        logger.info("TensorFlow %s using %d GPU(s): %s", tf.__version__, len(gpus), info["gpus"])
    else:
        logger.warning("No GPU detected — training will run on CPU.")

    _RUNTIME_CONFIGURED = True
    return info
