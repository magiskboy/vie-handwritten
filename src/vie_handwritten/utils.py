"""Shared helpers (I/O, seeding, paths, GPU runtime)."""

from __future__ import annotations

import logging
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_RUNTIME_CONFIGURED = False


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


def count_parameters(model: Any) -> int:
    """Return number of trainable parameters."""
    return int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))


def configure_runtime(*, memory_growth: bool = True) -> dict:
    """Configure TensorFlow devices before any model/tensor allocation.

    - macOS Apple Silicon: expects ``tensorflow-metal`` PluggableDevice.
    - Linux NVIDIA: uses all visible CUDA GPUs without a memory cap.
    - Enables memory growth on every GPU when requested.
    """
    global _RUNTIME_CONFIGURED
    if _RUNTIME_CONFIGURED:
        return _runtime_info()

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    info = {
        "platform": sys.platform,
        "machine": platform.machine(),
        "tensorflow": tf.__version__,
        "gpu_count": len(gpus),
        "gpus": [gpu.name for gpu in gpus],
        "backend": _detect_backend(gpus),
    }

    if gpus and memory_growth:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Enabled memory growth on %d GPU(s)", len(gpus))
        except RuntimeError as exc:
            logger.warning("Could not set GPU memory growth: %s", exc)

    if info["backend"] == "cpu":
        logger.warning(
            "No GPU detected (platform=%s machine=%s). Training will use CPU. "
            "On Apple Silicon install tensorflow-metal; on Linux NVIDIA ensure "
            "CUDA is available (try: pip install 'tensorflow[and-cuda]').",
            info["platform"],
            info["machine"],
        )
    else:
        logger.info(
            "TensorFlow %s using %s (%d device(s)): %s",
            info["tensorflow"],
            info["backend"],
            info["gpu_count"],
            info["gpus"],
        )

    _RUNTIME_CONFIGURED = True
    return info


def _detect_backend(gpus: list) -> str:
    if not gpus:
        return "cpu"
    if sys.platform == "darwin":
        return "metal"
    return "cuda"


def _runtime_info() -> dict:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    return {
        "platform": sys.platform,
        "machine": platform.machine(),
        "tensorflow": tf.__version__,
        "gpu_count": len(gpus),
        "gpus": [gpu.name for gpu in gpus],
        "backend": _detect_backend(gpus),
    }
