"""TF-free split loading + preprocessing for calibration and benchmarks.

Discovers samples from on-disk ``train_data`` / ``val_data`` / ``test_data``
and preprocesses with the (TF-free) OpenCV pipeline — no TensorFlow import.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np


def load_split(
    config: dict[str, Any],
    *,
    split: str,
    max_samples: int | None,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[str]]:
    """Load + preprocess up to ``max_samples`` images from a data split."""
    from vie_handwritten.dataset import load_split as discover_split
    from vie_handwritten.preprocess import load_image, preprocess
    from vie_handwritten.utils import abs_path

    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split={split}")
    records = discover_split(config, split)  # type: ignore[arg-type]
    if max_samples and len(records) > max_samples:
        records = random.Random(seed).sample(records, max_samples)

    root = abs_path(config["data"]["root_dir"])
    pp = config["preprocess"]
    arrays, texts = [], []
    for rec in records:
        image = load_image(str(root / rec["image"]))
        arrays.append(preprocess(image, pp))
        texts.append(rec["text"])
    return arrays, texts
