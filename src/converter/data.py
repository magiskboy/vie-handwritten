"""TF-free manifest loading + preprocessing for calibration and benchmarks.

Reads the JSONL manifests produced by ``vie-ocr build-data`` and preprocesses
images with the (TF-free) OpenCV pipeline. Manifests must already exist; this
never triggers the TF-bound dataset builder.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from vie_handwritten.utils import abs_path


def manifest_paths(config: dict[str, Any]) -> dict[str, Path]:
    base = abs_path(config["data"].get("manifest_dir", "data/manifests"))
    return {s: base / f"{s}.jsonl" for s in ("train", "val", "test")}


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_split(
    config: dict[str, Any],
    *,
    split: str,
    max_samples: int | None,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[str]]:
    """Load + preprocess up to ``max_samples`` images from a manifest split."""
    from vie_handwritten.preprocess import load_image, preprocess

    paths = manifest_paths(config)
    if split not in paths:
        raise ValueError(f"Unknown split={split}")
    manifest = paths[split]
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Manifest missing: {manifest}. Run `vie-ocr build-data` first."
        )
    records = load_manifest(manifest)
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
