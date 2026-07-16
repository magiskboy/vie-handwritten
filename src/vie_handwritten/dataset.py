"""Dataset discovery, manifest building, and the ``tf.data`` pipeline (line only).

On-disk layout (HWDB, official split by writer)::

    <root_dir>/                       # e.g. data/images
      HWDB_line/
        train_data/<writer_id>/
          1.jpg, 2.jpg, ...
          label.json                  # {"1.jpg": "văn bản dòng", ...}
        test_data/<writer_id>/...

Split is writer-independent: ``test`` = official ``test_data``; ``val`` = a
fraction of writers held out from ``train_data``. ``build-data`` writes
normalized JSONL manifests to ``data/manifests/{train,val,test}.jsonl`` so
train/eval only ever read manifests, decoupled from the disk layout.
"""

from __future__ import annotations

import json
import logging
import random
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from vie_handwritten.preprocess import load_image, preprocess
from vie_handwritten.utils import project_root

logger = logging.getLogger(__name__)

LINE_DIR = "HWDB_line"
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _abs_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def _read_writer_labels(writer_dir: Path) -> dict[str, str]:
    """Read ``label.json`` (or legacy ``labels.json``) → ``{filename: text}``."""
    for name in ("label.json", "labels.json"):
        f = writer_dir / name
        if f.is_file():
            with f.open(encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                logger.warning("Skipping non-object label file: %s", f)
                return {}
            return {str(k): str(v) for k, v in data.items()}
    return {}


def _resolve_image(writer_dir: Path, name: str) -> Path | None:
    exact = writer_dir / name
    if exact.is_file():
        return exact
    stem = Path(name).stem
    for cand in writer_dir.glob(f"{stem}.*"):
        if cand.is_file() and cand.suffix.lower() in _IMAGE_EXTS:
            return cand
    return None


def discover(root_dir: Path, subdir: str) -> list[dict[str, str]]:
    """Discover ``{image, text, writer}`` records for one HWDB_line split subdir."""
    src_root = root_dir / LINE_DIR / subdir
    records: list[dict[str, str]] = []
    if not src_root.is_dir():
        logger.warning("Source dir missing: %s", src_root)
        return records
    missing = 0
    for writer_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        for name, text in _read_writer_labels(writer_dir).items():
            img = _resolve_image(writer_dir, name)
            if img is None:
                missing += 1
                continue
            records.append(
                {
                    "image": str(img.relative_to(root_dir)),
                    "text": text,
                    "writer": writer_dir.name,
                }
            )
    if missing:
        logger.warning("%s: %d labeled images not found on disk", subdir, missing)
    return records


def build_manifests(config: dict[str, Any]) -> dict[str, Path]:
    """Scan HWDB_line, split by writer, filter OOV, write JSONL manifests.

    Returns ``{"train": path, "val": path, "test": path}``.
    """
    from vie_handwritten.charset import Charset

    data_cfg = config["data"]
    root_dir = _abs_path(data_cfg["root_dir"])
    manifest_dir = _abs_path(data_cfg.get("manifest_dir", "data/manifests"))
    manifest_dir.mkdir(parents=True, exist_ok=True)
    val_ratio = float(data_cfg.get("val_writers_ratio", 0.1))
    drop_oov = bool(data_cfg.get("drop_oov", True))
    seed = int(config.get("project", {}).get("seed", 42))

    charset = Charset.from_file(_abs_path(data_cfg["charset_path"]))
    vocab = set(charset.characters[1:])  # exclude blank at index 0

    train_recs = discover(root_dir, data_cfg.get("train_subdir", "train_data"))
    test_recs = discover(root_dir, data_cfg.get("test_subdir", "test_data"))

    writers = sorted({r["writer"] for r in train_recs})
    random.Random(seed).shuffle(writers)
    n_val = max(1, round(len(writers) * val_ratio)) if writers else 0
    val_writers = set(writers[:n_val])
    logger.info("Writers: total=%d val=%d train=%d", len(writers), len(val_writers), len(writers) - len(val_writers))

    buckets: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    dropped = 0

    def _accept(rec: dict[str, str], split: str) -> None:
        nonlocal dropped
        text = unicodedata.normalize("NFC", rec["text"])
        if drop_oov and any(ch not in vocab for ch in text):
            dropped += 1
            return
        buckets[split].append({"image": rec["image"], "text": text, "writer": rec["writer"], "split": split})

    for r in train_recs:
        _accept(r, "val" if r["writer"] in val_writers else "train")
    for r in test_recs:
        _accept(r, "test")

    paths: dict[str, Path] = {}
    for split, recs in buckets.items():
        out = manifest_dir / f"{split}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        paths[split] = out

    summary = {
        "root_dir": str(root_dir),
        "seed": seed,
        "drop_oov": drop_oov,
        "dropped_oov": dropped,
        "counts": {k: len(v) for k, v in buckets.items()},
    }
    (manifest_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Manifests → %s | counts=%s dropped_oov=%d", manifest_dir, summary["counts"], dropped)
    return paths


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    """Load a JSONL manifest into a list of record dicts."""
    records: list[dict[str, str]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def manifest_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Expected manifest paths (does not build them)."""
    base = _abs_path(config["data"].get("manifest_dir", "data/manifests"))
    return {s: base / f"{s}.jsonl" for s in ("train", "val", "test")}


def ensure_manifests(config: dict[str, Any], *, rebuild: bool = False) -> dict[str, Path]:
    """Build manifests if missing (or ``rebuild``), else return existing paths."""
    paths = manifest_paths(config)
    if rebuild or not all(p.is_file() for p in paths.values()):
        return build_manifests(config)
    return paths


def resolve_image_path(config: dict[str, Any], record: dict[str, str]) -> Path:
    """Absolute image path for a manifest record."""
    return _abs_path(config["data"]["root_dir"]) / record["image"]


def build_dataset(
    records: list[dict[str, str]],
    *,
    charset: Any,
    config: dict[str, Any],
    training: bool,
) -> tf.data.Dataset:
    """Build a finite ``tf.data`` pipeline (one pass = one epoch).

    Yields ``({"image", "label_length", "input_length"}, labels)`` batches with
    variable width padded per batch.
    """
    pp = config["preprocess"]
    root_dir = _abs_path(config["data"]["root_dir"])
    height = int(pp.get("target_height", 64))
    channels = int(pp.get("channels", 3))
    batch_size = int(config["train"]["batch_size"])

    paths = [str(root_dir / r["image"]) for r in records]
    texts = [r["text"] for r in records]

    def _load(path: tf.Tensor, text: tf.Tensor):
        def _py(path_bytes, text_bytes):
            arr = preprocess(load_image(path_bytes.numpy().decode("utf-8")), pp)
            encoded = charset.encode(text_bytes.numpy().decode("utf-8"))
            return (
                arr.astype(np.float32),
                np.array(encoded, dtype=np.int32),
                np.int32(len(encoded)),
                np.int32(arr.shape[1]),
            )

        image, label, label_len, width = tf.py_function(
            _py, [path, text], [tf.float32, tf.int32, tf.int32, tf.int32]
        )
        image.set_shape([height, None, channels])
        label.set_shape([None])
        label_len.set_shape([])
        width.set_shape([])
        return image, label, label_len, width

    pad_value = 0.0
    if pp.get("normalize") == "imagenet":
        white = (1.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        pad_value = float(white.mean())

    ds = tf.data.Dataset.from_tensor_slices((paths, texts))
    if training:
        ds = ds.shuffle(min(len(paths), 4096), reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.padded_batch(
        batch_size,
        padded_shapes=([height, None, channels], [None], [], []),
        padding_values=(
            tf.constant(pad_value, dtype=tf.float32),
            tf.constant(0, dtype=tf.int32),
            tf.constant(0, dtype=tf.int32),
            tf.constant(0, dtype=tf.int32),
        ),
    )
    ds = ds.map(
        lambda images, labels, label_lens, widths: (
            {
                "image": images,
                "label_length": label_lens,
                "input_length": tf.maximum(widths // 8, 1),
            },
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.prefetch(tf.data.AUTOTUNE)
