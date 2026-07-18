"""Dataset discovery and the ``tf.data`` pipeline.

On-disk layout (HWDB, official split by writer)::

    <dataset_dir>/                    # e.g. data/images/HWDB_line
      train_data/<writer_id>/
        1.jpg, 2.jpg, ...
        label.json                    # {"1.jpg": "văn bản", ...}
      val_data/<writer_id>/...        # held-out writers (materialized once)
      test_data/<writer_id>/...

``dataset_dir`` is the full path (from the project root) to the dataset,
chosen via ``config['data']['dataset_dir']`` (default
``data/images/HWDB_line``); switching it to ``data/images/HWDB_word`` etc.
lets the same pipeline drive a curriculum such as word-first pretraining →
line fine-tuning. ``root_dir`` is the base that image paths are stored
relative to.

Splits are on-disk subdirs: ``train`` / ``val`` / ``test`` map to
``train_data`` / ``val_data`` / ``test_data``.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path
from typing import Any, Literal

from vie_handwritten.utils import abs_path

logger = logging.getLogger(__name__)

DEFAULT_DATASET_DIR = "data/images/HWDB_line"
SplitName = Literal["train", "val", "test"]
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
_SPLIT_SUBDIR_KEYS = {
    "train": ("train_subdir", "train_data"),
    "val": ("val_subdir", "val_data"),
    "test": ("test_subdir", "test_data"),
}


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


def discover(root_dir: Path, dataset_root: Path, subdir: str) -> list[dict[str, str]]:
    """Discover ``{image, text, writer}`` records for one split subdir of ``dataset_root``.

    ``dataset_root`` is the absolute dataset path (e.g. ``.../data/images/HWDB_line``);
    image paths in the returned records are stored relative to ``root_dir``.
    """
    src_root = dataset_root / subdir
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


def split_subdir(config: dict[str, Any], split: SplitName) -> str:
    """Config subdir name for ``train`` / ``val`` / ``test``."""
    if split not in _SPLIT_SUBDIR_KEYS:
        raise ValueError(f"Unknown split={split!r}; expected train|val|test")
    key, default = _SPLIT_SUBDIR_KEYS[split]
    return str(config["data"].get(key, default))


def _oov_vocab(config: dict[str, Any]) -> set[str] | None:
    """Charset characters (no blank) for OOV filtering, or ``None`` to skip."""
    if not bool(config["data"].get("drop_oov", True)):
        return None
    from vie_handwritten.charset import Charset

    declared = abs_path(config["data"]["charset_path"])
    path = declared if declared.is_file() else abs_path("data/charset/vietnamese.txt")
    if not path.is_file():
        raise FileNotFoundError(
            f"Charset for OOV filter not found (tried {declared} and {path})"
        )
    if path != declared:
        logger.warning("charset_path %s missing; using %s for OOV filter", declared, path)
    charset = Charset.from_file(path)
    return set(charset.characters[1:])  # exclude blank at index 0


def load_split(config: dict[str, Any], split: SplitName) -> list[dict[str, str]]:
    """Discover records for a split; NFC-normalize text and optionally drop OOV."""
    data_cfg = config["data"]
    root_dir = abs_path(data_cfg["root_dir"])
    dataset_root = abs_path(data_cfg.get("dataset_dir", DEFAULT_DATASET_DIR))
    subdir = split_subdir(config, split)
    records = discover(root_dir, dataset_root, subdir)

    vocab = _oov_vocab(config)

    out: list[dict[str, str]] = []
    dropped = 0
    for rec in records:
        text = unicodedata.normalize("NFC", rec["text"])
        if vocab is not None and any(ch not in vocab for ch in text):
            dropped += 1
            continue
        out.append({"image": rec["image"], "text": text, "writer": rec["writer"]})
    if dropped:
        logger.info("%s: dropped %d OOV samples", split, dropped)
    logger.info("%s: %d samples from %s/%s", split, len(out), dataset_root.name, subdir)
    return out


def resolve_image_path(config: dict[str, Any], record: dict[str, str]) -> Path:
    """Absolute image path for a discovered record."""
    return abs_path(config["data"]["root_dir"]) / record["image"]


def build_dataset(
    records: list[dict[str, str]],
    *,
    charset: Any,
    config: dict[str, Any],
    training: bool,
):
    """Build a finite ``tf.data`` pipeline (one pass = one epoch).

    Yields ``({"image", "label_length", "input_length"}, labels)`` batches with
    variable width padded per batch.
    """
    import numpy as np
    import tensorflow as tf

    from vie_handwritten.model import WIDTH_DOWNSAMPLE
    from vie_handwritten.preprocess import load_image, preprocess

    pp = config["preprocess"]
    root_dir = abs_path(config["data"]["root_dir"])
    height = int(pp.get("target_height", 64))
    channels = int(pp.get("channels", 3))
    batch_size = int(config["train"]["batch_size"])
    # Single source of truth for width→time-step ratio (default = backbone stride product).
    width_downsample = int(config.get("model", {}).get("width_downsample", WIDTH_DOWNSAMPLE))

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
    # Drop samples whose time-step budget can't fit the label (T < label_length):
    # a too-narrow image vs a long label makes CTC invalid (+inf). These are rare
    # bad/mislabeled samples; skipping them keeps training robust across the full set.
    def _fits(image, label, label_len, width):
        return tf.maximum(width // width_downsample, 1) >= label_len

    ds = ds.filter(_fits)
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
                "input_length": tf.maximum(widths // width_downsample, 1),
            },
            labels,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.prefetch(tf.data.AUTOTUNE)
