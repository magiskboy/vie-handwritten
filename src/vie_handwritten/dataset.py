"""Dataset loading and ``tf.data`` pipelines.

Expected on-disk layout (see ``data/vn_handwritten_images``)::

    <dataset_dir>/
      labels.json          # {"1.jpg": "Số 3 Nguyễn Ngọc Vũ, Hà Nội", ...}
      data/
        1.jpg
        0001_samples.png
        ...

``labels.json`` keys are filenames relative to ``data/`` (images_subdir).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from vie_handwritten.preprocess import (
    load_image,
    normalized_pad_value,
    preprocess,
)

logger = logging.getLogger(__name__)


def load_labels(labels_path: str | Path) -> dict[str, str]:
    """Load ``labels.json`` → ``{filename: transcription}``."""
    path = Path(labels_path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"labels.json must be an object: {path}")
    return {str(k): str(v) for k, v in data.items()}


def discover_samples(
    dataset_dir: str | Path,
    *,
    images_subdir: str = "data",
    labels_file: str = "labels.json",
    skip_missing: bool = True,
    max_missing_ratio: float = 0.1,
) -> list[tuple[Path, str]]:
    """Discover ``(image_path, transcription)`` pairs.

    Missing images are skipped with a warning when ``skip_missing`` is set, as
    long as the fraction of missing files stays within ``max_missing_ratio``;
    otherwise a ``FileNotFoundError`` is raised so a broken dataset fails loudly.
    """
    dataset_dir = Path(dataset_dir)
    labels = load_labels(dataset_dir / labels_file)
    images_dir = dataset_dir / images_subdir
    samples: list[tuple[Path, str]] = []
    missing: list[str] = []
    for name, text in labels.items():
        path = images_dir / name
        if not path.is_file():
            missing.append(name)
            continue
        samples.append((path, text))
    if missing:
        ratio = len(missing) / max(1, len(labels))
        detail = (
            f"{len(missing)} labeled images missing under {images_dir} "
            f"(e.g. {missing[:5]})"
        )
        if not skip_missing or ratio > max_missing_ratio:
            raise FileNotFoundError(
                f"{detail}; missing ratio {ratio:.1%} exceeds allowed "
                f"{max_missing_ratio:.0%}"
            )
        logger.warning("Skipping %s (%.1f%% of dataset)", detail, ratio * 100)
    if not samples:
        raise ValueError(f"No samples found in {dataset_dir}")
    return samples


def train_val_test_split(
    samples: list[tuple[Path, str]],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> tuple[list, list, list]:
    """Split samples with scikit-learn."""
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1, got {total}")
    if val_ratio + test_ratio <= 0:
        raise ValueError("val_split + test_split must be > 0 for evaluation")
    train_samples, temp = train_test_split(
        samples, train_size=train_ratio, random_state=seed, shuffle=True
    )
    if test_ratio == 0:
        return train_samples, temp, []
    if val_ratio == 0:
        return train_samples, [], temp
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_samples, test_samples = train_test_split(
        temp, train_size=relative_val, random_state=seed, shuffle=True
    )
    return train_samples, val_samples, test_samples


def encode_batch(
    images: list,
    texts: list[str],
    *,
    charset: Any,
    preprocess_config: dict[str, Any],
) -> dict[str, Any]:
    """Preprocess images and encode labels for one CTC training batch."""
    processed = [preprocess(img, preprocess_config) for img in images]
    max_w = max(p.shape[1] for p in processed)
    max_w = min(max_w, int(preprocess_config.get("max_width", max_w)))
    pad_value = int(preprocess_config.get("pad_value", 255))
    # After normalize, pad with ImageNet-normalized white ≈ (1-mean)/std for white
    # Prefer padding before normalize in preprocess; here images already normalized.
    # Pad with zeros in feature space (neutral after centering is imperfect but OK).
    padded = []
    for p in processed:
        if p.shape[1] < max_w:
            h, w, c = p.shape
            canvas = np.zeros((h, max_w, c), dtype=np.float32)
            canvas[:, :w] = p
            # approximate white pad in imagenet space
            if preprocess_config.get("normalize") == "imagenet":
                white = (np.ones(3, dtype=np.float32) - np.array([0.485, 0.456, 0.406])) / np.array(
                    [0.229, 0.224, 0.225]
                )
                canvas[:, w:] = white.reshape(1, 1, 3)
            elif pad_value:
                canvas[:, w:] = pad_value / 255.0 if p.dtype != np.uint8 else pad_value
            padded.append(canvas)
        else:
            padded.append(p[:, :max_w])

    encoded = [charset.encode(t) for t in texts]
    max_label = max((len(e) for e in encoded), default=1)
    labels = np.zeros((len(encoded), max_label), dtype=np.int32)
    label_length = np.zeros((len(encoded),), dtype=np.int32)
    for i, e in enumerate(encoded):
        labels[i, : len(e)] = e
        label_length[i] = len(e)

    batch_images = np.stack(padded, axis=0).astype(np.float32)
    # Approximate CTC input length from width after ResNet HTR downsampling (~/8)
    input_length = np.full((len(padded),), max(1, max_w // 8), dtype=np.int32)
    return {
        "images": batch_images,
        "labels": labels,
        "label_length": label_length,
        "input_length": input_length,
    }


def build_tf_dataset(
    samples: list[tuple[Path, str]],
    *,
    charset: Any,
    config: dict[str, Any],
    training: bool = True,
) -> Any:
    """Build a ``tf.data.Dataset`` yielding model inputs + CTC targets."""
    preprocess_config = config["preprocess"]
    batch_size = int(config["train"]["batch_size"])
    seed = int(config.get("project", {}).get("seed", 42))
    curriculum_epochs = int(config["train"].get("curriculum_short_bias_epochs", 0))

    paths = [str(p) for p, _ in samples]
    texts = [t for _, t in samples]
    lengths = np.array([len(t) for t in texts], dtype=np.int32)

    def _load_one(path: tf.Tensor, text: tf.Tensor) -> tuple:
        def _py(path_bytes, text_bytes):
            p = path_bytes.numpy().decode("utf-8")
            t = text_bytes.numpy().decode("utf-8")
            img = load_image(p)
            arr = preprocess(img, preprocess_config)
            encoded = charset.encode(t)
            return (
                arr.astype(np.float32),
                np.array(encoded, dtype=np.int32),
                np.int32(len(encoded)),
                np.int32(arr.shape[1]),
            )

        image, label, label_len, width = tf.py_function(
            _py,
            [path, text],
            [tf.float32, tf.int32, tf.int32, tf.int32],
        )
        channels = int(preprocess_config.get("channels", 3))
        height = int(preprocess_config.get("target_height", 64))
        image.set_shape([height, None, channels])
        label.set_shape([None])
        label_len.set_shape([])
        width.set_shape([])
        return image, label, label_len, width

    ds = tf.data.Dataset.from_tensor_slices((paths, texts))
    if training:
        if curriculum_epochs > 0:
            # Prefer shorter labels early: sort by length then shuffle lightly
            order = np.argsort(lengths)
            paths_sorted = [paths[i] for i in order]
            texts_sorted = [texts[i] for i in order]
            ds = tf.data.Dataset.from_tensor_slices((paths_sorted, texts_sorted))
        ds = ds.shuffle(buffer_size=min(len(samples), 1024), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(_load_one, num_parallel_calls=tf.data.AUTOTUNE)

    pad_value = normalized_pad_value(preprocess_config)

    channels = int(preprocess_config.get("channels", 3))
    height = int(preprocess_config.get("target_height", 64))
    ds = ds.padded_batch(
        batch_size,
        padded_shapes=(
            [height, None, channels],
            [None],
            [],
            [],
        ),
        padding_values=(
            tf.constant(pad_value, dtype=tf.float32),
            tf.constant(0, dtype=tf.int32),
            tf.constant(0, dtype=tf.int32),
            tf.constant(0, dtype=tf.int32),
        ),
        drop_remainder=False,
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


def iter_samples(samples: list[tuple[Path, str]]) -> Iterator[tuple[Path, str]]:
    """Simple iterator over sample pairs (for debugging)."""
    yield from samples
