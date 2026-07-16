"""Dataset discovery, manifest building, and ``tf.data`` pipelines.

On-disk layout (HWDB, official split by writer)::

    <root_dir>/                       # e.g. data/images
      HWDB_line/
        train_data/<writer_id>/
          1.jpg, 2.jpg, ...
          label.json                  # {"1.jpg": "văn bản dòng", ...}
        test_data/<writer_id>/...
      HWDB_word/
        train_data/<writer_id>/...    # ảnh từ đơn
        test_data/<writer_id>/...

``line`` và ``word`` là cùng corpus ở 2 mức chi tiết (từ và dòng) → đều là
"một dòng chữ" nên feed thẳng vào CRNN+CTC. ``paragraph`` (ảnh nguyên trang
nhiều dòng) hiện KHÔNG dùng để train CRNN.

Chuẩn hoá:
- Mỗi writer folder đọc ``label.json`` (fallback ``labels.json`` rồi thống nhất
  về ``label.json`` khi build).
- Split theo *writer* (writer-independent): ``test`` = ``test_data`` chính thức,
  ``val`` = một phần writers tách ra từ ``train_data``.
- Sinh manifest JSONL chuẩn hoá (``data/manifests/{train,val,test}.jsonl``) để
  mọi bước train/eval chỉ đọc manifest, tách khỏi layout đĩa.
"""

from __future__ import annotations

import json
import logging
import random
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from vie_handwritten.preprocess import load_image, preprocess
from vie_handwritten.utils import project_root

logger = logging.getLogger(__name__)

SOURCE_DIRS: dict[str, str] = {
    "line": "HWDB_line",
    "word": "HWDB_word",
    "paragraph": "HWDB_paragraph",
}

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def _abs_path(path: str | Path) -> Path:
    """Resolve a possibly-relative path against the project root."""
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
    """Resolve an image filename, tolerating case/extension differences."""
    exact = writer_dir / name
    if exact.is_file():
        return exact
    stem = Path(name).stem
    for cand in writer_dir.glob(f"{stem}.*"):
        if cand.is_file() and cand.suffix.lower() in _IMAGE_EXTS:
            return cand
    return None


def discover_source(
    root_dir: str | Path,
    source: str,
    subdir: str,
) -> list[dict[str, str]]:
    """Discover records for one source (``line``/``word``) and split subdir.

    Returns dicts ``{"image": <rel to root_dir>, "text", "source", "writer"}``.
    """
    if source not in SOURCE_DIRS:
        raise ValueError(f"Unknown source={source!r} (expected {list(SOURCE_DIRS)})")
    root_dir = _abs_path(root_dir)
    src_root = root_dir / SOURCE_DIRS[source] / subdir
    records: list[dict[str, str]] = []
    if not src_root.is_dir():
        logger.warning("Source dir missing: %s", src_root)
        return records
    missing = 0
    for writer_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        labels = _read_writer_labels(writer_dir)
        for name, text in labels.items():
            img = _resolve_image(writer_dir, name)
            if img is None:
                missing += 1
                continue
            records.append(
                {
                    "image": str(img.relative_to(root_dir)),
                    "text": text,
                    "source": source,
                    "writer": writer_dir.name,
                }
            )
    if missing:
        logger.warning("%s/%s: %d labeled images not found on disk", source, subdir, missing)
    return records


# --------------------------------------------------------------------------- #
# Manifest building (writer-independent split + OOV filter)
# --------------------------------------------------------------------------- #
def _normalize_text(text: str) -> str:
    """Unicode NFC normalize (keep casing, diacritics, spaces)."""
    return unicodedata.normalize("NFC", text)


def _oov_chars(text: str, vocab: set[str]) -> set[str]:
    return {ch for ch in text if ch not in vocab}


def build_manifests(config: dict[str, Any]) -> dict[str, dict[str, Path]]:
    """Scan datasets, split by writer, filter OOV, write **per-source** manifests.

    Each source (``word``, ``line``, …) gets its own frozen train/val/test split
    under ``<manifest_dir>/<source>/{train,val,test}.jsonl``. Multi-phase training
    then pins each phase to one source's manifests (word phase → line phase).

    The writer-independent validation split holds out the *same* writer ids across
    all sources (writer-id is consistent between HWDB_word / HWDB_line), so no
    writer leaks between train and val in any phase.

    Returns ``{source: {"train": path, "val": path, "test": path}}``.
    """
    from vie_handwritten.charset import Charset

    data_cfg = config["data"]
    root_dir = _abs_path(data_cfg["root_dir"])
    manifest_dir = _abs_path(data_cfg.get("manifest_dir", "data/manifests"))
    manifest_dir.mkdir(parents=True, exist_ok=True)
    sources = list(data_cfg.get("sources", ["word", "line"]))
    train_subdir = data_cfg.get("train_subdir", "train_data")
    test_subdir = data_cfg.get("test_subdir", "test_data")
    val_ratio = float(data_cfg.get("val_writers_ratio", 0.1))
    drop_oov = bool(data_cfg.get("drop_oov", True))
    seed = int(config.get("project", {}).get("seed", 42))

    charset = Charset.from_file(_abs_path(data_cfg["charset_path"]))
    vocab = set(charset.characters[1:])  # exclude blank at index 0

    train_records = {s: discover_source(root_dir, s, train_subdir) for s in sources}
    test_records = {s: discover_source(root_dir, s, test_subdir) for s in sources}

    # Writer-independent val split: hold out the SAME writer ids across sources.
    all_writers = sorted(
        {r["writer"] for recs in train_records.values() for r in recs}
    )
    rng = random.Random(seed)
    rng.shuffle(all_writers)
    n_val = max(1, round(len(all_writers) * val_ratio)) if all_writers else 0
    val_writers = set(all_writers[:n_val])
    logger.info(
        "Writers: total=%d val=%d train=%d",
        len(all_writers),
        len(val_writers),
        len(all_writers) - len(val_writers),
    )

    result: dict[str, dict[str, Path]] = {}
    for source in sources:
        dropped = 0
        buckets: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}

        def _accept(rec: dict[str, str], split: str) -> None:
            nonlocal dropped
            text = _normalize_text(rec["text"])
            if drop_oov and _oov_chars(text, vocab):
                dropped += 1
                return
            buckets[split].append({**rec, "text": text, "split": split})

        for r in train_records[source]:
            _accept(r, "val" if r["writer"] in val_writers else "train")
        for r in test_records[source]:
            _accept(r, "test")

        out_dir = manifest_dir / source
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        for split, recs in buckets.items():
            out = out_dir / f"{split}.jsonl"
            with out.open("w", encoding="utf-8") as fh:
                for rec in recs:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            paths[split] = out

        summary = {
            "root_dir": str(root_dir),
            "source": source,
            "seed": seed,
            "val_writers_ratio": val_ratio,
            "drop_oov": drop_oov,
            "dropped_oov": dropped,
            "counts": {k: len(v) for k, v in buckets.items()},
            "val_writers": sorted(val_writers),
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            "[%s] manifests → %s | counts=%s dropped_oov=%d",
            source,
            out_dir,
            {k: len(v) for k, v in buckets.items()},
            dropped,
        )
        result[source] = paths

    return result


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    """Load a JSONL manifest into a list of record dicts."""
    path = Path(path)
    records: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def source_manifest_paths(config: dict[str, Any], source: str) -> dict[str, Path]:
    """Return expected manifest paths for one source (does not build them)."""
    manifest_dir = _abs_path(config["data"].get("manifest_dir", "data/manifests"))
    base = manifest_dir / source
    return {s: base / f"{s}.jsonl" for s in ("train", "val", "test")}


def ensure_source_manifests(
    config: dict[str, Any], source: str, *, rebuild: bool = False
) -> dict[str, Path]:
    """Build all manifests if the requested source is missing (or ``rebuild``).

    Manifests are built for every source in ``data.sources`` in one pass so the
    writer-independent split stays consistent across sources.
    """
    paths = source_manifest_paths(config, source)
    if rebuild or not all(p.is_file() for p in paths.values()):
        built = build_manifests(config)
        if source not in built:
            raise ValueError(
                f"Source {source!r} not in data.sources={config['data'].get('sources')}"
            )
        return built[source]
    return paths


def group_by_source(records: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Group manifest records by their ``source`` field."""
    grouped: dict[str, list[dict[str, str]]] = {}
    for r in records:
        grouped.setdefault(r["source"], []).append(r)
    return grouped


# --------------------------------------------------------------------------- #
# tf.data pipelines
# --------------------------------------------------------------------------- #
def _make_load_fn(charset: Any, preprocess_config: dict[str, Any], root_dir: Path):
    import tensorflow as tf

    channels = int(preprocess_config.get("channels", 3))
    height = int(preprocess_config.get("target_height", 64))

    def _load_one(path: tf.Tensor, text: tf.Tensor):
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
            _py, [path, text], [tf.float32, tf.int32, tf.int32, tf.int32]
        )
        image.set_shape([height, None, channels])
        label.set_shape([None])
        label_len.set_shape([])
        width.set_shape([])
        return image, label, label_len, width

    return _load_one


def _source_dataset(
    records: list[dict[str, str]],
    *,
    charset: Any,
    config: dict[str, Any],
    root_dir: Path,
    training: bool,
    repeat: bool,
    seed: int,
) -> "Any":
    import tensorflow as tf

    preprocess_config = config["preprocess"]
    paths = [str(root_dir / r["image"]) for r in records]
    texts = [r["text"] for r in records]
    ds = tf.data.Dataset.from_tensor_slices((paths, texts))
    if training:
        ds = ds.shuffle(
            buffer_size=min(len(paths), 4096),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    if repeat:
        ds = ds.repeat()
    ds = ds.map(
        _make_load_fn(charset, preprocess_config, root_dir),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds


def _batch_and_format(ds: "Any", config: dict[str, Any]) -> "Any":
    import tensorflow as tf

    preprocess_config = config["preprocess"]
    batch_size = int(config["train"]["batch_size"])
    channels = int(preprocess_config.get("channels", 3))
    height = int(preprocess_config.get("target_height", 64))

    pad_value = 0.0
    if preprocess_config.get("normalize") == "imagenet":
        white = (1.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        pad_value = float(white.mean())

    ds = ds.padded_batch(
        batch_size,
        padded_shapes=([height, None, channels], [None], [], []),
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


def build_training_dataset(
    records_by_source: dict[str, list[dict[str, str]]],
    *,
    weights: dict[str, float],
    charset: Any,
    config: dict[str, Any],
    seed: int = 42,
) -> "Any":
    """Build an (infinite) training dataset mixing sources by ``weights``.

    Each source is shuffled + repeated, then interleaved with
    ``tf.data.Dataset.sample_from_datasets``. Use ``steps_per_epoch`` in
    ``model.fit`` because the dataset repeats forever.
    """
    import tensorflow as tf

    root_dir = _abs_path(config["data"]["root_dir"])
    datasets: list[Any] = []
    src_weights: list[float] = []
    used: list[str] = []
    for source, records in records_by_source.items():
        if not records:
            continue
        w = float(weights.get(source, 0.0))
        if w <= 0:
            continue
        datasets.append(
            _source_dataset(
                records,
                charset=charset,
                config=config,
                root_dir=root_dir,
                training=True,
                repeat=True,
                seed=seed,
            )
        )
        src_weights.append(w)
        used.append(source)
    if not datasets:
        raise ValueError(f"No training records for weights={weights}")
    total = sum(src_weights)
    src_weights = [w / total for w in src_weights]
    logger.info("Mixing sources %s with weights %s", used, [round(w, 3) for w in src_weights])
    if len(datasets) == 1:
        ds = datasets[0]
    else:
        ds = tf.data.Dataset.sample_from_datasets(
            datasets, weights=src_weights, seed=seed, stop_on_empty_dataset=False
        )
    return _batch_and_format(ds, config)


def build_eval_dataset(
    records: list[dict[str, str]],
    *,
    charset: Any,
    config: dict[str, Any],
    max_samples: int | None = None,
    seed: int = 42,
) -> "Any":
    """Build a finite dataset for validation/eval (no mixing, no repeat)."""
    root_dir = _abs_path(config["data"]["root_dir"])
    if max_samples is not None and len(records) > max_samples:
        rng = random.Random(seed)
        records = rng.sample(records, max_samples)
    ds = _source_dataset(
        records,
        charset=charset,
        config=config,
        root_dir=root_dir,
        training=False,
        repeat=False,
        seed=seed,
    )
    return _batch_and_format(ds, config)


def resolve_image_path(config: dict[str, Any], record: dict[str, str]) -> Path:
    """Absolute image path for a manifest record."""
    return _abs_path(config["data"]["root_dir"]) / record["image"]
