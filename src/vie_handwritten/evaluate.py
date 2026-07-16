"""Evaluation helpers (CER / WER on a data split)."""

from __future__ import annotations

import logging
from pathlib import Path

from vie_handwritten.charset import Charset
from vie_handwritten.config import load_config
from vie_handwritten.dataset import discover_samples, train_val_test_split
from vie_handwritten.metrics import evaluate_corpus
from vie_handwritten.model import build_crnn, load_crnn_weights
from vie_handwritten.pipeline import OCRPipeline
from vie_handwritten.preprocess import load_image
from vie_handwritten.utils import project_root

logger = logging.getLogger(__name__)


def evaluate(
    config_path: str | Path,
    checkpoint: str | Path,
    *,
    split: str = "test",
) -> dict[str, float]:
    """Evaluate a checkpoint on train/val/test and return CER/WER metrics."""
    config = load_config(config_path)
    root = project_root()
    dataset_dir = Path(config["data"]["dataset_dir"])
    if not dataset_dir.is_absolute():
        dataset_dir = root / dataset_dir
    charset_path = Path(config["data"]["charset_path"])
    if not charset_path.is_absolute():
        charset_path = root / charset_path

    charset = Charset.from_file(charset_path)
    samples = discover_samples(
        dataset_dir,
        images_subdir=config["data"].get("images_subdir", "data"),
        labels_file=config["data"].get("labels_file", "labels.json"),
    )
    seed = int(config.get("project", {}).get("seed", 42))
    train_s, val_s, test_s = train_val_test_split(
        samples,
        train_ratio=float(config["data"]["train_split"]),
        val_ratio=float(config["data"]["val_split"]),
        test_ratio=float(config["data"]["test_split"]),
        seed=seed,
    )
    split_map = {"train": train_s, "val": val_s, "test": test_s}
    if split not in split_map:
        raise ValueError(f"Unknown split={split}")
    eval_samples = split_map[split]

    model = build_crnn(config, num_classes=charset.num_classes)
    load_crnn_weights(model, checkpoint)
    pipeline = OCRPipeline(model, charset, config)

    batch_size = int(
        config.get("eval", {}).get("batch_size", config["train"].get("batch_size", 16))
    )
    refs: list[str] = []
    hyps: list[str] = []
    for start in range(0, len(eval_samples), batch_size):
        chunk = eval_samples[start : start + batch_size]
        images = [load_image(str(path)) for path, _ in chunk]
        # predict_batch pads to a common width and already applies postprocess.
        preds = pipeline.predict_batch(images)
        for (_, text), pred in zip(chunk, preds):
            refs.append(text)
            hyps.append(pred)

    metrics = evaluate_corpus(refs, hyps)
    logger.info(
        "split=%s n=%s CER=%.4f WER=%.4f",
        split,
        metrics["n"],
        metrics["cer"],
        metrics["wer"],
    )
    return metrics
