"""Evaluation helpers (CER / WER on a data split)."""

from __future__ import annotations

import logging
from pathlib import Path

from vie_handwritten.charset import Charset
from vie_handwritten.config import load_config
from vie_handwritten.ctc import decode_predictions
from vie_handwritten.dataset import discover_samples, train_val_test_split
from vie_handwritten.metrics import evaluate_corpus
from vie_handwritten.model import build_crnn, load_crnn_weights, pack_crnn_inputs
from vie_handwritten.postprocess import postprocess
from vie_handwritten.preprocess import load_image, preprocess
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

    refs: list[str] = []
    hyps: list[str] = []
    ctc_cfg = config.get("ctc", {})
    for path, text in eval_samples:
        img = load_image(str(path))
        arr = preprocess(img, config["preprocess"])
        inputs = pack_crnn_inputs(arr[None, ...])
        logits = model.predict(inputs, verbose=0)
        pred = decode_predictions(
            logits,
            charset,
            method=ctc_cfg.get("decode", "greedy"),
            blank_index=int(ctc_cfg.get("blank_index", 0)),
            beam_width=int(ctc_cfg.get("beam_width", 10)),
        )[0]
        refs.append(text)
        hyps.append(postprocess(pred))

    metrics = evaluate_corpus(refs, hyps)
    logger.info(
        "split=%s n=%s CER=%.4f WER=%.4f",
        split,
        metrics["n"],
        metrics["cer"],
        metrics["wer"],
    )
    return metrics
