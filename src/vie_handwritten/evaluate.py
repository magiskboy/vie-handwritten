"""Evaluation helpers (CER / WER on a data split)."""

from __future__ import annotations

import logging
from pathlib import Path

from vie_handwritten.charset import Charset
from vie_handwritten.config import load_config
from vie_handwritten.ctc import decode_predictions
from vie_handwritten.dataset import (
    ensure_source_manifests,
    load_manifest,
    resolve_image_path,
)
from vie_handwritten.metrics import evaluate_corpus
from vie_handwritten.model import build_crnn, load_crnn_weights
from vie_handwritten.postprocess import postprocess
from vie_handwritten.preprocess import load_image, preprocess
from vie_handwritten.utils import project_root

logger = logging.getLogger(__name__)


def evaluate(
    config_path: str | Path,
    checkpoint: str | Path,
    *,
    split: str = "test",
    source: str | None = None,
    max_samples: int | None = None,
) -> dict[str, float]:
    """Evaluate a checkpoint on a manifest split and return CER/WER metrics.

    ``source`` optionally restricts to one source (e.g. ``line``); defaults to
    ``data.eval_source`` in the config, else all sources in the split.
    """
    config = load_config(config_path)
    charset_path = Path(config["data"]["charset_path"])
    if not charset_path.is_absolute():
        charset_path = project_root() / charset_path

    charset = Charset.from_file(charset_path)
    source = source or config["data"].get("eval_source")
    if not source:
        raise ValueError(
            "evaluate needs a source (pass --source or set data.eval_source): "
            "manifests are now per-source"
        )
    manifests = ensure_source_manifests(config, source)
    if split not in manifests:
        raise ValueError(f"Unknown split={split}")
    records = load_manifest(manifests[split])

    if max_samples is not None and len(records) > max_samples:
        import random

        seed = int(config.get("project", {}).get("seed", 42))
        records = random.Random(seed).sample(records, max_samples)

    model = build_crnn(config, num_classes=charset.num_classes)
    load_crnn_weights(model, checkpoint)

    refs: list[str] = []
    hyps: list[str] = []
    ctc_cfg = config.get("ctc", {})
    for record in records:
        text = record["text"]
        img = load_image(str(resolve_image_path(config, record)))
        arr = preprocess(img, config["preprocess"])
        logits = model.predict(arr[None, ...], verbose=0)
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
