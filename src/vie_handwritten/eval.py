"""Evaluation + inference: CER/WER metrics and end-to-end OCR orchestration.

Text decoding + Vietnamese normalization live in :mod:`vie_handwritten.postprocess`
(wrapped by :class:`vie_handwritten.model.OCRModel`); this module only measures
quality and drives the "load checkpoint dir -> decode a split / an image" flow.

A checkpoint is a directory containing ``model.weights.h5`` and ``config.yaml``.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import editdistance

from vie_handwritten.dataset import ensure_manifests, load_manifest, resolve_image_path
from vie_handwritten.model import OCRModel
from vie_handwritten.preprocess import load_image, preprocess

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Metrics (edit distance)
# --------------------------------------------------------------------------- #
def character_error_rate(reference: str, hypothesis: str) -> float:
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return editdistance.eval(reference, hypothesis) / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words, hyp_words = reference.split(), hypothesis.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return editdistance.eval(ref_words, hyp_words) / len(ref_words)


def evaluate_corpus(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Aggregate CER / WER over paired reference/hypothesis strings."""
    if not references:
        return {"cer": 0.0, "wer": 0.0, "n": 0}
    cer = sum(character_error_rate(r, h) for r, h in zip(references, hypotheses))
    wer = sum(word_error_rate(r, h) for r, h in zip(references, hypotheses))
    n = len(references)
    return {"cer": cer / n, "wer": wer / n, "n": n}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def evaluate_split(
    model: OCRModel,
    records: list[dict[str, str]],
) -> dict[str, float]:
    """Decode every record with an in-memory ``OCRModel`` -> CER/WER metrics."""
    config = model.config
    refs, hyps = [], []
    for rec in records:
        image = load_image(str(resolve_image_path(config, rec)))
        arr = preprocess(image, config["preprocess"])
        hyps.append(model.recognize(arr))
        refs.append(rec["text"])
    return evaluate_corpus(refs, hyps)


def evaluate(
    checkpoint: str | Path,
    *,
    split: str = "test",
    max_samples: int | None = None,
    decode: str | None = None,
) -> dict[str, float]:
    """Load a checkpoint directory and evaluate CER/WER on a manifest split."""
    model = OCRModel.from_checkpoint(checkpoint, decode=decode)
    config = model.config
    manifests = ensure_manifests(config)
    if split not in manifests:
        raise ValueError(f"Unknown split={split}")
    records = load_manifest(manifests[split])
    if max_samples is not None and len(records) > max_samples:
        seed = int(config.get("project", {}).get("seed", 42))
        records = random.Random(seed).sample(records, max_samples)

    metrics = evaluate_split(model, records)
    logger.info("split=%s n=%s CER=%.4f WER=%.4f", split, metrics["n"], metrics["cer"], metrics["wer"])
    return metrics


def infer(
    checkpoint: str | Path,
    image_path: str | Path,
    *,
    decode: str | None = None,
) -> str:
    """Run OCR on a single image file and return the decoded text."""
    model = OCRModel.from_checkpoint(checkpoint, decode=decode)
    arr = preprocess(load_image(str(image_path)), model.config["preprocess"])
    return model.recognize(arr)
