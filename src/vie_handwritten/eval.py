"""Evaluation + inference: CER/WER metrics and end-to-end OCR orchestration.

Text decoding + Vietnamese normalization live in :mod:`vie_handwritten.postprocess`
(wrapped by :class:`vie_handwritten.model.OCRModel`); this module only measures
quality and drives the "load checkpoint dir -> decode a split / an image" flow.

A checkpoint is a directory containing ``model.weights.h5`` and ``config.yaml``.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import editdistance

from vie_handwritten.dataset import load_split, resolve_image_path
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
) -> dict[str, Any]:
    """Decode every record with an in-memory ``OCRModel`` -> CER/WER + failures.

    Returns ``{"cer", "wer", "n", "failures"}`` where ``failures`` is a list of
    ``{"path", "gt", "predict"}`` for samples where prediction != ground truth.
    """
    config = model.config
    refs, hyps = [], []
    failures: list[dict[str, str]] = []
    for rec in records:
        path = resolve_image_path(config, rec)
        image = load_image(str(path))
        arr = preprocess(image, config["preprocess"])
        pred = model.recognize(arr)
        gt = rec["text"]
        hyps.append(pred)
        refs.append(gt)
        if pred != gt:
            failures.append({"path": str(path), "gt": gt, "predict": pred})
    metrics = evaluate_corpus(refs, hyps)
    metrics["failures"] = failures
    return metrics


def evaluate(
    checkpoint: str | Path,
    *,
    split: str = "test",
    max_samples: int | None = None,
    decode: str | None = None,
    failures_out: str | Path | None = None,
) -> dict[str, Any]:
    """Load a checkpoint directory and evaluate CER/WER on a data split.

    Failed samples (pred != GT) are written as JSON to ``failures_out`` when set,
    otherwise to ``{checkpoint}/failures_{split}.json``.
    """
    checkpoint = Path(checkpoint)
    model = OCRModel.from_checkpoint(checkpoint, decode=decode)
    config = model.config
    if split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split={split}")
    records = load_split(config, split)  # type: ignore[arg-type]
    if max_samples is not None and len(records) > max_samples:
        seed = int(config.get("project", {}).get("seed", 42))
        records = random.Random(seed).sample(records, max_samples)

    metrics = evaluate_split(model, records)
    failures: list[dict[str, str]] = metrics.pop("failures", [])
    out_path = Path(failures_out) if failures_out is not None else checkpoint / f"failures_{split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "split=%s n=%s CER=%.4f WER=%.4f failures=%d -> %s",
        split,
        metrics["n"],
        metrics["cer"],
        metrics["wer"],
        len(failures),
        out_path,
    )
    metrics["failures_out"] = str(out_path)
    metrics["n_failures"] = len(failures)
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
