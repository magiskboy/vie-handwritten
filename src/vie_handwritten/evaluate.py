"""Evaluation + inference: text metrics, post-processing, CER/WER, and OCR.

Consolidates what used to be ``metrics.py``, ``postprocess.py`` and
``pipeline.py`` so the whole "logits → text → score" path lives in one place.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import editdistance

from vie_handwritten.charset import Charset
from vie_handwritten.ctc import build_lm_decoder, decode_predictions
from vie_handwritten.dataset import ensure_manifests, load_manifest, resolve_image_path
from vie_handwritten.model import build_crnn, load_crnn_weights
from vie_handwritten.preprocess import load_image, preprocess
from vie_handwritten.text_norm import normalize_vietnamese
from vie_handwritten.utils import load_config, project_root

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Text post-processing
# --------------------------------------------------------------------------- #
def postprocess(text: str, pp_cfg: dict[str, Any] | None = None) -> str:
    """Vietnamese normalization (NFC, tone placement, whitespace/punctuation)."""
    pp_cfg = pp_cfg or {}
    return normalize_vietnamese(
        text,
        nfc=bool(pp_cfg.get("nfc", True)),
        tone_marks=bool(pp_cfg.get("normalize_tone_marks", True)),
    )


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
# Inference
# --------------------------------------------------------------------------- #
def _charset_path(config: dict[str, Any]) -> Path:
    p = Path(config["data"]["charset_path"])
    return p if p.is_absolute() else project_root() / p


def maybe_build_lm_decoder(charset: Charset, config: dict[str, Any]):
    """Build the LM beam-search decoder once, if ``ctc.decode == 'beam_lm'``."""
    ctc_cfg = config.get("ctc", {})
    if ctc_cfg.get("decode") != "beam_lm":
        return None
    return build_lm_decoder(charset, ctc_cfg)


def predict_image_array(
    crnn, image, charset: Charset, config: dict[str, Any], lm_decoder: Any = None
) -> str:
    """Run OCR on an in-memory image array → decoded, post-processed text."""
    arr = preprocess(image, config["preprocess"])
    logits = crnn.predict(arr[None, ...], verbose=0)
    ctc_cfg = config.get("ctc", {})
    pred = decode_predictions(
        logits,
        charset,
        method=ctc_cfg.get("decode", "greedy"),
        blank_index=int(ctc_cfg.get("blank_index", 0)),
        beam_width=int(ctc_cfg.get("beam_width", 10)),
        lm_decoder=lm_decoder,
        token_min_logp=float(ctc_cfg.get("token_min_logp", -5.0)),
        beam_prune_logp=float(ctc_cfg.get("beam_prune_logp", -10.0)),
    )[0]
    return postprocess(pred, config.get("postprocess", {}))


def evaluate_split(
    crnn,
    records: list[dict[str, str]],
    charset: Charset,
    config: dict[str, Any],
    lm_decoder: Any = None,
) -> dict[str, float]:
    """Decode every record with the in-memory CRNN → CER/WER metrics."""
    refs, hyps = [], []
    for rec in records:
        image = load_image(str(resolve_image_path(config, rec)))
        hyps.append(predict_image_array(crnn, image, charset, config, lm_decoder))
        refs.append(rec["text"])
    return evaluate_corpus(refs, hyps)


def evaluate(
    config_path: str | Path,
    checkpoint: str | Path,
    *,
    split: str = "test",
    max_samples: int | None = None,
    decode: str | None = None,
) -> dict[str, float]:
    """Load a checkpoint and evaluate CER/WER on a manifest split."""
    config = load_config(config_path)
    if decode:
        config.setdefault("ctc", {})["decode"] = decode
    charset = Charset.from_file(_charset_path(config))
    manifests = ensure_manifests(config)
    if split not in manifests:
        raise ValueError(f"Unknown split={split}")
    records = load_manifest(manifests[split])
    if max_samples is not None and len(records) > max_samples:
        seed = int(config.get("project", {}).get("seed", 42))
        records = random.Random(seed).sample(records, max_samples)

    crnn = build_crnn(config, num_classes=charset.num_classes)
    load_crnn_weights(crnn, checkpoint)
    lm_decoder = maybe_build_lm_decoder(charset, config)
    metrics = evaluate_split(crnn, records, charset, config, lm_decoder)
    logger.info("split=%s n=%s CER=%.4f WER=%.4f", split, metrics["n"], metrics["cer"], metrics["wer"])
    return metrics


def infer(
    config_path: str | Path,
    checkpoint: str | Path,
    image_path: str | Path,
    *,
    decode: str | None = None,
) -> str:
    """Run OCR on a single image file and return the decoded text."""
    config = load_config(config_path)
    if decode:
        config.setdefault("ctc", {})["decode"] = decode
    charset = Charset.from_file(_charset_path(config))
    crnn = build_crnn(config, num_classes=charset.num_classes)
    load_crnn_weights(crnn, checkpoint)
    lm_decoder = maybe_build_lm_decoder(charset, config)
    return predict_image_array(crnn, load_image(str(image_path)), charset, config, lm_decoder)
