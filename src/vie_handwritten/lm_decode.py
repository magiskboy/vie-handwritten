"""TF-free KenLM + pyctcdecode CTC beam search (shared by Keras postprocess and OpenVINO)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def load_unigrams(path: str | Path | None) -> list[str] | None:
    """Load a unigram list (one token per line); ``None`` if unavailable."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        logger.warning("Unigrams file not found: %s (decoding without unigrams)", p)
        return None
    toks = [line.strip() for line in p.read_text(encoding="utf-8").splitlines()]
    return [t for t in toks if t and not t.startswith("##")]


def build_lm_decoder(charset: Any, ctc_cfg: dict[str, Any]):
    """Build a pyctcdecode beam-search decoder with a KenLM shallow-fusion LM.

    ``labels`` mirror the model's logit order: index 0 (``<BLANK>``) maps to the
    empty string pyctcdecode reserves for the CTC blank; index 1 (space) stays a
    literal space so it acts as the Vietnamese syllable/word boundary.

    Paths in ``ctc_cfg`` must already be absolute (or resolvable) filesystem paths.
    """
    from pyctcdecode import build_ctcdecoder

    from vie_handwritten.utils import abs_path

    lm_raw = ctc_cfg.get("lm_path")
    lm_path = Path(lm_raw) if lm_raw else None
    if lm_path is not None and not lm_path.is_file():
        lm_path = abs_path(lm_raw)
    if lm_path is None or not lm_path.is_file():
        raise FileNotFoundError(
            f"LM file not found: {lm_raw!r}. Build it with `make build-lm` "
            "or ensure the artifact includes lm/vi.binary."
        )
    uni_raw = ctc_cfg.get("unigrams_path") or ctc_cfg.get("lexicon_path")
    uni_path = None
    if uni_raw:
        uni_path = Path(uni_raw)
        if not uni_path.is_file():
            uni_path = abs_path(uni_raw)
    labels = [""] + list(charset.characters[1:])
    unigrams = load_unigrams(uni_path)
    logger.info(
        "Building LM decoder: lm=%s labels=%d unigrams=%s",
        lm_path,
        len(labels),
        len(unigrams) if unigrams else 0,
    )
    return build_ctcdecoder(
        labels,
        kenlm_model_path=str(lm_path),
        unigrams=unigrams,
        alpha=float(ctc_cfg.get("alpha", 0.5)),
        beta=float(ctc_cfg.get("beta", 1.0)),
    )


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically-stable log-softmax over the last axis (per time step)."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def ctc_lm_decode(
    logits: np.ndarray,
    decoder: Any,
    *,
    beam_width: int = 100,
    token_min_logp: float = -5.0,
    beam_prune_logp: float = -10.0,
) -> list[str]:
    """LM-fused CTC beam search over a batch of logits ``(B, T, C)``."""
    if logits.ndim != 3:
        raise ValueError(f"Expected logits (B, T, C), got {logits.shape}")
    log_probs = log_softmax(np.asarray(logits, dtype=np.float32))
    return [
        decoder.decode(
            lp,
            beam_width=beam_width,
            token_min_logp=token_min_logp,
            beam_prune_logp=beam_prune_logp,
        )
        for lp in log_probs
    ]
