"""CTC loss and decoding utilities (TensorFlow)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def ctc_loss(
    y_true,
    y_pred,
    *,
    blank_index: int = 0,
    label_length=None,
    logit_length=None,
):
    """Compute mean CTC loss between label sequences and model logits."""
    y_true = tf.cast(y_true, tf.int32)
    if label_length is None:
        label_length = tf.reduce_sum(tf.ones_like(y_true), axis=1)
    if logit_length is None:
        logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])

    per_example = tf.nn.ctc_loss(
        labels=y_true,
        logits=y_pred,
        label_length=tf.cast(label_length, tf.int32),
        logit_length=tf.cast(logit_length, tf.int32),
        logits_time_major=False,
        blank_index=blank_index,
    )
    # zero_infinity: when a sample's time steps can't fit its label (T < label_length,
    # or repeats needing extra blanks) CTC returns +inf; zero it so one bad sample in a
    # batch can't blow up the mean loss / gradients (mirrors torch ``zero_infinity=True``).
    per_example = tf.where(tf.math.is_finite(per_example), per_example, tf.zeros_like(per_example))
    return tf.reduce_mean(per_example)


def ctc_greedy_decode(logits: np.ndarray, *, blank_index: int = 0) -> list[list[int]]:
    """Greedy CTC decode: argmax per timestep → remove blanks & collapse repeats."""
    if logits.ndim != 3:
        raise ValueError(f"Expected logits (B, T, C), got {logits.shape}")
    batch = []
    for seq in logits:
        idxs = np.argmax(seq, axis=-1).tolist()
        collapsed: list[int] = []
        prev = None
        for i in idxs:
            if i == blank_index:
                prev = None
                continue
            if i == prev:
                continue
            collapsed.append(int(i))
            prev = i
        batch.append(collapsed)
    return batch


def ctc_beam_decode(
    logits: np.ndarray,
    *,
    blank_index: int = 0,
    beam_width: int = 10,
) -> list[list[int]]:
    """Beam-search CTC decode via TensorFlow."""
    logits_t = tf.convert_to_tensor(logits, dtype=tf.float32)
    log_probs = tf.nn.log_softmax(logits_t, axis=-1)
    time_major = tf.transpose(log_probs, [1, 0, 2])
    seq_len = tf.fill([tf.shape(logits_t)[0]], tf.shape(logits_t)[1])
    decoded, _ = tf.nn.ctc_beam_search_decoder(
        time_major, seq_len, beam_width=beam_width, top_paths=1
    )
    dense = tf.sparse.to_dense(decoded[0], default_value=blank_index).numpy()
    return [[int(i) for i in row if i != blank_index] for row in dense]


def _load_unigrams(path: str | Path | None) -> list[str] | None:
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
    """
    from pyctcdecode import build_ctcdecoder

    lm_path = ctc_cfg.get("lm_path")
    if not lm_path or not Path(lm_path).is_file():
        raise FileNotFoundError(
            f"LM file not found: {lm_path!r}. Build it with `make build-lm`."
        )
    labels = [""] + list(charset.characters[1:])
    unigrams = _load_unigrams(ctc_cfg.get("unigrams_path") or ctc_cfg.get("lexicon_path"))
    logger.info("Building LM decoder: lm=%s labels=%d unigrams=%s",
                lm_path, len(labels), len(unigrams) if unigrams else 0)
    return build_ctcdecoder(
        labels,
        kenlm_model_path=str(lm_path),
        unigrams=unigrams,
        alpha=float(ctc_cfg.get("alpha", 0.5)),
        beta=float(ctc_cfg.get("beta", 1.0)),
    )


def _log_softmax(logits: np.ndarray) -> np.ndarray:
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
    log_probs = _log_softmax(np.asarray(logits, dtype=np.float32))
    return [
        decoder.decode(
            lp,
            beam_width=beam_width,
            token_min_logp=token_min_logp,
            beam_prune_logp=beam_prune_logp,
        )
        for lp in log_probs
    ]


def decode_predictions(
    logits: np.ndarray,
    charset: Any,
    *,
    method: str = "greedy",
    blank_index: int = 0,
    beam_width: int = 10,
    lm_decoder: Any = None,
    token_min_logp: float = -5.0,
    beam_prune_logp: float = -10.0,
) -> list[str]:
    """Decode a batch of logits into Vietnamese text strings."""
    if method == "beam_lm":
        if lm_decoder is None:
            raise ValueError("method='beam_lm' requires a prebuilt lm_decoder")
        return ctc_lm_decode(
            logits,
            lm_decoder,
            beam_width=beam_width,
            token_min_logp=token_min_logp,
            beam_prune_logp=beam_prune_logp,
        )
    if method == "beam":
        paths = ctc_beam_decode(logits, blank_index=blank_index, beam_width=beam_width)
    else:
        paths = ctc_greedy_decode(logits, blank_index=blank_index)
    return [str(charset.decode(path, join=True)) for path in paths]
