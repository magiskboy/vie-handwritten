"""Postprocessing: CTC decoding (logits -> ids/text) + Vietnamese text cleanup.

Consolidates the whole "logits -> clean text" path in one place:

* CTC decoders: greedy, beam search (TensorFlow), and LM-fused beam search
  (pyctcdecode + KenLM shallow fusion).
* Deterministic Vietnamese normalization applied to the decoded string
  (Underthesea ``text_normalize`` by default: NFC, tone placement, composition
  fixes; optional local fallback; conservative whitespace/punctuation cleanup).

The :class:`CTCDecoder` bundles a decode method together with the text
normalization so callers get a single ``logits -> list[str]`` step.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

import numpy as np
import tensorflow as tf

from vie_handwritten.lm_decode import build_lm_decoder, ctc_lm_decode  # noqa: F401

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# CTC decoders
# --------------------------------------------------------------------------- #
def ctc_greedy_decode(logits: np.ndarray, *, blank_index: int = 0) -> list[list[int]]:
    """Greedy CTC decode: argmax per timestep -> remove blanks & collapse repeats."""
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
    """Decode a batch of logits into (raw, un-normalized) text strings."""
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


# --------------------------------------------------------------------------- #
# Vietnamese text normalization
# --------------------------------------------------------------------------- #
# Local fallback (used only when underthesea is disabled): map open-syllable
# "glide-then-toned-vowel" (o+á, u+ý, ...) to "toned-glide-then-vowel"
# (ó+a, ú+y, ...). ``(?!\w)`` keeps closed syllables like "soát" / "toàn"
# untouched. The uy case is guarded against a preceding "q" ("quý").
_TONE_MAP = {
    "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
    "Oà": "Òa", "Oá": "Óa", "Oả": "Ỏa", "Oã": "Õa", "Oạ": "Ọa",
    "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
    "Oè": "Òe", "Oé": "Óe", "Oẻ": "Ỏe", "Oẽ": "Õe", "Oẹ": "Ọe",
    "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
    "Uỳ": "Ùy", "Uý": "Úy", "Uỷ": "Ủy", "Uỹ": "Ũy", "Uỵ": "Ụy",
}
_UY_KEYS = {"uỳ", "uý", "uỷ", "uỹ", "uỵ", "Uỳ", "Uý", "Uỷ", "Uỹ", "Uỵ"}
_TONE_RE = re.compile(
    "(?:" + "|".join(re.escape(k) for k in _TONE_MAP) + r")(?!\w)"
)


def normalize_tone_placement(text: str) -> str:
    """Local fallback: standardize tone marks on open oa / oe / uy clusters."""

    def _sub(m: re.Match[str]) -> str:
        key = m.group(0)
        if key in _UY_KEYS:
            start = m.start()
            prev = text[start - 1] if start > 0 else ""
            if prev in ("q", "Q"):
                return key
        return _TONE_MAP[key]

    return _TONE_RE.sub(_sub, text)


def _underthesea_normalize(text: str, *, tokenizer: str = "regex") -> str:
    """Underthesea character/tone normalization (NFC, hoá→hóa, sóat→soát, …)."""
    from underthesea import text_normalize

    return text_normalize(text, tokenizer=tokenizer)


def _fix_punct_spacing(text: str) -> str:
    """Trim unambiguously-spurious spaces around punctuation (conservative)."""
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)  # no space before these
    text = re.sub(r"\s+\)", ")", text)  # no space before closing paren
    text = re.sub(r"\(\s+", "(", text)  # no space after opening paren
    return text


def normalize_vietnamese(
    text: str,
    *,
    nfc: bool = True,
    tone_marks: bool = True,
    underthesea: bool = True,
    underthesea_tokenizer: str = "regex",
) -> str:
    """Full Vietnamese normalization pipeline for a decoded line.

    Prefer Underthesea ``text_normalize`` (covers NFC, open/closed tone placement,
    Ð/Đ, vowel composition). Fall back to NFC + local open-syllable tone map when
    ``underthesea=False``.
    """
    if underthesea:
        text = _underthesea_normalize(text, tokenizer=underthesea_tokenizer)
    else:
        if nfc:
            text = unicodedata.normalize("NFC", text)
        if tone_marks:
            text = normalize_tone_placement(text)
    text = _fix_punct_spacing(text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str, pp_cfg: dict[str, Any] | None = None) -> str:
    """Vietnamese normalization driven by a ``postprocess`` config block."""
    pp_cfg = pp_cfg or {}
    return normalize_vietnamese(
        text,
        nfc=bool(pp_cfg.get("nfc", True)),
        tone_marks=bool(pp_cfg.get("normalize_tone_marks", True)),
        underthesea=bool(pp_cfg.get("underthesea", True)),
        underthesea_tokenizer=str(pp_cfg.get("underthesea_tokenizer", "regex")),
    )


# --------------------------------------------------------------------------- #
# Composition: decode + normalize
# --------------------------------------------------------------------------- #
class CTCDecoder:
    """Postprocess step: turn CRNN logits into clean Vietnamese text.

    Bundles a CTC decode method (greedy / beam / beam_lm) with the Vietnamese
    text normalization so ``decode(logits)`` yields ready-to-use strings.
    """

    def __init__(
        self,
        charset: Any,
        *,
        method: str = "greedy",
        blank_index: int = 0,
        beam_width: int = 10,
        lm_decoder: Any = None,
        token_min_logp: float = -5.0,
        beam_prune_logp: float = -10.0,
        nfc: bool = True,
        tone_marks: bool = True,
        underthesea: bool = True,
        underthesea_tokenizer: str = "regex",
    ):
        self.charset = charset
        self.method = method
        self.blank_index = blank_index
        self.beam_width = beam_width
        self.lm_decoder = lm_decoder
        self.token_min_logp = token_min_logp
        self.beam_prune_logp = beam_prune_logp
        self.nfc = nfc
        self.tone_marks = tone_marks
        self.underthesea = underthesea
        self.underthesea_tokenizer = underthesea_tokenizer

    @classmethod
    def from_config(cls, charset: Any, config: dict[str, Any]) -> "CTCDecoder":
        """Build a decoder from config; constructs the LM decoder if ``decode='beam_lm'``."""
        ctc_cfg = config.get("ctc", {})
        pp_cfg = config.get("postprocess", {})
        method = ctc_cfg.get("decode", "greedy")
        lm_decoder = build_lm_decoder(charset, ctc_cfg) if method == "beam_lm" else None
        return cls(
            charset,
            method=method,
            blank_index=int(ctc_cfg.get("blank_index", 0)),
            beam_width=int(ctc_cfg.get("beam_width", 10)),
            lm_decoder=lm_decoder,
            token_min_logp=float(ctc_cfg.get("token_min_logp", -5.0)),
            beam_prune_logp=float(ctc_cfg.get("beam_prune_logp", -10.0)),
            nfc=bool(pp_cfg.get("nfc", True)),
            tone_marks=bool(pp_cfg.get("normalize_tone_marks", True)),
            underthesea=bool(pp_cfg.get("underthesea", True)),
            underthesea_tokenizer=str(pp_cfg.get("underthesea_tokenizer", "regex")),
        )

    def decode(self, logits: np.ndarray) -> list[str]:
        """Decode a batch of logits ``(B, T, C)`` into normalized text strings."""
        raw = decode_predictions(
            logits,
            self.charset,
            method=self.method,
            blank_index=self.blank_index,
            beam_width=self.beam_width,
            lm_decoder=self.lm_decoder,
            token_min_logp=self.token_min_logp,
            beam_prune_logp=self.beam_prune_logp,
        )
        return [
            normalize_vietnamese(
                t,
                nfc=self.nfc,
                tone_marks=self.tone_marks,
                underthesea=self.underthesea,
                underthesea_tokenizer=self.underthesea_tokenizer,
            )
            for t in raw
        ]
