"""TF-free CTC greedy decode + Vietnamese normalization.

The deploy path must not import TensorFlow, so this module reimplements the
NumPy-only greedy CTC decode and mirrors the Vietnamese cleanup done in
:mod:`vie_handwritten.postprocess` (Underthesea ``text_normalize`` + whitespace /
punctuation tidy) without pulling in the TF-bound module.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


def greedy_ctc(logits: Any, *, blank_index: int = 0) -> list[list[int]]:
    """Greedy CTC decode: argmax per timestep, drop blanks, collapse repeats."""
    import numpy as np

    arr = np.asarray(logits)
    if arr.ndim != 3:
        raise ValueError(f"Expected logits (B, T, C), got {arr.shape}")
    out: list[list[int]] = []
    for seq in arr:
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
        out.append(collapsed)
    return out


def _fix_punct_spacing(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\(\s+", "(", text)
    return text


def normalize_vi(
    text: str,
    *,
    nfc: bool = True,
    underthesea: bool = True,
    underthesea_tokenizer: str = "regex",
) -> str:
    """Vietnamese normalization mirroring the training-time postprocess defaults."""
    if underthesea:
        try:
            from underthesea import text_normalize

            text = text_normalize(text, tokenizer=underthesea_tokenizer)
        except Exception:  # noqa: BLE001 - fall back if underthesea unavailable
            if nfc:
                text = unicodedata.normalize("NFC", text)
    elif nfc:
        text = unicodedata.normalize("NFC", text)
    text = _fix_punct_spacing(text)
    return re.sub(r"\s+", " ", text).strip()


class GreedyDecoder:
    """Charset-aware greedy decoder: logits ``(B, T, C)`` -> normalized text."""

    def __init__(
        self,
        charset: Any,
        *,
        blank_index: int = 0,
        nfc: bool = True,
        underthesea: bool = True,
        underthesea_tokenizer: str = "regex",
    ):
        self.charset = charset
        self.blank_index = blank_index
        self.nfc = nfc
        self.underthesea = underthesea
        self.underthesea_tokenizer = underthesea_tokenizer

    @classmethod
    def from_config(cls, charset: Any, config: dict[str, Any]) -> "GreedyDecoder":
        pp = config.get("postprocess", {})
        return cls(
            charset,
            blank_index=int(config.get("ctc", {}).get("blank_index", 0)),
            nfc=bool(pp.get("nfc", True)),
            underthesea=bool(pp.get("underthesea", True)),
            underthesea_tokenizer=str(pp.get("underthesea_tokenizer", "regex")),
        )

    def decode(self, logits: Any) -> list[str]:
        paths = greedy_ctc(logits, blank_index=self.blank_index)
        raw = [str(self.charset.decode(p, join=True)) for p in paths]
        return [
            normalize_vi(
                text,
                nfc=self.nfc,
                underthesea=self.underthesea,
                underthesea_tokenizer=self.underthesea_tokenizer,
            )
            for text in raw
        ]
