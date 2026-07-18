"""TF-free CTC decode + Vietnamese normalization for the OpenVINO deploy path.

Supports greedy and KenLM ``beam_lm`` (default when LM artifacts are present)
without importing TensorFlow. Mirrors ``vie_handwritten.postprocess`` cleanup.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

logger = logging.getLogger(__name__)


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


class ArtifactDecoder:
    """Charset-aware decoder: logits ``(B, T, C)`` -> normalized text.

    Defaults to ``beam_lm`` when an LM decoder can be built; falls back to greedy.
    """

    def __init__(
        self,
        charset: Any,
        *,
        method: str = "beam_lm",
        blank_index: int = 0,
        beam_width: int = 100,
        lm_decoder: Any = None,
        token_min_logp: float = -5.0,
        beam_prune_logp: float = -10.0,
        nfc: bool = True,
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
        self.underthesea = underthesea
        self.underthesea_tokenizer = underthesea_tokenizer

    @classmethod
    def from_config(
        cls,
        charset: Any,
        config: dict[str, Any],
        *,
        prefer_beam_lm: bool = True,
    ) -> "ArtifactDecoder":
        """Build decoder; OpenVINO path defaults to ``beam_lm`` when LM is available."""
        from vie_handwritten.lm_decode import build_lm_decoder

        ctc = config.get("ctc", {})
        pp = config.get("postprocess", {})
        method = "beam_lm" if prefer_beam_lm else str(ctc.get("decode", "greedy"))
        lm_decoder = None
        if method == "beam_lm":
            try:
                lm_decoder = build_lm_decoder(charset, ctc)
            except Exception as exc:  # noqa: BLE001
                logger.warning("beam_lm unavailable (%s); falling back to greedy", exc)
                method = "greedy"
        beam_width = int(ctc.get("beam_width", 100 if method == "beam_lm" else 10))
        if method == "beam_lm" and beam_width < 50:
            beam_width = 100
        return cls(
            charset,
            method=method,
            blank_index=int(ctc.get("blank_index", 0)),
            beam_width=beam_width,
            lm_decoder=lm_decoder,
            token_min_logp=float(ctc.get("token_min_logp", -5.0)),
            beam_prune_logp=float(ctc.get("beam_prune_logp", -10.0)),
            nfc=bool(pp.get("nfc", True)),
            underthesea=bool(pp.get("underthesea", True)),
            underthesea_tokenizer=str(pp.get("underthesea_tokenizer", "regex")),
        )

    def decode(self, logits: Any) -> list[str]:
        from vie_handwritten.lm_decode import ctc_lm_decode

        if self.method == "beam_lm" and self.lm_decoder is not None:
            raw = ctc_lm_decode(
                logits,
                self.lm_decoder,
                beam_width=self.beam_width,
                token_min_logp=self.token_min_logp,
                beam_prune_logp=self.beam_prune_logp,
            )
        else:
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


# Back-compat alias used by older imports.
GreedyDecoder = ArtifactDecoder
