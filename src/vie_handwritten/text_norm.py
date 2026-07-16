"""Vietnamese text normalization for OCR post-processing (Step 2).

Deterministic, rule-based clean-up applied to decoded text:
  * NFC Unicode (matches the label normalization used at training time)
  * tone-mark placement standardization for the oa / oe / uy clusters
    (e.g. ``hoà`` -> ``hòa``, ``thuý`` -> ``thúy``)
  * conservative whitespace / punctuation clean-up

The clean-up is intentionally conservative: it never inserts spaces after
``.``/``,`` (Vietnamese uses them as decimal / thousands separators, common in
the address & form domain), only trims spaces that are unambiguously spurious.
"""

from __future__ import annotations

import re
import unicodedata

# oa / oe / uy tone-placement: map "glide-then-toned-vowel" (o+á, u+ý, ...) to the
# dictionary-standard "toned-glide-then-vowel" (ó+a, ú+y, ...). Applied via a
# single regex alternation. The uy case is guarded against a preceding "q"
# (in "quý" the u is part of "qu", so the tone already sits correctly on y).
_TONE_MAP = {
    # o + toned a  ->  toned o + a
    "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
    "Oà": "Òa", "Oá": "Óa", "Oả": "Ỏa", "Oã": "Õa", "Oạ": "Ọa",
    # o + toned e  ->  toned o + e
    "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
    "Oè": "Òe", "Oé": "Óe", "Oẻ": "Ỏe", "Oẽ": "Õe", "Oẹ": "Ọe",
    # u + toned y  ->  toned u + y
    "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
    "Uỳ": "Ùy", "Uý": "Úy", "Uỷ": "Ủy", "Uỹ": "Ũy", "Uỵ": "Ụy",
}
_UY_KEYS = {"uỳ", "uý", "uỷ", "uỹ", "uỵ", "Uỳ", "Uý", "Uỷ", "Uỹ", "Uỵ"}
_TONE_RE = re.compile("|".join(re.escape(k) for k in _TONE_MAP))


def normalize_tone_placement(text: str) -> str:
    """Standardize tone-mark placement in oa / oe / uy clusters."""

    def _sub(m: re.Match[str]) -> str:
        key = m.group(0)
        if key in _UY_KEYS:
            start = m.start()
            prev = text[start - 1] if start > 0 else ""
            if prev in ("q", "Q"):  # "quý": u is a glide after q, leave as-is
                return key
        return _TONE_MAP[key]

    return _TONE_RE.sub(_sub, text)


def _fix_punct_spacing(text: str) -> str:
    """Trim unambiguously-spurious spaces around punctuation (conservative)."""
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)  # no space before these
    text = re.sub(r"\s+\)", ")", text)  # no space before closing paren
    text = re.sub(r"\(\s+", "(", text)  # no space after opening paren
    return text


def normalize_vietnamese(text: str, *, nfc: bool = True, tone_marks: bool = True) -> str:
    """Full Vietnamese normalization pipeline for a decoded line."""
    if nfc:
        text = unicodedata.normalize("NFC", text)
    if tone_marks:
        text = normalize_tone_placement(text)
    text = _fix_punct_spacing(text)
    return re.sub(r"\s+", " ", text).strip()
