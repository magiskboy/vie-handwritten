"""Post-processing of decoded OCR strings."""

from __future__ import annotations

import re


def collapse_whitespace(text: str) -> str:
    """Normalize runs of whitespace to a single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


def fix_common_ocr_errors(text: str) -> str:
    """Apply lightweight Vietnamese OCR heuristics."""
    # Placeholder for lexicon / confusion pairs; keep identity for now.
    return text


def postprocess(text: str) -> str:
    """Full post-process chain on a decoded transcription."""
    return fix_common_ocr_errors(collapse_whitespace(text))
