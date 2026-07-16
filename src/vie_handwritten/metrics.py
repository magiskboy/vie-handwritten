"""Evaluation metrics: CER / WER (edit distance)."""

from __future__ import annotations

import editdistance


def character_error_rate(reference: str, hypothesis: str) -> float:
    """CER = edit_distance(chars) / len(reference_chars)."""
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return editdistance.eval(reference, hypothesis) / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER = edit_distance(words) / len(reference_words)."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    return editdistance.eval(ref_words, hyp_words) / len(ref_words)


def evaluate_corpus(
    references: list[str],
    hypotheses: list[str],
) -> dict[str, float]:
    """Aggregate CER / WER over a list of paired strings."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses length mismatch")
    if not references:
        return {"cer": 0.0, "wer": 0.0, "n": 0}
    cer_sum = 0.0
    wer_sum = 0.0
    for ref, hyp in zip(references, hypotheses):
        cer_sum += character_error_rate(ref, hyp)
        wer_sum += word_error_rate(ref, hyp)
    n = len(references)
    return {"cer": cer_sum / n, "wer": wer_sum / n, "n": n}
