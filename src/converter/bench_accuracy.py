"""Accuracy benchmark: CER/WER for OV variants vs the Keras baseline.

OV evaluation is TF-free; the Keras baseline is only loaded when a checkpoint is
provided and TensorFlow is importable. Decoding defaults to greedy so the measured
delta reflects numeric precision (FP32 vs FP16 vs INT8), not the decode method.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from converter.data import load_split
from converter.runtime import OpenVINOCR

logger = logging.getLogger(__name__)


def _cer(reference: str, hypothesis: str) -> float:
    import editdistance

    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return editdistance.eval(reference, hypothesis) / len(reference)


def _wer(reference: str, hypothesis: str) -> float:
    import editdistance

    ref, hyp = reference.split(), hypothesis.split()
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return editdistance.eval(ref, hyp) / len(ref)


def _corpus(refs: list[str], hyps: list[str]) -> dict[str, float]:
    if not refs:
        return {"cer": 0.0, "wer": 0.0, "n": 0}
    return {
        "cer": float(np.mean([_cer(r, h) for r, h in zip(refs, hyps)])),
        "wer": float(np.mean([_wer(r, h) for r, h in zip(refs, hyps)])),
        "n": len(refs),
    }


def _config_from_ov(ov_dir: str | Path) -> dict[str, Any]:
    from vie_handwritten.utils import load_config

    local = Path(ov_dir) / "config.yaml"
    if local.is_file():
        return load_config(local)
    raise FileNotFoundError(f"No config.yaml in {ov_dir}; run `vie-ov convert` first.")


def _keras_hyps(checkpoint: str | Path, arrays: list[np.ndarray]) -> list[str]:
    from vie_handwritten.model import OCRModel

    model = OCRModel.from_checkpoint(checkpoint, decode="greedy")
    return [model.recognize(a) for a in arrays]


def bench_accuracy(
    ov_dir: str | Path,
    *,
    checkpoint: str | Path | None = None,
    split: str = "val",
    max_samples: int = 500,
    precisions: tuple[str, ...] = ("fp16", "int8"),
    batch: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    """Run accuracy benchmark and return per-model CER/WER + deltas."""
    config = _config_from_ov(ov_dir)
    arrays, refs = load_split(config, split=split, max_samples=max_samples, seed=seed)
    logger.info("Accuracy bench on split=%s n=%d", split, len(arrays))

    results: dict[str, dict[str, float]] = {}
    for precision in precisions:
        model = OpenVINOCR.from_dir(ov_dir, batch=batch, precision=precision)
        hyps = model.recognize_batch(arrays)
        results[f"ov_{precision}"] = _corpus(refs, hyps)

    if checkpoint is not None:
        try:
            hyps = _keras_hyps(checkpoint, arrays)
            results["keras"] = _corpus(refs, hyps)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Keras baseline skipped: %s", exc)

    baseline = results.get("keras") or results.get("ov_fp16")
    deltas = {}
    if baseline:
        for name, m in results.items():
            deltas[name] = {
                "cer_delta": round(m["cer"] - baseline["cer"], 4),
                "wer_delta": round(m["wer"] - baseline["wer"], 4),
            }

    return {
        "split": split,
        "n": len(arrays),
        "batch": batch,
        "results": results,
        "deltas": deltas,
    }


def format_report(report: dict[str, Any]) -> str:
    """Human-readable table: model | CER | WER | ΔCER | ΔWER."""
    lines = [
        f"Accuracy (split={report['split']}, n={report['n']}, batch={report['batch']})",
        f"{'model':<12}{'CER':>10}{'WER':>10}{'ΔCER':>10}{'ΔWER':>10}",
    ]
    for name, m in report["results"].items():
        d = report["deltas"].get(name, {})
        lines.append(
            f"{name:<12}{m['cer']:>10.4f}{m['wer']:>10.4f}"
            f"{d.get('cer_delta', 0.0):>10.4f}{d.get('wer_delta', 0.0):>10.4f}"
        )
    return "\n".join(lines)
