"""Performance benchmark: CPU latency/throughput for OV variants vs Keras.

Measures synchronous single-batch inference latency (p50/p95) and images/s for
each OpenVINO variant. The optional Keras baseline is pinned to CPU so the
comparison is apples-to-apples (no GPU).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from converter.config import ArtifactPaths
from converter.runtime import OpenVINOCR, imagenet_pad_value, pad_width

logger = logging.getLogger(__name__)


def _make_batch(model: OpenVINOCR, split: str, seed: int) -> np.ndarray:
    """Build one representative ``(B, H, W, C)`` batch (real images if available)."""
    shape = (model.batch, model.height, model.width, 3)
    try:
        from converter.data import load_split

        arrays, _ = load_split(model.config, split=split, max_samples=model.batch, seed=seed)
        if arrays:
            pv = imagenet_pad_value(model.config.get("preprocess", {}))
            padded = [pad_width(a, model.width, pv) for a in arrays]
            reps = (model.batch + len(padded) - 1) // len(padded)
            filled = (padded * reps)[: model.batch]
            return np.stack(filled).astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Falling back to random input for perf bench: %s", exc)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def _time_ov(model: OpenVINOCR, batch_arr: np.ndarray, warmup: int, iters: int) -> dict[str, float]:
    compiled = model.compiled_model
    output = compiled.output(0)
    for _ in range(warmup):
        compiled(batch_arr)[output]
    lat = []
    for _ in range(iters):
        t0 = time.perf_counter()
        compiled(batch_arr)[output]
        lat.append((time.perf_counter() - t0) * 1000.0)
    return _summarize(lat, model.batch)


def _time_keras(net: Any, batch_arr: np.ndarray, warmup: int, iters: int, batch: int) -> dict[str, float]:
    import tensorflow as tf

    with tf.device("/CPU:0"):
        tensor = tf.convert_to_tensor(batch_arr)
        for _ in range(warmup):
            net(tensor, training=False)
        lat = []
        for _ in range(iters):
            t0 = time.perf_counter()
            net(tensor, training=False)
            lat.append((time.perf_counter() - t0) * 1000.0)
    return _summarize(lat, batch)


def _summarize(latencies_ms: list[float], batch: int) -> dict[str, float]:
    arr = np.asarray(latencies_ms)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    return {
        "latency_ms_p50": round(p50, 3),
        "latency_ms_p95": round(p95, 3),
        "images_per_s": round(batch * 1000.0 / p50, 2) if p50 > 0 else 0.0,
    }


def bench_perf(
    ov_dir: str | Path,
    *,
    checkpoint: str | Path | None = None,
    precisions: tuple[str, ...] = ("fp16", "int8"),
    batches: tuple[int, ...] = (1, 16),
    warmup: int = 20,
    iters: int = 100,
    split: str = "val",
    seed: int = 42,
) -> dict[str, Any]:
    """Benchmark latency/throughput for each precision x batch on CPU."""
    results: dict[str, dict[str, float]] = {}
    keras_net = None
    for batch in batches:
        for precision in precisions:
            xml = ArtifactPaths.for_dir(ov_dir).model_xml(precision, batch)
            if not xml.is_file():
                logger.warning("Skip missing IR: %s", xml)
                continue
            model = OpenVINOCR.from_dir(ov_dir, batch=batch, precision=precision)
            batch_arr = _make_batch(model, split, seed)
            results[f"ov_{precision}_b{batch}"] = _time_ov(model, batch_arr, warmup, iters)

        if checkpoint is not None:
            try:
                if keras_net is None:
                    from converter.export import build_keras_net

                    keras_net, _, _ = build_keras_net(checkpoint)
                probe = OpenVINOCR.from_dir(
                    ov_dir, batch=batch, precision=precisions[0]
                )
                batch_arr = _make_batch(probe, split, seed)
                results[f"keras_b{batch}"] = _time_keras(
                    keras_net, batch_arr, warmup, iters, batch
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Keras perf baseline skipped (batch=%d): %s", batch, exc)

    return {"device": "CPU", "warmup": warmup, "iters": iters, "results": results}


def format_report(report: dict[str, Any]) -> str:
    """Human-readable table: model | p50 ms | p95 ms | images/s."""
    lines = [
        f"Performance (device={report['device']}, warmup={report['warmup']}, iters={report['iters']})",
        f"{'model':<18}{'p50 ms':>12}{'p95 ms':>12}{'images/s':>12}",
    ]
    for name, m in report["results"].items():
        lines.append(
            f"{name:<18}{m['latency_ms_p50']:>12.3f}{m['latency_ms_p95']:>12.3f}{m['images_per_s']:>12.2f}"
        )
    return "\n".join(lines)
