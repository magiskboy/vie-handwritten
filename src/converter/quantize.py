"""NNCF INT8 post-training quantization + full convert orchestration.

``convert_checkpoint`` is the entry point behind ``vie-ov convert``: it builds the
Keras net once, exports FP16 + INT8 IRs for each requested batch size, writes a
self-contained OpenVINO artifact (charset, LM, config, build_info, meta), and
records a quick CER sanity check (FP16 vs INT8) in ``meta.yaml``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from vie_handwritten.utils import (
    CHARSET_NAME,
    WEIGHTS_NAME,
    checkpoint_weights_path,
    file_sha256,
    resolve_checkpoint_dir,
    save_sidecar_bundle,
)

from converter.config import ArtifactPaths, ShapeSpec, shape_from_checkpoint
from converter.data import load_split
from converter.export import build_keras_net, save_ir, to_ov_model
from converter.runtime import OpenVINOCR, imagenet_pad_value, pad_width

logger = logging.getLogger(__name__)


def calibration_dataset(
    arrays: list[np.ndarray],
    shape: ShapeSpec,
    pad_value: float,
) -> tuple[Any, int]:
    """Build an ``nncf.Dataset`` of full ``(B, H, W, C)`` batches from images."""
    import nncf

    batches: list[np.ndarray] = []
    padded = [pad_width(a, shape.width, pad_value) for a in arrays]
    # Keep only full batches so every calibration item matches the static shape.
    for start in range(0, len(padded) - shape.batch + 1, shape.batch):
        chunk = padded[start : start + shape.batch]
        batches.append(np.stack(chunk).astype(np.float32))
    if not batches:  # fewer images than one batch: tile to fill exactly one batch
        if padded:
            reps = (shape.batch + len(padded) - 1) // len(padded)
            filled = (padded * reps)[: shape.batch]
            batches.append(np.stack(filled).astype(np.float32))
    dataset = nncf.Dataset(batches, lambda item: item)
    return dataset, len(batches)


def quantize_model(
    ov_model: Any,
    dataset: Any,
    *,
    subset_size: int,
) -> Any:
    """Apply NNCF 8-bit PTQ targeting CPU."""
    import nncf

    logger.info("Running NNCF INT8 PTQ (subset_size=%d, target_device=CPU)", subset_size)
    return nncf.quantize(
        ov_model,
        dataset,
        subset_size=subset_size,
        target_device=nncf.TargetDevice.CPU,
    )


def _char_error_rate(reference: str, hypothesis: str) -> float:
    import editdistance

    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0
    return editdistance.eval(reference, hypothesis) / len(reference)


def _mean_cer(model: OpenVINOCR, arrays: list[np.ndarray], texts: list[str]) -> float:
    if not arrays:
        return 0.0
    hyps = model.recognize_batch(arrays)
    return float(np.mean([_char_error_rate(r, h) for r, h in zip(texts, hyps)]))


def convert_checkpoint(
    checkpoint: str | Path,
    ov_config: dict[str, Any],
    *,
    batches: list[int] | None = None,
) -> dict[str, Any]:
    """Convert a checkpoint to FP16 + INT8 IRs for each batch size; write meta."""
    import openvino as ov

    try:
        import nncf

        nncf_version = getattr(nncf, "__version__", None)
    except Exception:  # noqa: BLE001
        nncf_version = None

    ov_opts = ov_config["openvino"]
    batches = batches or [int(b) for b in ov_opts["batches"]]
    ckpt = resolve_checkpoint_dir(checkpoint)
    paths = ArtifactPaths.for_checkpoint(ckpt)
    paths.root.mkdir(parents=True, exist_ok=True)

    net, config, charset = build_keras_net(ckpt)
    seed = int(config.get("project", {}).get("seed", 42))
    pp = config.get("preprocess", {})
    pad_value = imagenet_pad_value(pp)
    wds = int(config.get("model", {}).get("width_downsample", 4))
    weights_path = checkpoint_weights_path(ckpt)
    weights_hash = file_sha256(weights_path)

    calib_arrays, _ = load_split(
        config,
        split=str(ov_opts["calib_split"]),
        max_samples=int(ov_opts["calib_samples"]),
        seed=seed,
    )
    logger.info("Calibration: %d images from split=%s", len(calib_arrays), ov_opts["calib_split"])

    variants: list[dict[str, Any]] = []
    for batch in batches:
        shape = shape_from_checkpoint(ckpt, batch=batch)

        fp16_model = to_ov_model(net, shape)
        save_ir(fp16_model, paths.model_xml("fp16", batch), compress_to_fp16=True)

        dataset, n_items = calibration_dataset(calib_arrays, shape, pad_value)
        subset_size = min(int(ov_opts["subset_size"]), n_items) or 1
        int8_model = quantize_model(fp16_model, dataset, subset_size=subset_size)
        save_ir(int8_model, paths.model_xml("int8", batch), compress_to_fp16=True)

        variants.append({"batch": batch, "shape": shape.input_shape, "calib_items": n_items})

    # Self-contained deploy assets next to the IR (charset, LM, config, build_info).
    save_sidecar_bundle(
        config,
        paths.root,
        source_root=ckpt,
        openvino_version=ov.__version__,
        nncf_version=nncf_version,
        source_checkpoint=ckpt.name,
        weights_sha256=weights_hash,
    )

    meta = {
        "source_checkpoint": ckpt.name,
        "charset": CHARSET_NAME,
        "num_classes": charset.num_classes,
        "width_downsample": wds,
        "input_layout": "NHWC",
        "normalize": pp.get("normalize"),
        "pad_value": pad_value,
        "weights_sha256": weights_hash,
        "weights_file": WEIGHTS_NAME,
        "openvino_version": ov.__version__,
        "nncf_version": nncf_version,
        "batches": batches,
        "precision_out": list(ov_opts.get("precision_out", ["fp16", "int8"])),
        "variants": variants,
        "default_decode": "beam_lm",
    }
    _write_meta(paths.meta_path, meta)

    cer = _accuracy_sanity_check(config, paths, ov_opts, seed)
    meta["accuracy_check"] = cer
    _write_meta(paths.meta_path, meta)

    logger.info("Convert done -> %s", paths.root)
    return meta


def _accuracy_sanity_check(
    config: dict[str, Any],
    paths: ArtifactPaths,
    ov_opts: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Quick CER FP16 vs INT8 on batch-1 IRs; flags large drops."""
    n = min(64, int(ov_opts["calib_samples"]))
    arrays, texts = load_split(
        config, split=str(ov_opts["calib_split"]), max_samples=n, seed=seed
    )
    result: dict[str, Any] = {"n": len(arrays), "split": ov_opts["calib_split"]}
    try:
        fp16 = OpenVINOCR.from_dir(paths.root, batch=1, precision="fp16")
        int8 = OpenVINOCR.from_dir(paths.root, batch=1, precision="int8")
        cer_fp16 = _mean_cer(fp16, arrays, texts)
        cer_int8 = _mean_cer(int8, arrays, texts)
        drop = cer_int8 - cer_fp16
        max_drop = float(ov_opts.get("max_cer_drop", 0.01))
        result.update(
            {
                "cer_fp16": round(cer_fp16, 4),
                "cer_int8": round(cer_int8, 4),
                "cer_drop": round(drop, 4),
                "max_cer_drop": max_drop,
                "warning": bool(drop > max_drop),
            }
        )
        if result["warning"]:
            logger.warning(
                "INT8 CER drop %.4f exceeds max_cer_drop %.4f (fp16=%.4f int8=%.4f)",
                drop, max_drop, cer_fp16, cer_int8,
            )
    except Exception as exc:  # noqa: BLE001 - sanity check must not fail convert
        logger.warning("Accuracy sanity check skipped: %s", exc)
        result["error"] = str(exc)
    return result


def _write_meta(meta_path: Path, meta: dict[str, Any]) -> None:
    import yaml

    with meta_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(meta, fh, allow_unicode=True, sort_keys=False)
