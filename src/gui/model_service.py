"""OCR model loading + inference for the GTK GUI (no GI imports).

Auto-detects artifact format:

* Keras checkpoint (``model.weights.h5`` + sidecars) → TensorFlow/Keras, prefer GPU
* OpenVINO artifact (``meta.yaml`` + IR sidecars) → OpenVINO, CPU only

Decode defaults to ``beam_lm`` when ``lm/vi.binary`` is present.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable, Literal

import editdistance
import numpy as np

from vie_handwritten.eval import character_error_rate, word_error_rate
from vie_handwritten.preprocess import load_image, preprocess
from vie_handwritten.utils import (
    ARTIFACT_LM_BINARY_REL,
    ARTIFACT_LM_LEXICON_REL,
    ARTIFACT_LM_UNIGRAMS_REL,
    BUILD_INFO_NAME,
    CHECKPOINT_CONFIG_NAME,
    WEIGHTS_NAME,
    artifact_has_lm,
    configure_runtime,
    load_checkpoint_config,
    load_config,
    project_root,
    resolve_artifact_path,
    resolve_checkpoint_dir,
    resolve_ctc_paths,
)

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
ModelBackend = Literal["keras", "openvino"]

# Preferred OpenVINO IR variants for interactive GUI (latency over throughput).
_OV_VARIANT_PREFERENCE: tuple[tuple[str, int], ...] = (
    ("int8", 1),
    ("fp16", 1),
    ("int8", 16),
    ("fp16", 16),
)


def list_images(folder: str | Path) -> list[Path]:
    """Return sorted image paths under ``folder`` (non-recursive)."""
    root = Path(folder)
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def load_folder_labels(folder: str | Path) -> dict[str, str]:
    """Load ``label.json`` / ``labels.json`` from ``folder`` → ``{filename: text}``.

    Keys are matched by basename (``12.jpg``). Values are NFC-normalized.
    """
    root = Path(folder)
    for name in ("label.json", "labels.json"):
        path = root / name
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read labels %s: %s", path, exc)
            return {}
        if not isinstance(data, dict):
            logger.warning("Skipping non-object label file: %s", path)
            return {}
        out: dict[str, str] = {}
        for key, value in data.items():
            fname = Path(str(key)).name
            out[fname] = unicodedata.normalize("NFC", str(value))
        logger.info("Loaded %d labels from %s", len(out), path)
        return out
    return {}


def lookup_label(labels: dict[str, str], image_path: str | Path) -> str | None:
    """Resolve ground-truth text for an image path, if present in ``labels``."""
    if not labels:
        return None
    name = Path(image_path).name
    if name in labels:
        return labels[name]
    stem = Path(name).stem
    for key, text in labels.items():
        if Path(key).stem == stem:
            return text
    return None


def compare_prediction(reference: str, hypothesis: str) -> dict[str, Any]:
    """Levenshtein distance + CER/WER between ground truth and prediction."""
    ref = unicodedata.normalize("NFC", reference or "")
    hyp = unicodedata.normalize("NFC", hypothesis or "")
    dist = int(editdistance.eval(ref, hyp))
    return {
        "reference": ref,
        "hypothesis": hyp,
        "levenshtein": dist,
        "cer": character_error_rate(ref, hyp),
        "wer": word_error_rate(ref, hyp),
        "ref_len": len(ref),
        "exact": ref == hyp,
    }


def detect_model_format(path: str | Path) -> ModelBackend:
    """Detect whether ``path`` is a Keras checkpoint or OpenVINO artifact directory."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Model path must be a directory: {path}")

    has_keras = (root / WEIGHTS_NAME).is_file() and (root / CHECKPOINT_CONFIG_NAME).is_file()
    has_ov = (root / "meta.yaml").is_file() and (root / CHECKPOINT_CONFIG_NAME).is_file()

    if has_ov and not has_keras:
        return "openvino"
    if has_keras and not has_ov:
        return "keras"
    if has_ov and has_keras:
        # Prefer OpenVINO when meta.yaml marks a deploy artifact.
        return "openvino"
    raise FileNotFoundError(
        f"Not a Keras checkpoint or OpenVINO artifact: {root} "
        f"(need {WEIGHTS_NAME} or meta.yaml + {CHECKPOINT_CONFIG_NAME})"
    )


def _prepare_config(
    config: dict[str, Any], checkpoint: str | Path
) -> tuple[dict[str, Any], str]:
    """Force ``beam_lm`` when LM files exist in the checkpoint; else greedy.

    Returns ``(config, note)`` where ``note`` explains any fallback.
    """
    root = Path(checkpoint)
    config = resolve_ctc_paths(config, root)
    ctc = config.setdefault("ctc", {})
    # Ensure relative defaults resolve inside the artifact if not already set.
    for key, rel in (
        ("lm_path", ARTIFACT_LM_BINARY_REL),
        ("unigrams_path", ARTIFACT_LM_UNIGRAMS_REL),
        ("lexicon_path", ARTIFACT_LM_LEXICON_REL),
    ):
        if not ctc.get(key):
            ctc[key] = str(resolve_artifact_path(root, rel))
        else:
            ctc[key] = str(resolve_artifact_path(root, ctc[key]))

    ctc.setdefault("beam_width", 100)
    ctc.setdefault("alpha", 0.5)
    ctc.setdefault("beta", 1.0)

    note = ""
    lm = Path(ctc["lm_path"])
    if artifact_has_lm(root) or lm.is_file():
        ctc["decode"] = "beam_lm"
        if int(ctc.get("beam_width", 10)) < 50:
            ctc["beam_width"] = 100
    else:
        ctc["decode"] = "greedy"
        note = f"LM missing in checkpoint ({root / ARTIFACT_LM_BINARY_REL}); using greedy decode"
        logger.warning(note)
    return config, note


def _gpu_display_names() -> list[str]:
    """Human-readable GPU names from TensorFlow device details."""
    import tensorflow as tf

    names: list[str] = []
    for gpu in tf.config.list_physical_devices("GPU"):
        detail = {}
        try:
            detail = tf.config.experimental.get_device_details(gpu) or {}
        except Exception:  # noqa: BLE001
            detail = {}
        name = detail.get("device_name") or detail.get("name") or gpu.name
        names.append(str(name).strip())
    return names


def _pick_ov_variant(ov_dir: Path) -> tuple[str, int]:
    """Choose an IR under ``ov_dir`` (prefer int8 batch-1 for interactive use)."""
    from converter.config import ArtifactPaths

    paths = ArtifactPaths.for_dir(ov_dir)
    for precision, batch in _OV_VARIANT_PREFERENCE:
        if paths.model_xml(precision, batch).is_file():
            return precision, batch

    # Fallback: scan ``<precision>_b<batch>/model.xml`` directories.
    found: list[tuple[str, int]] = []
    for child in sorted(ov_dir.iterdir()):
        if not child.is_dir() or not (child / "model.xml").is_file():
            continue
        name = child.name
        if "_b" not in name:
            continue
        precision, _, batch_s = name.rpartition("_b")
        if precision and batch_s.isdigit():
            found.append((precision, int(batch_s)))
    if found:
        found.sort(key=lambda pb: (0 if pb[0] == "int8" else 1, pb[1], pb[0]))
        return found[0]

    raise FileNotFoundError(
        f"No OpenVINO IR found under {ov_dir} "
        f"(expected e.g. int8_b1/model.xml or fp16_b1/model.xml)"
    )


class ModelService:
    """Thread-safe OCR service: load once, recognize many images."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: Any | None = None
        self._config: dict[str, Any] | None = None
        self._checkpoint_dir: Path | None = None
        self._backend: ModelBackend | None = None
        self._decode_note: str = ""
        self._runtime: dict[str, Any] = {}
        self._busy = False

    @property
    def ready(self) -> bool:
        return self._model is not None

    @property
    def busy(self) -> bool:
        return self._busy

    def info(self) -> dict[str, Any]:
        """Snapshot of loaded model metadata for the info panel."""
        cfg = self._config or {}
        pp = cfg.get("preprocess", {})
        ctc = cfg.get("ctc", {})
        charset_n = None
        if self._model is not None:
            charset_n = self._model.charset.num_classes
        ckpt = self._checkpoint_dir
        backend = self._backend
        weights = ""
        if ckpt and backend == "keras":
            weights = str(ckpt / WEIGHTS_NAME)
        elif ckpt and backend == "openvino":
            precision = self._runtime.get("precision", "")
            batch = self._runtime.get("batch", "")
            if precision and batch != "":
                weights = str(ckpt / f"{precision}_b{batch}" / "model.xml")
        return {
            "ready": self.ready,
            "backend": backend or "",
            "weights": weights,
            "config": str(ckpt / CHECKPOINT_CONFIG_NAME) if ckpt else "",
            "checkpoint": str(ckpt) if ckpt else "",
            "decode": ctc.get("decode", "") or self._runtime.get("decode", ""),
            "decode_note": self._decode_note,
            "beam_width": ctc.get("beam_width", ""),
            "num_classes": charset_n if charset_n is not None else "",
            "target_height": pp.get("target_height", ""),
            "max_width": pp.get("max_width", ""),
            "gpus": self._runtime.get("gpus", []),
            "gpu_names": self._runtime.get("gpu_names", []),
            "device": self._runtime.get("device", ""),
            "precision": self._runtime.get("precision", ""),
            "batch": self._runtime.get("batch", ""),
            "tensorflow": self._runtime.get("tensorflow", ""),
            "openvino": self._runtime.get("openvino", ""),
            "project_root": str(project_root()),
        }

    def load(self, checkpoint: str | Path) -> dict[str, Any]:
        """Load a Keras or OpenVINO artifact directory (blocking)."""
        root = Path(checkpoint)
        backend = detect_model_format(root)
        if backend == "keras":
            return self._load_keras(root)
        return self._load_openvino(root)

    def _load_keras(self, checkpoint: Path) -> dict[str, Any]:
        """Load Keras CRNN; prefer GPU via TensorFlow runtime config."""
        from vie_handwritten.model import OCRModel

        root = resolve_checkpoint_dir(checkpoint)
        config, note = _prepare_config(load_checkpoint_config(root), root)

        runtime = configure_runtime()
        runtime["gpu_names"] = _gpu_display_names()
        runtime["device"] = "GPU" if runtime.get("gpu_count") else "CPU"
        runtime["backend"] = "keras"
        try:
            model = OCRModel.from_checkpoint(root, config=config)
        except Exception as exc:  # noqa: BLE001
            if config.get("ctc", {}).get("decode") == "beam_lm":
                logger.warning("beam_lm failed (%s); retrying with greedy", exc)
                config["ctc"]["decode"] = "greedy"
                note = f"beam_lm failed ({exc}); using greedy decode"
                model = OCRModel.from_checkpoint(root, config=config)
            else:
                raise

        with self._lock:
            self._model = model
            self._config = model.config
            self._checkpoint_dir = root
            self._backend = "keras"
            self._decode_note = note
            self._runtime = runtime
        logger.info(
            "Loaded Keras checkpoint %s on %s (decode=%s)",
            root,
            runtime["device"],
            model.config.get("ctc", {}).get("decode"),
        )
        return self.info()

    def _load_openvino(self, ov_dir: Path) -> dict[str, Any]:
        """Load OpenVINO IR; always compile on CPU."""
        try:
            from converter.runtime import OpenVINOCR, resolve_openvino_dir
        except ImportError as exc:
            raise ImportError(
                "OpenVINO artifact selected but the openvino package is not installed. "
                "Install with: uv sync --extra openvino"
            ) from exc

        root = resolve_openvino_dir(ov_dir)
        precision, batch = _pick_ov_variant(root)
        # Force CPU regardless of host GPUs.
        model = OpenVINOCR.from_dir(root, batch=batch, precision=precision, device="CPU")

        note = ""
        if model.decoder.method != "beam_lm":
            note = (
                f"LM missing or unavailable in artifact; using {model.decoder.method} decode"
            )

        # Keep config decode field in sync for the info panel.
        config = dict(model.config)
        ctc = dict(config.get("ctc", {}))
        ctc["decode"] = model.decoder.method
        ctc["beam_width"] = model.decoder.beam_width
        config["ctc"] = ctc

        ov_ver = ""
        try:
            import openvino as ov

            ov_ver = getattr(ov, "__version__", "") or ""
        except Exception:  # noqa: BLE001
            ov_ver = ""

        runtime: dict[str, Any] = {
            "backend": "openvino",
            "device": "CPU",
            "precision": precision,
            "batch": batch,
            "decode": model.decoder.method,
            "openvino": ov_ver,
            "gpus": [],
            "gpu_names": [],
            "tensorflow": "",
        }
        # Prefer version recorded at convert time when present.
        build_info = root / BUILD_INFO_NAME
        if build_info.is_file():
            try:
                bi = load_config(build_info)
                ov_ver = bi.get("openvino_version") or ov_ver
                runtime["openvino"] = ov_ver
            except Exception:  # noqa: BLE001
                pass

        with self._lock:
            self._model = model
            self._config = config
            self._checkpoint_dir = root
            self._backend = "openvino"
            self._decode_note = note
            self._runtime = runtime
        return self.info()

    def recognize(self, image_path: str | Path) -> tuple[str, float]:
        """Run OCR on one image (blocking). Returns ``(text, elapsed_ms)``."""
        with self._lock:
            model = self._model
        if model is None:
            raise RuntimeError("Model is not loaded")

        path = Path(image_path)
        t0 = time.perf_counter()
        arr = preprocess(load_image(str(path)), model.config["preprocess"])
        text = model.recognize(arr)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return text, elapsed_ms

    def recognize_lines(self, line_images: list[np.ndarray]) -> list[tuple[str, float]]:
        """Run OCR on multiple pre-segmented line images (blocking).

        Each line_image should be a grayscale ndarray (dark text on white bg).
        Use SegmentationResult.lines_gray for best results.
        Returns list of ``(text, elapsed_ms)`` per line.
        """
        with self._lock:
            model = self._model
        if model is None:
            raise RuntimeError("Model is not loaded")

        pp_config = model.config["preprocess"]
        results: list[tuple[str, float]] = []
        for img in line_images:
            t0 = time.perf_counter()
            arr = preprocess(img, pp_config)
            text = model.recognize(arr)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            results.append((text, elapsed_ms))
        return results

    def load_async(
        self,
        checkpoint: str | Path,
        on_done: Callable[[dict[str, Any] | None, BaseException | None], None],
    ) -> bool:
        """Start a background load. Returns False if already busy."""
        if self._busy:
            return False
        self._busy = True

        def _run() -> None:
            info: dict[str, Any] | None = None
            err: BaseException | None = None
            try:
                info = self.load(checkpoint)
            except BaseException as exc:  # noqa: BLE001
                err = exc
            finally:
                self._busy = False
                on_done(info, err)

        threading.Thread(target=_run, name="ocr-load", daemon=True).start()
        return True

    def recognize_async(
        self,
        image_path: str | Path,
        on_done: Callable[[str | None, float | None, BaseException | None], None],
    ) -> bool:
        """Start a background recognize. Returns False if already busy.

        ``on_done(text, elapsed_ms, error)`` — ``elapsed_ms`` is set on success.
        """
        if self._busy:
            return False
        self._busy = True
        path = Path(image_path)

        def _run() -> None:
            text: str | None = None
            elapsed_ms: float | None = None
            err: BaseException | None = None
            try:
                text, elapsed_ms = self.recognize(path)
            except BaseException as exc:  # noqa: BLE001
                err = exc
            finally:
                self._busy = False
                on_done(text, elapsed_ms, err)

        threading.Thread(target=_run, name="ocr-infer", daemon=True).start()
        return True
