"""OCR model loading + inference for the GTK GUI (no GI imports).

Loads a checkpoint directory ``{model.weights.h5, config.yaml}`` and prefers
``beam_lm`` decoding when KenLM artifacts are present.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Callable

import editdistance

from vie_handwritten.eval import character_error_rate, word_error_rate
from vie_handwritten.model import OCRModel
from vie_handwritten.preprocess import load_image, preprocess
from vie_handwritten.utils import (
    CHECKPOINT_CONFIG_NAME,
    WEIGHTS_NAME,
    abs_path,
    configure_runtime,
    load_checkpoint_config,
    project_root,
    resolve_checkpoint_dir,
)

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

_LM_PATH = "lm/vi.binary"
_UNIGRAMS_PATH = "lm/unigrams.txt"
_LEXICON_PATH = "data/charset/vi_syllables.txt"


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


def _prepare_config(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Force ``beam_lm`` when LM files exist; otherwise fall back to ``greedy``.

    Returns ``(config, note)`` where ``note`` explains any fallback.
    """
    ctc = config.setdefault("ctc", {})
    lm = abs_path(ctc.get("lm_path") or _LM_PATH)
    unigrams = abs_path(ctc.get("unigrams_path") or _UNIGRAMS_PATH)
    lexicon = abs_path(ctc.get("lexicon_path") or _LEXICON_PATH)

    ctc["lm_path"] = str(lm)
    ctc["unigrams_path"] = str(unigrams)
    ctc["lexicon_path"] = str(lexicon)
    ctc.setdefault("beam_width", 100)
    ctc.setdefault("alpha", 0.5)
    ctc.setdefault("beta", 1.0)

    note = ""
    if lm.is_file():
        ctc["decode"] = "beam_lm"
        if int(ctc.get("beam_width", 10)) < 50:
            ctc["beam_width"] = 100
    else:
        ctc["decode"] = "greedy"
        note = f"LM missing ({lm}); using greedy decode"
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


class ModelService:
    """Thread-safe OCR service: load once, recognize many images."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: OCRModel | None = None
        self._config: dict[str, Any] | None = None
        self._checkpoint_dir: Path | None = None
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
        return {
            "ready": self.ready,
            "weights": str(ckpt / WEIGHTS_NAME) if ckpt else "",
            "config": str(ckpt / CHECKPOINT_CONFIG_NAME) if ckpt else "",
            "checkpoint": str(ckpt) if ckpt else "",
            "decode": ctc.get("decode", ""),
            "decode_note": self._decode_note,
            "beam_width": ctc.get("beam_width", ""),
            "num_classes": charset_n if charset_n is not None else "",
            "target_height": pp.get("target_height", ""),
            "max_width": pp.get("max_width", ""),
            "gpus": self._runtime.get("gpus", []),
            "gpu_names": self._runtime.get("gpu_names", []),
            "tensorflow": self._runtime.get("tensorflow", ""),
            "project_root": str(project_root()),
        }

    def load(self, checkpoint: str | Path) -> dict[str, Any]:
        """Load a checkpoint directory (blocking). Prefer calling from a worker thread."""
        root = resolve_checkpoint_dir(checkpoint)
        config, note = _prepare_config(load_checkpoint_config(root))

        self._runtime = configure_runtime()
        self._runtime["gpu_names"] = _gpu_display_names()
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
            self._decode_note = note
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

    def recognize_lines(self, line_images: list["np.ndarray"]) -> list[tuple[str, float]]:
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
