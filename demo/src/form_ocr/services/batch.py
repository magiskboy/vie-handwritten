"""Batch extract field values from scan images."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2

from form_ocr.domain.records import ExtractionResult, FieldValue, Record
from form_ocr.domain.template import FormTemplate
from form_ocr.services.recognize import FieldRecognizer

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

ProgressCb = Callable[[int, int, str], None]  # done, total, filename
DoneCb = Callable[[ExtractionResult | None, BaseException | None], None]


def list_images(folder: str | Path) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def extract_image(
    path: Path,
    template: FormTemplate,
    recognizer: FieldRecognizer,
) -> Record:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return Record(filename=path.name, error=f"Không đọc được ảnh: {path.name}")
    fields: list[FieldValue] = []
    for fd in template.fields:
        try:
            text = recognizer.recognize_roi(image, fd.bbox)
        except Exception as exc:  # noqa: BLE001 — per-field resilience
            logger.warning("ROI %s on %s failed: %s", fd.label, path.name, exc)
            text = ""
        fields.append(FieldValue(label=fd.label, value=text))
    return Record(filename=path.name, fields=fields)


def extract_batch(
    paths: Sequence[Path],
    template: FormTemplate,
    recognizer: FieldRecognizer,
    *,
    on_progress: ProgressCb | None = None,
) -> ExtractionResult:
    result = ExtractionResult()
    total = len(paths)
    for i, path in enumerate(paths):
        try:
            rec = extract_image(path, template, recognizer)
        except Exception as exc:  # noqa: BLE001
            logger.exception("extract failed for %s", path)
            rec = Record(filename=path.name, error=str(exc))
        result.records.append(rec)
        if on_progress:
            on_progress(i + 1, total, path.name)
    return result


class BatchExtractor:
    """Run ``extract_batch`` on a background thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._busy = False

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def run_async(
        self,
        paths: Sequence[Path],
        template: FormTemplate,
        recognizer: FieldRecognizer,
        *,
        on_progress: ProgressCb | None,
        on_done: DoneCb,
    ) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        def _run() -> None:
            try:
                result = extract_batch(
                    paths, template, recognizer, on_progress=on_progress
                )
                on_done(result, None)
            except BaseException as exc:  # noqa: BLE001
                on_done(None, exc)
            finally:
                with self._lock:
                    self._busy = False

        threading.Thread(target=_run, daemon=True).start()
        return True
