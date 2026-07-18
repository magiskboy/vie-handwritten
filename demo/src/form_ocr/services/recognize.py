"""Field ROI recognition (single-line MVP; multi-line hook for later)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import cv2
import numpy as np

from converter.runtime import OpenVINOCR
from form_ocr.services.crop import crop_roi
from vie_handwritten.preprocess import preprocess


class FieldRecognizer(Protocol):
    """Recognize text inside a normalized ROI of a full-page BGR image."""

    def recognize_roi(self, image_bgr: np.ndarray, bbox_norm: Sequence[float]) -> str: ...


class SingleLineRecognizer:
    """MVP: crop ROI → preprocess → OpenVINOCR.recognize.

    Future multi-line support can implement ``FieldRecognizer`` via
    ``segment_lines`` on the crop, then join line texts.
    """

    def __init__(self, ov: OpenVINOCR) -> None:
        self.ov = ov

    def recognize_roi(self, image_bgr: np.ndarray, bbox_norm: Sequence[float]) -> str:
        crop = crop_roi(image_bgr, bbox_norm)
        # OpenVINOCR preprocess expects BGR/gray like training pipeline.
        if crop.ndim == 2:
            bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        else:
            bgr = crop
        arr = preprocess(bgr, self.ov.config["preprocess"])
        return self.ov.recognize(arr)
