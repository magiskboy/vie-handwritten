"""Line segmentation service for the GTK GUI (no GI imports).

Provides async segmentation with auto-detection of multi-line images.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from vie_handwritten.segment import SegmentationResult, segment_lines

logger = logging.getLogger(__name__)

MULTILINE_MIN_LINES = 2
MULTILINE_MIN_HEIGHT_PX = 100


def is_multiline(image_path: str | Path) -> bool:
    """Heuristic: detect if an image likely contains multiple text lines.

    Criteria: image height > threshold AND estimated line count >= 2.
    Uses a fast projection check without full segmentation.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    h, w = img.shape
    if h < MULTILINE_MIN_HEIGHT_PX:
        return False

    # Quick binarize + projection
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    projection = binary.sum(axis=1).astype(np.float64)

    # Simple valley counting with fixed threshold
    if projection.max() == 0:
        return False
    threshold = projection.max() * 0.15
    is_ink = projection > threshold

    padded = np.concatenate([[False], is_ink, [False]])
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]

    return len(starts) >= MULTILINE_MIN_LINES


class SegmentService:
    """Thread-safe line segmentation service."""

    def __init__(self) -> None:
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    def segment(
        self,
        image_path: str | Path,
        *,
        known_dpi: float | None = None,
        bin_method: str = "sauvola",
    ) -> SegmentationResult:
        """Run segmentation (blocking)."""
        return segment_lines(
            image_path,
            known_dpi=known_dpi,
            bin_method=bin_method,
        )

    def segment_async(
        self,
        image_path: str | Path,
        on_done: Callable[[SegmentationResult | None, float | None, BaseException | None], None],
        *,
        known_dpi: float | None = None,
        bin_method: str = "sauvola",
    ) -> bool:
        """Start background segmentation. Returns False if already busy.

        on_done(result, elapsed_ms, error)
        """
        if self._busy:
            return False
        self._busy = True

        def _run() -> None:
            result: SegmentationResult | None = None
            elapsed_ms: float | None = None
            err: BaseException | None = None
            try:
                t0 = time.perf_counter()
                result = self.segment(image_path, known_dpi=known_dpi, bin_method=bin_method)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
            except BaseException as exc:  # noqa: BLE001
                err = exc
            finally:
                self._busy = False
                on_done(result, elapsed_ms, err)

        threading.Thread(target=_run, name="segment", daemon=True).start()
        return True


def render_seam_overlay(
    image_path: str | Path,
    seg_result: SegmentationResult,
    *,
    line_color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw seam curves on the original image for display.

    Returns BGR image with seam lines drawn.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    # Scale seams from binary coords to original image coords
    orig_h, orig_w = img.shape[:2]
    bin_h, bin_w = seg_result.binary.shape
    scale_y = orig_h / bin_h
    scale_x = orig_w / bin_w

    for seam in seg_result.seams:
        pts = np.column_stack([
            (np.arange(len(seam)) * scale_x).astype(np.int32),
            (seam * scale_y).astype(np.int32),
        ])
        cv2.polylines(img, [pts], isClosed=False, color=line_color, thickness=thickness)

    return img
