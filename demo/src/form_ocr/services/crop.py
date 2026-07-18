"""Normalized bbox → pixel crop."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_MIN_SIDE_PX = 4


def norm_to_pixels(
    bbox: Sequence[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Convert normalized ``(x, y, w, h)`` to pixel ``(x0, y0, x1, y1)`` clamped."""
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image size")
    x, y, w, h = (float(v) for v in bbox)
    x0 = int(round(x * width))
    y0 = int(round(y * height))
    x1 = int(round((x + w) * width))
    y1 = int(round((y + h) * height))
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return x0, y0, x1, y1


def crop_roi(image: np.ndarray, bbox: Sequence[float]) -> np.ndarray:
    """Crop ``image`` (H×W×C or H×W) using a normalized bbox.

    Raises ``ValueError`` if the resulting crop is too small.
    """
    h, w = image.shape[:2]
    x0, y0, x1, y1 = norm_to_pixels(bbox, w, h)
    if (x1 - x0) < _MIN_SIDE_PX or (y1 - y0) < _MIN_SIDE_PX:
        raise ValueError(f"ROI quá nhỏ: {(x1 - x0)}×{(y1 - y0)} px")
    return image[y0:y1, x0:x1].copy()
