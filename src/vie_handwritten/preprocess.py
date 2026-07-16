"""Image preprocessing for Vietnamese handwriting OCR.

Stack: OpenCV (CLAHE) + scikit-image (deskew / resize).

Pipeline:
  load → grayscale → CLAHE → deskew → resize height (keep aspect)
       → pad → RGB×3 → ImageNet normalize
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from skimage.transform import rotate

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(path: str) -> np.ndarray:
    """Load an image from disk (BGR or grayscale)."""
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB/gray input to a single-channel uint8 image."""
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0].astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def apply_clahe(
    image: np.ndarray,
    *,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization (OpenCV)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def deskew(image: np.ndarray) -> np.ndarray:
    """Estimate and correct skew angle (OpenCV moments / minAreaRect)."""
    inverted = 255 - image
    coords = cv2.findNonZero(inverted)
    if coords is None or len(coords) < 10:
        return image
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5 or abs(angle) > 15:
        return image
    return (rotate(image, angle, resize=False, cval=1.0, preserve_range=True)).astype(
        np.uint8
    )


def resize_keep_aspect(
    image: np.ndarray,
    *,
    target_height: int,
    max_width: int | None = None,
) -> np.ndarray:
    """Resize to fixed height while preserving aspect ratio."""
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Empty image")
    scale = target_height / float(h)
    new_w = max(1, int(round(w * scale)))
    if max_width is not None:
        new_w = min(new_w, max_width)
    return cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_AREA)


def pad_to_width(image: np.ndarray, width: int, *, pad_value: int = 255) -> np.ndarray:
    """Right-pad to a target width for batching."""
    h, w = image.shape[:2]
    if w >= width:
        return image[:, :width]
    if image.ndim == 2:
        out = np.full((h, width), pad_value, dtype=image.dtype)
        out[:, :w] = image
        return out
    c = image.shape[2]
    out = np.full((h, width, c), pad_value, dtype=image.dtype)
    out[:, :w] = image
    return out


def normalize(image: np.ndarray, *, mode: str | bool = "imagenet") -> np.ndarray:
    """Scale pixels to float32; ImageNet mean/std or [0, 1]."""
    x = image.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    if mode is True or mode == "01" or mode == "[0,1]":
        return x
    if mode is False:
        return x
    # imagenet
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    mean = IMAGENET_MEAN.reshape(1, 1, 3)
    std = IMAGENET_STD.reshape(1, 1, 3)
    return (x - mean) / std


def to_channels(image: np.ndarray, channels: int) -> np.ndarray:
    """Ensure ``(H, W, C)`` with the requested channel count."""
    if channels == 1:
        if image.ndim == 2:
            return image[:, :, None]
        return image[:, :, :1]
    if channels == 3:
        if image.ndim == 2:
            return np.stack([image, image, image], axis=-1)
        if image.shape[2] == 1:
            return np.repeat(image, 3, axis=2)
        return image[:, :, :3]
    raise ValueError(f"Unsupported channels={channels}")


def preprocess(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Full preprocess chain → model-ready array ``(H, W, C)``."""
    gray = to_grayscale(image)
    if config.get("use_clahe", True):
        tile = config.get("clahe_tile_grid", [8, 8])
        gray = apply_clahe(
            gray,
            clip_limit=float(config.get("clahe_clip_limit", 2.0)),
            tile_grid_size=(int(tile[0]), int(tile[1])),
        )
    gray = deskew(gray)
    gray = resize_keep_aspect(
        gray,
        target_height=int(config.get("target_height", 64)),
        max_width=config.get("max_width"),
    )
    channels = int(config.get("channels", 3))
    arr = to_channels(gray, channels)
    norm_mode = config.get("normalize", "imagenet")
    return normalize(arr, mode=norm_mode)
