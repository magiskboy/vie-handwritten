"""Line segmentation for handwritten document images.

Pipeline:
    image → grayscale → DPI normalization → binarization
          → horizontal projection → stability threshold → linear boundaries
          → seam carving refinement → non-linear line extraction

Public API:
    segment_lines(image, ...) → list[np.ndarray]
    SegmentationResult (dataclass with all intermediates)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from skimage.morphology import thin

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TARGET_DPI = 150
MAX_DIMENSION = 2048
ASSUMED_LINE_HEIGHT_MM = 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SegmentationResult:
    """Holds all intermediate and final outputs of line segmentation."""

    lines: list[np.ndarray]
    """Extracted line images (binary, ink=255, bg=0)."""

    lines_gray: list[np.ndarray]
    """Extracted line images (grayscale, natural: dark text on white bg)."""

    boundaries: list[tuple[int, int]]
    """Linear (y_start, y_end) per detected line."""

    seams: list[np.ndarray]
    """Non-linear seam curves between lines. Length = len(boundaries) - 1."""

    binary: np.ndarray
    """Full-page binary image used for segmentation."""

    normalized: np.ndarray
    """Normalized grayscale image (after DPI rescaling + dimension limiting)."""

    source_dpi: float
    target_dpi: float
    threshold_ratio: float
    n_lines: int

    @property
    def line_heights(self) -> list[int]:
        return [img.shape[0] for img in self.lines]


# ─────────────────────────────────────────────────────────────────────────────
# Grayscale conversion
# ─────────────────────────────────────────────────────────────────────────────


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ─────────────────────────────────────────────────────────────────────────────
# DPI estimation
# ─────────────────────────────────────────────────────────────────────────────


def _estimate_stroke_width(gray: np.ndarray) -> float:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    skeleton = thin(binary > 0)
    if skeleton.sum() == 0:
        return 2.0
    return float(np.median(dist[skeleton]) * 2)


def _estimate_line_height(gray: np.ndarray) -> float | None:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    projection = binary.sum(axis=1).astype(np.float64)

    kernel_size = max(3, int(gray.shape[0] * 0.01) | 1)
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(projection, kernel, mode="same")

    threshold = smoothed.max() * 0.2
    above = smoothed > threshold

    padded = np.concatenate([[False], above, [False]])
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    if len(starts) == 0 or len(ends) == 0:
        return None
    if ends[0] < starts[0]:
        ends = ends[1:]
    min_len = min(len(starts), len(ends))
    starts, ends = starts[:min_len], ends[:min_len]

    centers = (starts + ends) / 2.0
    if len(centers) < 2:
        return None
    return float(np.median(np.diff(centers)))


def _estimate_dpi(gray: np.ndarray) -> float:
    line_h = _estimate_line_height(gray)
    if line_h is not None and line_h >= 10:
        return line_h / (ASSUMED_LINE_HEIGHT_MM / 25.4)
    stroke = _estimate_stroke_width(gray)
    return stroke / (0.4 / 25.4)


# ─────────────────────────────────────────────────────────────────────────────
# Image normalization
# ─────────────────────────────────────────────────────────────────────────────


def _rescale(image: np.ndarray, source_dpi: float, target_dpi: float) -> np.ndarray:
    scale = target_dpi / source_dpi
    if abs(scale - 1.0) < 0.05:
        return image
    h, w = image.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def _limit_dimension(image: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _binarize(gray: np.ndarray, method: str = "sauvola", **kwargs) -> np.ndarray:
    """Binarize to ink=255, background=0."""
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    if method == "sauvola":
        window_size = kwargs.get("window_size", 25)
        k = kwargs.get("k", 0.2)
        if window_size % 2 == 0:
            window_size += 1
        thresh_map = threshold_sauvola(gray, window_size=window_size, k=k)
        return ((gray < thresh_map) * 255).astype(np.uint8)

    if method == "adaptive":
        block_size = kwargs.get("block_size", 31)
        C = kwargs.get("C", 10)
        if block_size % 2 == 0:
            block_size += 1
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, C,
        )

    raise ValueError(f"Unknown binarization method: {method}")


# ─────────────────────────────────────────────────────────────────────────────
# Horizontal projection & threshold
# ─────────────────────────────────────────────────────────────────────────────


def _smooth_projection(projection: np.ndarray, image_height: int) -> np.ndarray:
    kernel_size = max(3, int(image_height * 0.01) | 1)
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(projection, kernel, mode="same")


def _find_boundaries_at_thresh(
    smoothed: np.ndarray,
    thresh_ratio: float,
    min_line_height_ratio: float,
) -> list[tuple[int, int]]:
    """Given smoothed projection and threshold ratio, return line boundaries."""
    threshold = smoothed.max() * thresh_ratio
    is_ink = smoothed > threshold

    padded = np.concatenate([[False], is_ink, [False]])
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    if len(starts) == 0:
        return []

    boundaries = list(zip(starts.tolist(), ends.tolist()))

    if len(boundaries) >= 2:
        heights = [e - s for s, e in boundaries]
        median_h = np.median(heights)
        min_h = median_h * min_line_height_ratio
        boundaries = [(s, e) for s, e in boundaries if (e - s) >= min_h]

    return boundaries


def _auto_thresh_stability(
    smoothed: np.ndarray,
    min_line_height_ratio: float = 0.3,
    sweep_range: tuple[float, float] = (0.01, 0.50),
    n_steps: int = 50,
) -> float:
    """Find optimal threshold by locating the longest stability plateau."""
    thresholds = np.linspace(sweep_range[0], sweep_range[1], n_steps)
    line_counts = np.array([
        len(_find_boundaries_at_thresh(smoothed, t, min_line_height_ratio))
        for t in thresholds
    ])

    # Longest run of identical counts
    best_start = 0
    best_len = 1
    cur_start = 0

    for i in range(1, len(line_counts)):
        if line_counts[i] == line_counts[cur_start]:
            run_len = i - cur_start + 1
            if run_len > best_len:
                best_len = run_len
                best_start = cur_start
        else:
            cur_start = i

    best_mid = best_start + best_len // 2
    return float(thresholds[best_mid])


# ─────────────────────────────────────────────────────────────────────────────
# Seam carving (non-linear boundary refinement)
# ─────────────────────────────────────────────────────────────────────────────


def _compute_energy_map(binary: np.ndarray) -> np.ndarray:
    """Distance-transform energy: low energy far from ink."""
    background = (binary == 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(background, cv2.DIST_L2, 5)
    max_dist = dist.max()
    if max_dist == 0:
        return np.ones_like(binary, dtype=np.float64)
    return (max_dist - dist).astype(np.float64)


def _find_seam(energy_strip: np.ndarray) -> np.ndarray:
    """DP minimum-energy horizontal seam (left → right)."""
    h, w = energy_strip.shape
    if h == 0 or w == 0:
        return np.zeros(w, dtype=np.int32)

    dp = np.full((h, w), np.inf, dtype=np.float64)
    dp[:, 0] = energy_strip[:, 0]

    for x in range(1, w):
        dp[:, x] = dp[:, x - 1] + energy_strip[:, x]
        dp[1:, x] = np.minimum(dp[1:, x], dp[:-1, x - 1] + energy_strip[1:, x])
        dp[:-1, x] = np.minimum(dp[:-1, x], dp[1:, x - 1] + energy_strip[:-1, x])

    seam = np.zeros(w, dtype=np.int32)
    seam[w - 1] = np.argmin(dp[:, w - 1])

    for x in range(w - 2, -1, -1):
        row = seam[x + 1]
        r_min = max(0, row - 1)
        r_max = min(h - 1, row + 1)
        seam[x] = r_min + np.argmin(dp[r_min : r_max + 1, x])

    return seam


def _compute_seams(
    binary: np.ndarray,
    boundaries: list[tuple[int, int]],
    margin: int = 5,
) -> list[np.ndarray]:
    """Compute a non-linear seam between each pair of adjacent lines."""
    if len(boundaries) < 2:
        return []

    energy = _compute_energy_map(binary)
    img_h, img_w = binary.shape
    seams = []

    for i in range(len(boundaries) - 1):
        gap_top = max(0, boundaries[i][1] - margin)
        gap_bot = min(img_h, boundaries[i + 1][0] + margin)

        if gap_bot <= gap_top:
            mid = (boundaries[i][1] + boundaries[i + 1][0]) // 2
            seams.append(np.full(img_w, mid, dtype=np.int32))
            continue

        strip = energy[gap_top:gap_bot, :]
        seam_local = _find_seam(strip)
        seams.append(seam_local + gap_top)

    return seams


# ─────────────────────────────────────────────────────────────────────────────
# Line extraction using seam curves
# ─────────────────────────────────────────────────────────────────────────────


def _extract_lines(
    binary: np.ndarray,
    boundaries: list[tuple[int, int]],
    seams: list[np.ndarray],
    grayscale: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Cut lines using seam curves. Pixels outside boundary are zeroed.

    Returns (binary_lines, grayscale_lines). Grayscale lines have white bg
    outside seam boundaries for clean OCR input.
    """
    img_h, img_w = binary.shape
    n_lines = len(boundaries)
    lines_bin = []
    lines_gray = []

    for i in range(n_lines):
        top_seam = seams[i - 1] if i > 0 else np.zeros(img_w, dtype=np.int32)
        bot_seam = seams[i] if i < n_lines - 1 else np.full(img_w, img_h - 1, dtype=np.int32)

        y_min = int(top_seam.min())
        y_max = min(int(bot_seam.max()) + 1, img_h)

        strip_bin = binary[y_min:y_max, :].copy()

        if grayscale is not None:
            strip_gray = grayscale[y_min:y_max, :].copy()
        else:
            strip_gray = np.full_like(strip_bin, 255)

        for x in range(img_w):
            local_top = top_seam[x] - y_min
            local_bot = bot_seam[x] - y_min
            if local_top > 0:
                strip_bin[:local_top, x] = 0
                strip_gray[:local_top, x] = 255
            if local_bot < strip_bin.shape[0]:
                strip_bin[local_bot:, x] = 0
                strip_gray[local_bot:, x] = 255

        lines_bin.append(strip_bin)
        lines_gray.append(strip_gray)

    return lines_bin, lines_gray


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def segment_lines(
    image: np.ndarray | str | Path,
    *,
    known_dpi: float | None = None,
    target_dpi: float = TARGET_DPI,
    max_dimension: int = MAX_DIMENSION,
    bin_method: str = "sauvola",
    min_line_height_ratio: float = 0.3,
    seam_margin: int = 5,
) -> SegmentationResult:
    """Segment a handwritten document image into individual text lines.

    Args:
        image: Input image (BGR/gray ndarray, or path to image file).
        known_dpi: Source DPI if known; otherwise estimated from content.
        target_dpi: Normalize to this DPI before segmentation.
        max_dimension: Cap largest dimension to this value.
        bin_method: Binarization method ("sauvola", "otsu", "adaptive").
        min_line_height_ratio: Reject lines shorter than this fraction of median.
        seam_margin: Pixels of margin around gap for seam search.

    Returns:
        SegmentationResult with extracted lines and metadata.
    """
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
        image = img

    gray = _to_grayscale(image)

    source_dpi = known_dpi if known_dpi is not None else _estimate_dpi(gray)

    normalized = _rescale(gray, source_dpi, target_dpi)
    normalized = _limit_dimension(normalized, max_dimension)
    binary = _binarize(normalized, method=bin_method)

    # Projection + auto threshold (stability)
    projection = binary.sum(axis=1).astype(np.float64)
    smoothed = _smooth_projection(projection, binary.shape[0])
    thresh_ratio = _auto_thresh_stability(smoothed, min_line_height_ratio)

    # Linear boundaries
    boundaries = _find_boundaries_at_thresh(smoothed, thresh_ratio, min_line_height_ratio)

    # Non-linear seam refinement
    seams = _compute_seams(binary, boundaries, margin=seam_margin)

    # Extract lines (binary + grayscale for OCR)
    if boundaries:
        lines, lines_gray = _extract_lines(binary, boundaries, seams, grayscale=normalized)
    else:
        lines, lines_gray = [], []

    return SegmentationResult(
        lines=lines,
        lines_gray=lines_gray,
        boundaries=boundaries,
        seams=seams,
        binary=binary,
        normalized=normalized,
        source_dpi=source_dpi,
        target_dpi=target_dpi,
        threshold_ratio=thresh_ratio,
        n_lines=len(lines),
    )
