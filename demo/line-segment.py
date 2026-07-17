"""Line Segmentation Research — Step 1: Image Normalization.

Goal: Normalize images of varying size/DPI to a consistent representation
so that downstream heuristics (horizontal projection, connected components)
operate on predictable pixel dimensions.

Target: 150 DPI equivalent — stroke width ~2-4px, line height ~30-60px.

Usage:
    python demo/main.py path/to/image.jpg
    python demo/main.py path/to/image.jpg --dpi 300
    python demo/main.py path/to/folder/
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import thin

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

TARGET_DPI = 150
MAX_DIMENSION = 2048
ASSUMED_LINE_HEIGHT_MM = 10.0  # typical ruled notebook line spacing


# ─────────────────────────────────────────────────────────────────────
# 1. Load image
# ─────────────────────────────────────────────────────────────────────


def load_image(path: str | Path) -> np.ndarray:
    """Load image as-is (BGR, grayscale, or with alpha)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    return img


# ─────────────────────────────────────────────────────────────────────
# 2. Convert to grayscale
# ─────────────────────────────────────────────────────────────────────


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert any input to single-channel uint8."""
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ─────────────────────────────────────────────────────────────────────
# 3. Estimate DPI from image content (when metadata unavailable)
# ─────────────────────────────────────────────────────────────────────


def estimate_stroke_width(gray: np.ndarray) -> float:
    """Estimate median stroke width using distance transform on binarized text.

    Approach: binarize → distance transform → median of skeleton pixels.
    The distance at skeleton pixels equals half the local stroke width.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    skeleton = thin(binary > 0)
    if skeleton.sum() == 0:
        return 2.0

    stroke_half_widths = dist[skeleton]
    median_half = np.median(stroke_half_widths)
    return float(median_half * 2)


def estimate_line_height(gray: np.ndarray) -> float | None:
    """Estimate line height via horizontal projection profile peaks.

    Returns median distance between consecutive line centers, or None if
    fewer than 2 lines detected.
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    projection = binary.sum(axis=1).astype(np.float64)

    # Smooth to reduce noise
    kernel_size = max(3, int(gray.shape[0] * 0.01) | 1)
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(projection, kernel, mode="same")

    # Find peaks: rows where projection exceeds a threshold
    threshold = smoothed.max() * 0.2
    above = smoothed > threshold

    # Find contiguous runs (line regions)
    diffs = np.diff(above.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    if len(starts) == 0 or len(ends) == 0:
        return None

    # Align starts and ends
    if ends[0] < starts[0]:
        ends = ends[1:]
    min_len = min(len(starts), len(ends))
    starts, ends = starts[:min_len], ends[:min_len]

    centers = (starts + ends) / 2.0
    if len(centers) < 2:
        return None

    spacings = np.diff(centers)
    return float(np.median(spacings))


def estimate_dpi_from_content(gray: np.ndarray) -> float:
    """Estimate effective DPI from detected line height.

    Assumes lines are ~10mm apart (standard ruled paper).
    DPI = line_height_px / (LINE_HEIGHT_MM / 25.4)
    """
    line_height_px = estimate_line_height(gray)
    if line_height_px is None or line_height_px < 10:
        # Fallback: use stroke width, assume ballpoint pen ~0.4mm
        stroke_px = estimate_stroke_width(gray)
        pen_width_mm = 0.4
        return stroke_px / (pen_width_mm / 25.4)

    line_height_inch = ASSUMED_LINE_HEIGHT_MM / 25.4
    return line_height_px / line_height_inch


# ─────────────────────────────────────────────────────────────────────
# 4. Rescale to target DPI
# ─────────────────────────────────────────────────────────────────────


def rescale_to_target_dpi(
    image: np.ndarray,
    source_dpi: float,
    target_dpi: float = TARGET_DPI,
) -> np.ndarray:
    """Rescale image so that its effective resolution matches target_dpi."""
    scale = target_dpi / source_dpi
    if abs(scale - 1.0) < 0.05:
        return image

    h, w = image.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


# ─────────────────────────────────────────────────────────────────────
# 5. Limit max dimension
# ─────────────────────────────────────────────────────────────────────


def limit_dimension(image: np.ndarray, max_dim: int = MAX_DIMENSION) -> np.ndarray:
    """Downscale if any dimension exceeds max_dim, preserving aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image

    scale = max_dim / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ─────────────────────────────────────────────────────────────────────
# 6. Binarization
# ─────────────────────────────────────────────────────────────────────


def binarize(gray: np.ndarray, method: str = "sauvola", **kwargs) -> np.ndarray:
    """Binarize grayscale image (ink=255, background=0).

    Methods:
        - "otsu": Global Otsu threshold. Fast but struggles with uneven lighting.
        - "sauvola": Adaptive local threshold (scikit-image). Handles uneven
          backgrounds well — preferred for handwritten documents.
        - "adaptive": OpenCV adaptive Gaussian threshold.

    Returns uint8 binary image: ink pixels = 255, background = 0.
    """
    from skimage.filters import threshold_sauvola

    if method == "otsu":
        thresh_val, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    if method == "sauvola":
        window_size = kwargs.get("window_size", 25)
        k = kwargs.get("k", 0.2)
        if window_size % 2 == 0:
            window_size += 1
        thresh_map = threshold_sauvola(gray, window_size=window_size, k=k)
        binary = ((gray < thresh_map) * 255).astype(np.uint8)
        return binary

    if method == "adaptive":
        block_size = kwargs.get("block_size", 31)
        C = kwargs.get("C", 10)
        if block_size % 2 == 0:
            block_size += 1
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, C,
        )
        return binary

    raise ValueError(f"Unknown binarization method: {method}")


# ─────────────────────────────────────────────────────────────────────
# 7. Full normalization pipeline
# ─────────────────────────────────────────────────────────────────────


def normalize_for_segmentation(
    image: np.ndarray,
    known_dpi: float | None = None,
    target_dpi: float = TARGET_DPI,
    bin_method: str = "sauvola",
) -> dict:
    """Run the full normalization pipeline.

    Pipeline: grayscale → estimate DPI → rescale → limit size → binarize

    Returns a dict with intermediate results for inspection.
    """
    gray = to_grayscale(image)

    if known_dpi is not None:
        source_dpi = known_dpi
    else:
        source_dpi = estimate_dpi_from_content(gray)

    rescaled = rescale_to_target_dpi(gray, source_dpi, target_dpi)
    limited = limit_dimension(rescaled, MAX_DIMENSION)
    binary = binarize(limited, method=bin_method)

    # Diagnostic measurements on the binarized image
    stroke_w = estimate_stroke_width(limited)
    line_h = estimate_line_height(limited)

    return {
        "original": image,
        "grayscale": gray,
        "source_dpi": source_dpi,
        "target_dpi": target_dpi,
        "scale_factor": target_dpi / source_dpi,
        "rescaled": rescaled,
        "limited": limited,
        "binary": binary,
        "bin_method": bin_method,
        "normalized": binary,
        "stroke_width_px": stroke_w,
        "line_height_px": line_h,
    }


# ─────────────────────────────────────────────────────────────────────
# 7. Line boundary detection via horizontal projection
# ─────────────────────────────────────────────────────────────────────


def compute_projection(binary: np.ndarray) -> np.ndarray:
    """Compute horizontal projection (ink pixels per row).

    Expects a binary image where ink=255, background=0.
    """
    return binary.sum(axis=1).astype(np.float64)


def smooth_projection(projection: np.ndarray, image_height: int) -> np.ndarray:
    """Smooth projection with adaptive kernel (~1% of image height)."""
    kernel_size = max(3, int(image_height * 0.01) | 1)
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(projection, kernel, mode="same")


def find_line_boundaries(
    binary: np.ndarray,
    valley_thresh_ratio: float = 0.15,
    min_line_height_ratio: float = 0.3,
) -> list[tuple[int, int]]:
    """Detect line boundaries using horizontal projection valleys.

    Args:
        binary: Binary image (ink=255, background=0).
        valley_thresh_ratio: Projection values below max*ratio are valleys.
        min_line_height_ratio: Reject lines shorter than ratio * median_height.

    Returns:
        List of (y_start, y_end) tuples, one per detected text line.
    """
    projection = compute_projection(binary)
    smoothed = smooth_projection(projection, binary.shape[0])

    # Threshold: rows with ink below this are considered "valley" (gap)
    threshold = smoothed.max() * valley_thresh_ratio

    is_ink = smoothed > threshold

    # Find transitions: 0→1 = line start, 1→0 = line end
    padded = np.concatenate([[False], is_ink, [False]])
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    if len(starts) == 0:
        return []

    # Build raw boundaries
    boundaries = list(zip(starts.tolist(), ends.tolist()))

    # Filter out noise: reject lines much shorter than median
    if len(boundaries) >= 2:
        heights = [e - s for s, e in boundaries]
        median_h = np.median(heights)
        min_h = median_h * min_line_height_ratio
        boundaries = [(s, e) for s, e in boundaries if (e - s) >= min_h]

    return boundaries


def expand_boundaries_to_fill(
    boundaries: list[tuple[int, int]],
    image_height: int,
) -> list[tuple[int, int]]:
    """Expand boundaries so adjacent lines share the midpoint of each gap.

    This avoids leaving unassigned pixels between lines — useful for
    downstream cropping where every pixel should belong to a line.
    """
    if not boundaries:
        return []

    expanded = []
    for i, (start, end) in enumerate(boundaries):
        if i == 0:
            new_start = 0
        else:
            prev_end = boundaries[i - 1][1]
            new_start = (prev_end + start) // 2

        if i == len(boundaries) - 1:
            new_end = image_height
        else:
            next_start = boundaries[i + 1][0]
            new_end = (end + next_start) // 2

        expanded.append((new_start, new_end))

    return expanded


# ─────────────────────────────────────────────────────────────────────
# 8. Auto-threshold estimation
# ─────────────────────────────────────────────────────────────────────


def auto_thresh_stability(
    smoothed: np.ndarray,
    min_line_height_ratio: float = 0.3,
    sweep_range: tuple[float, float] = (0.01, 0.50),
    n_steps: int = 50,
) -> float:
    """Find optimal threshold via stability analysis.

    Sweep threshold and find the longest plateau where line count is constant.
    Return the midpoint of that plateau.
    """
    thresholds = np.linspace(sweep_range[0], sweep_range[1], n_steps)
    line_counts = []

    for t in thresholds:
        threshold_val = smoothed.max() * t
        is_ink = smoothed > threshold_val
        padded = np.concatenate([[False], is_ink, [False]])
        diffs = np.diff(padded.astype(np.int8))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        boundaries = list(zip(starts.tolist(), ends.tolist())) if len(starts) > 0 else []
        if len(boundaries) >= 2:
            heights = [e - s for s, e in boundaries]
            median_h = np.median(heights)
            min_h = median_h * min_line_height_ratio
            boundaries = [(s, e) for s, e in boundaries if (e - s) >= min_h]

        line_counts.append(len(boundaries))

    # Find longest plateau
    counts = np.array(line_counts)
    best_start = 0
    best_len = 1
    cur_start = 0

    for i in range(1, len(counts)):
        if counts[i] == counts[cur_start]:
            run_len = i - cur_start + 1
            if run_len > best_len:
                best_len = run_len
                best_start = cur_start
        else:
            cur_start = i

    best_mid = best_start + best_len // 2
    return float(thresholds[best_mid])


def auto_thresh_otsu(smoothed: np.ndarray) -> float:
    """Find threshold using Otsu on projection values.

    Treats projection as a 1D signal and separates "text rows" from "gap rows".
    """
    if smoothed.max() == 0:
        return 0.15
    proj_normalized = (smoothed / smoothed.max() * 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(proj_normalized, 0, 255, cv2.THRESH_OTSU)
    return float(otsu_val / 255.0)


def auto_thresh_prominence(
    smoothed: np.ndarray,
    min_line_height_ratio: float = 0.3,
) -> float:
    """Find threshold using valley prominence analysis.

    Uses scipy.signal.find_peaks to detect valleys. The threshold is set
    just above the deepest valleys that separate real lines.
    """
    from scipy.signal import find_peaks

    # Find peaks (line centers) in projection
    peaks, properties = find_peaks(smoothed, distance=10, prominence=smoothed.max() * 0.05)

    if len(peaks) < 2:
        return 0.15

    # Find valleys between peaks
    valleys = []
    for i in range(len(peaks) - 1):
        segment = smoothed[peaks[i] : peaks[i + 1]]
        if len(segment) > 0:
            valleys.append(segment.min())

    if not valleys:
        return 0.15

    # Threshold = median valley depth relative to max (with small margin above)
    median_valley = np.median(valleys)
    # Set threshold slightly above median valley to ensure valleys are detected
    thresh_ratio = (median_valley / smoothed.max()) * 1.2
    return float(np.clip(thresh_ratio, 0.01, 0.50))


# ─────────────────────────────────────────────────────────────────────
# 9. Non-linear boundary refinement via seam carving
# ─────────────────────────────────────────────────────────────────────


def compute_energy_map(binary: np.ndarray) -> np.ndarray:
    """Compute energy map using distance transform on background.

    High energy = near ink (seam should avoid).
    Low energy = far from ink (seam prefers to pass through).
    We invert so that DP minimization finds paths far from text.
    """
    background = (binary == 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(background, cv2.DIST_L2, 5)
    max_dist = dist.max()
    if max_dist == 0:
        return np.ones_like(binary, dtype=np.float64)
    # Invert: pixels far from ink get low energy
    energy = (max_dist - dist).astype(np.float64)
    return energy


def find_seam_in_strip(energy_strip: np.ndarray) -> np.ndarray:
    """Find minimum-energy horizontal seam (left→right) using DP.

    Args:
        energy_strip: 2D energy array (h x w), lower = preferred path.

    Returns:
        Array of row indices (length = width), one per column.
    """
    h, w = energy_strip.shape
    if h == 0 or w == 0:
        return np.zeros(w, dtype=np.int32)

    # Forward pass: accumulate minimum energy left→right
    dp = np.full((h, w), np.inf, dtype=np.float64)
    dp[:, 0] = energy_strip[:, 0]

    for x in range(1, w):
        # Same row
        dp[:, x] = dp[:, x - 1] + energy_strip[:, x]
        # Row above (shift down)
        dp[1:, x] = np.minimum(dp[1:, x], dp[:-1, x - 1] + energy_strip[1:, x])
        # Row below (shift up)
        dp[:-1, x] = np.minimum(dp[:-1, x], dp[1:, x - 1] + energy_strip[:-1, x])

    # Backward pass: trace minimum path right→left
    seam = np.zeros(w, dtype=np.int32)
    seam[w - 1] = np.argmin(dp[:, w - 1])

    for x in range(w - 2, -1, -1):
        row = seam[x + 1]
        r_min = max(0, row - 1)
        r_max = min(h - 1, row + 1)
        seam[x] = r_min + np.argmin(dp[r_min : r_max + 1, x])

    return seam


def compute_seam_boundaries(
    binary: np.ndarray,
    boundaries: list[tuple[int, int]],
    margin: int = 5,
) -> list[np.ndarray]:
    """Compute non-linear seam between each pair of consecutive lines.

    Args:
        binary: Binary image (ink=255, bg=0).
        boundaries: Linear line boundaries [(y_start, y_end), ...].
        margin: Extra pixels above/below the gap to include in strip.

    Returns:
        List of seam arrays (length = n_boundaries - 1).
        Each seam[i] has shape (width,) with absolute y-coordinates.
    """
    if len(boundaries) < 2:
        return []

    energy = compute_energy_map(binary)
    img_h, img_w = binary.shape
    seams = []

    for i in range(len(boundaries) - 1):
        gap_top = max(0, boundaries[i][1] - margin)
        gap_bot = min(img_h, boundaries[i + 1][0] + margin)

        if gap_bot <= gap_top:
            # No gap — fallback to midpoint
            mid = (boundaries[i][1] + boundaries[i + 1][0]) // 2
            seams.append(np.full(img_w, mid, dtype=np.int32))
            continue

        strip = energy[gap_top:gap_bot, :]
        seam_local = find_seam_in_strip(strip)
        seam_global = seam_local + gap_top
        seams.append(seam_global)

    return seams


# ─────────────────────────────────────────────────────────────────────
# 10. Interactive visualization with parameter sliders
# ─────────────────────────────────────────────────────────────────────

from matplotlib.widgets import CheckButtons, RadioButtons, Slider


def show_line_segmentation_interactive(result: dict, title: str = "") -> None:
    """Interactive figure with sliders, auto-threshold, and seam carving toggle.

    Sliders:
        - valley_thresh: ratio of max projection to define valleys (0.01–0.5)
        - min_line_h: reject lines shorter than ratio * median height (0.1–1.0)
        - smooth_k: smoothing kernel as % of image height (0.5–5.0)

    Toggles:
        - Threshold method: Manual / Stability / Otsu / Prominence
        - Seam Carving: on/off non-linear boundary refinement
    """
    binary = result["binary"]
    gray = result["limited"]
    img_h, img_w = binary.shape

    projection_raw = compute_projection(binary)
    rows = np.arange(img_h)

    # State
    state = {"use_seam": False, "thresh_mode": "Manual", "_updating": False}

    # Initial parameters
    init_thresh = 0.15
    init_min_h = 0.3
    init_smooth = 1.0

    # Layout: 2x2 plot panels + controls
    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        f"Line Segmentation: {title} (binarized with {result['bin_method']})"
        if title else f"Line Segmentation (binarized with {result['bin_method']})",
        fontsize=13,
    )

    gs = fig.add_gridspec(4, 2, height_ratios=[4, 4, 0.7, 1.2], hspace=0.35)
    ax_gray = fig.add_subplot(gs[0, 0])
    ax_bin = fig.add_subplot(gs[0, 1])
    ax_proj = fig.add_subplot(gs[1, 0])
    ax_lines = fig.add_subplot(gs[1, 1])

    # Slider axes (row 2)
    gs_sliders = gs[2, :].subgridspec(1, 3, wspace=0.3)
    ax_s_thresh = fig.add_subplot(gs_sliders[0, 0])
    ax_s_minh = fig.add_subplot(gs_sliders[0, 1])
    ax_s_smooth = fig.add_subplot(gs_sliders[0, 2])

    # Controls row (row 3): radio for threshold method + checkbox for seam
    gs_controls = gs[3, :].subgridspec(1, 2, wspace=0.3)
    ax_radio = fig.add_subplot(gs_controls[0, 0])
    ax_toggle = fig.add_subplot(gs_controls[0, 1])

    slider_thresh = Slider(
        ax_s_thresh, "valley_thresh", 0.01, 0.50,
        valinit=init_thresh, valstep=0.01, color="green",
    )
    slider_minh = Slider(
        ax_s_minh, "min_line_h", 0.1, 1.0,
        valinit=init_min_h, valstep=0.05, color="orange",
    )
    slider_smooth = Slider(
        ax_s_smooth, "smooth %", 0.5, 5.0,
        valinit=init_smooth, valstep=0.1, color="blue",
    )

    radio_thresh = RadioButtons(
        ax_radio,
        ["Manual", "Stability", "Otsu", "Prominence"],
        active=0,
    )
    ax_radio.set_title("Threshold Method", fontsize=10)

    check_seam = CheckButtons(ax_toggle, ["Seam Carving (non-linear)"], [False])

    # Static panel: grayscale (top-left)
    ax_gray.imshow(gray, cmap="gray", aspect="auto")
    h, w = gray.shape
    ax_gray.set_title(f"Grayscale @{result['target_dpi']:.0f}DPI ({w}×{h})")
    ax_gray.axis("off")

    def update(_=None):
        if state["_updating"]:
            return
        state["_updating"] = True

        try:
            _do_update()
        finally:
            state["_updating"] = False

    def _do_update():
        min_h_ratio = slider_minh.val
        smooth_pct = slider_smooth.val
        use_seam = state["use_seam"]
        thresh_mode = state["thresh_mode"]

        # Recompute smoothing
        kernel_size = max(3, int(img_h * smooth_pct / 100) | 1)
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(projection_raw, kernel, mode="same")

        # Determine threshold based on mode
        if thresh_mode == "Manual":
            thresh_ratio = slider_thresh.val
        elif thresh_mode == "Stability":
            thresh_ratio = auto_thresh_stability(smoothed, min_h_ratio)
        elif thresh_mode == "Otsu":
            thresh_ratio = auto_thresh_otsu(smoothed)
        elif thresh_mode == "Prominence":
            thresh_ratio = auto_thresh_prominence(smoothed, min_h_ratio)
        else:
            thresh_ratio = slider_thresh.val

        # Update slider position to reflect auto value (visual feedback)
        if thresh_mode != "Manual":
            slider_thresh.set_val(np.clip(thresh_ratio, 0.01, 0.50))

        # Recompute boundaries (linear)
        threshold = smoothed.max() * thresh_ratio
        is_ink = smoothed > threshold

        padded = np.concatenate([[False], is_ink, [False]])
        diffs = np.diff(padded.astype(np.int8))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        boundaries = list(zip(starts.tolist(), ends.tolist())) if len(starts) > 0 else []

        if len(boundaries) >= 2:
            heights = [e - s for s, e in boundaries]
            median_h = np.median(heights)
            min_h = median_h * min_h_ratio
            boundaries = [(s, e) for s, e in boundaries if (e - s) >= min_h]

        # Compute seams if enabled
        seams = []
        if use_seam and len(boundaries) >= 2:
            seams = compute_seam_boundaries(binary, boundaries)

        expanded = expand_boundaries_to_fill(boundaries, img_h)

        # ── Panel: Binary + boundaries (top-right) ──
        ax_bin.clear()
        ax_bin.imshow(binary, cmap="gray", aspect="auto")

        if use_seam and seams:
            xs = np.arange(img_w)
            for seam in seams:
                ax_bin.plot(xs, seam, color="yellow", linewidth=1.0, alpha=0.9)
            for y_start, y_end in boundaries:
                ax_bin.axhline(y=y_start, color="red", linewidth=0.4, linestyle=":", alpha=0.4)
                ax_bin.axhline(y=y_end, color="red", linewidth=0.4, linestyle=":", alpha=0.4)
            ax_bin.set_title(f"{len(boundaries)} lines — SEAM [{thresh_mode}: {thresh_ratio:.2f}]")
        else:
            for y_start, y_end in boundaries:
                ax_bin.axhline(y=y_start, color="red", linewidth=0.8, linestyle="--")
                ax_bin.axhline(y=y_end, color="red", linewidth=0.8, linestyle="--")
            for y_start, _ in expanded:
                ax_bin.axhline(y=y_start, color="cyan", linewidth=0.5, alpha=0.7)
            ax_bin.set_title(f"{len(boundaries)} lines — LINEAR [{thresh_mode}: {thresh_ratio:.2f}]")
        ax_bin.axis("off")

        # ── Panel: Projection (bottom-left) ──
        ax_proj.clear()
        ax_proj.plot(projection_raw, rows, color="gray", alpha=0.3, linewidth=0.5)
        ax_proj.plot(smoothed, rows, color="blue", linewidth=1.0)
        ax_proj.axvline(x=threshold, color="green", linestyle=":", linewidth=1.5)

        for y_start, y_end in boundaries:
            ax_proj.axhspan(y_start, y_end, alpha=0.1, color="red")
            mid = (y_start + y_end) // 2
            ax_proj.annotate(
                f"{y_end - y_start}px",
                xy=(smoothed.max() * 0.7, mid),
                fontsize=7, color="red", va="center",
            )

        ax_proj.invert_yaxis()
        mode_label = "SEAM" if use_seam else "LINEAR"
        ax_proj.set_title(
            f"Projection (k={kernel_size}px, thresh={thresh_ratio:.2f}={threshold:.0f}) "
            f"[{mode_label}|{thresh_mode}]"
        )
        ax_proj.set_xlabel("Ink pixel count")
        ax_proj.set_ylabel("Row")

        # ── Panel: Extracted lines (bottom-right) ──
        ax_lines.clear()
        if use_seam and seams and len(boundaries) >= 2:
            line_imgs = _extract_lines_with_seams(binary, boundaries, seams)
            if line_imgs:
                gap = 4
                total_h = sum(im.shape[0] for im in line_imgs) + gap * (len(line_imgs) - 1)
                max_w = max(im.shape[1] for im in line_imgs)
                canvas = np.zeros((total_h, max_w), dtype=np.uint8)
                y_offset = 0
                for im in line_imgs:
                    lh, lw = im.shape
                    canvas[y_offset : y_offset + lh, :lw] = im
                    y_offset += lh + gap
                ax_lines.imshow(canvas, cmap="gray", aspect="auto")
                ax_lines.set_title(f"{len(line_imgs)} lines (seam-cut)")
            else:
                ax_lines.text(0.5, 0.5, "No lines", ha="center", va="center", transform=ax_lines.transAxes)
                ax_lines.set_title("Extracted lines")
        elif expanded:
            line_imgs = [binary[s:e, :] for s, e in expanded]
            gap = 4
            total_h = sum(im.shape[0] for im in line_imgs) + gap * (len(line_imgs) - 1)
            max_w = max(im.shape[1] for im in line_imgs)
            canvas = np.zeros((total_h, max_w), dtype=np.uint8)
            y_offset = 0
            for im in line_imgs:
                lh, lw = im.shape
                canvas[y_offset : y_offset + lh, :lw] = im
                y_offset += lh + gap
            ax_lines.imshow(canvas, cmap="gray", aspect="auto")
            ax_lines.set_title(f"{len(line_imgs)} lines (linear-cut)")
        else:
            ax_lines.text(0.5, 0.5, "No lines", ha="center", oyva="center", transform=ax_lines.transAxes)
            ax_lines.set_title("Extracted lines")
        ax_lines.axis("off")

        fig.canvas.draw_idle()

    def on_thresh_mode(label):
        state["thresh_mode"] = label
        update()

    def toggle_seam(_label):
        state["use_seam"] = not state["use_seam"]
        update()

    radio_thresh.on_clicked(on_thresh_mode)
    check_seam.on_clicked(toggle_seam)
    slider_thresh.on_changed(update)
    slider_minh.on_changed(update)
    slider_smooth.on_changed(update)

    update()
    plt.show()


def _extract_lines_with_seams(
    binary: np.ndarray,
    boundaries: list[tuple[int, int]],
    seams: list[np.ndarray],
) -> list[np.ndarray]:
    """Extract line images using seam curves as non-linear boundaries.

    Each line is bounded:
        - Top: seam[i-1] (or image top for first line)
        - Bottom: seam[i] (or image bottom for last line)

    Pixels outside the seam boundary are zeroed out per column.
    """
    img_h, img_w = binary.shape
    n_lines = len(boundaries)
    line_imgs = []

    for i in range(n_lines):
        # Determine top/bottom seam for this line
        if i == 0:
            top_seam = np.zeros(img_w, dtype=np.int32)
        else:
            top_seam = seams[i - 1]

        if i == n_lines - 1:
            bot_seam = np.full(img_w, img_h - 1, dtype=np.int32)
        else:
            bot_seam = seams[i]

        # Bounding box
        y_min = int(top_seam.min())
        y_max = int(bot_seam.max()) + 1
        y_max = min(y_max, img_h)

        strip = binary[y_min:y_max, :].copy()

        # Zero out pixels above top_seam and below bot_seam per column
        for x in range(img_w):
            local_top = top_seam[x] - y_min
            local_bot = bot_seam[x] - y_min
            if local_top > 0:
                strip[:local_top, x] = 0
            if local_bot < strip.shape[0]:
                strip[local_bot:, x] = 0

        line_imgs.append(strip)

    return line_imgs


def show_comparison_grid(results: list[dict]) -> None:
    """Compare multiple images side by side after normalization."""
    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, res in enumerate(results):
        ax = axes[i, 0]
        orig = res["original"]
        if orig.ndim == 3:
            ax.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(orig, cmap="gray")
        h, w = orig.shape[:2]
        ax.set_title(f"Original ({w}×{h})")
        ax.axis("off")

        ax = axes[i, 1]
        normed = res["normalized"]
        nh, nw = normed.shape[:2]
        ax.imshow(normed, cmap="gray")
        tdpi = res.get("target_dpi", TARGET_DPI)
        ax.set_title(f"@{tdpi:.0f}DPI ({nw}×{nh})")
        ax.axis("off")

        ax = axes[i, 2]
        proj = compute_projection(res["binary"])
        ax.plot(proj, np.arange(len(proj)))
        ax.invert_yaxis()
        ax.set_title(f"DPI={res['source_dpi']:.0f}→{tdpi:.0f} ({res['bin_method']})")
        ax.set_xlabel("Ink px")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────
# 8. CLI entry point
# ─────────────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Image normalization for line segmentation")
    parser.add_argument("input", help="Image file or directory of images")
    parser.add_argument("--dpi", type=float, default=None, help="Known source DPI (skip estimation)")
    parser.add_argument("--target-dpi", type=float, default=TARGET_DPI, help="Target DPI")
    parser.add_argument(
        "--bin", dest="bin_method", default="sauvola",
        choices=["otsu", "sauvola", "adaptive"],
        help="Binarization method (default: sauvola)",
    )
    args = parser.parse_args()

    target_dpi = args.target_dpi
    bin_method = args.bin_method

    input_path = Path(args.input)

    if input_path.is_file():
        image = load_image(input_path)
        result = normalize_for_segmentation(image, known_dpi=args.dpi, target_dpi=target_dpi, bin_method=bin_method)

        print(f"Source: {input_path.name}")
        print(f"  Original size  : {image.shape[1]}×{image.shape[0]}")
        print(f"  Estimated DPI  : {result['source_dpi']:.1f}")
        print(f"  Scale factor   : {result['scale_factor']:.3f}")
        print(f"  Binary size    : {result['binary'].shape[1]}×{result['binary'].shape[0]}")
        print(f"  Binarization   : {bin_method}")
        print(f"  Stroke width   : {result['stroke_width_px']:.1f} px")
        print(f"  Line height    : {result['line_height_px']:.1f} px" if result["line_height_px"] else "  Line height    : N/A")

        show_line_segmentation_interactive(result, title=input_path.name)

    elif input_path.is_dir():
        extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
        files = sorted(p for p in input_path.iterdir() if p.suffix.lower() in extensions)
        if not files:
            print(f"No images found in {input_path}")
            return

        print(f"Processing {len(files)} images from {input_path}\n")
        results = []
        for f in files:
            image = load_image(f)
            res = normalize_for_segmentation(image, known_dpi=args.dpi, target_dpi=target_dpi, bin_method=bin_method)
            results.append(res)
            print(
                f"  {f.name:30s} | {image.shape[1]:5d}×{image.shape[0]:<5d} "
                f"→ {res['normalized'].shape[1]:5d}×{res['normalized'].shape[0]:<5d} "
                f"| DPI≈{res['source_dpi']:.0f} | stroke={res['stroke_width_px']:.1f}px"
            )

        show_comparison_grid(results)
    else:
        print(f"Error: {input_path} not found")


if __name__ == "__main__":
    main()
