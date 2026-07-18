"""Tiền xử lí ảnh chụp giấy chữ viết tay — document-scanner có UI chỉnh tham số.

Pipeline (thứ tự chuẩn: TÁCH TRANG TRƯỚC, binarize SAU):

    ảnh gốc
      → resize để xử lí nhanh
      → PHÁT HIỆN 4 GÓC TRANG (Canny+contour theo kênh gray/value/sat; fallback GrabCut)
      → NỚI GÓC quad (chống cắt hụt mép) → nắn phối cảnh (perspective warp)
      → CHUẨN HÓA ÁNH SÁNG (flat-field) → nhị phân hóa (Sauvola/adaptive/otsu)
      → KHỬ DÒNG KẺ NGANG (morphology)
      → deskew tinh chỉnh (góc dòng chữ) → crop sát nội dung

Mọi tham số con-người-cần-chỉnh đều nằm trên slider / radio / checkbox của matplotlib.

Cách dùng:
    python demo/main.py path/to/image.jpg            # mở UI tương tác
    python demo/main.py path/to/image.jpg --save out.png   # render tĩnh với mặc định
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons, RadioButtons, Slider
from skimage.filters import threshold_sauvola

WORK_DIM = 1080


# ─────────────────────────────────────────────────────────────────────
# Tham số (giá trị mặc định = giá trị khởi tạo cho các widget)
# ─────────────────────────────────────────────────────────────────────


@dataclass
class Params:
    channel: str = "auto"          # auto | gray | value | sat | grabcut
    canny_low: int = 50
    canny_high: int = 150
    close_ksize: int = 7           # kernel CLOSE để xóa chữ khi dò biên trang
    min_area: float = 0.20         # trang phải chiếm >= tỉ lệ này
    expand_px: int = 8             # nới 4 góc ra ngoài (chống cắt hụt mép)
    illum_on: bool = True
    illum_bg: int = 41             # kernel ước lượng nền (flat-field)
    bin_method: str = "sauvola"    # sauvola | adaptive | otsu
    sauvola_window: int = 25
    sauvola_k: float = 0.20
    ruled_on: bool = True
    ruled_len: int = 40            # độ dài kernel ngang để bắt dòng kẻ
    deskew_on: bool = True


# ─────────────────────────────────────────────────────────────────────
# Load & resize
# ─────────────────────────────────────────────────────────────────────


def load_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    return img


def resize_work(image: np.ndarray, work_dim: int = WORK_DIM) -> np.ndarray:
    h, w = image.shape[:2]
    if max(h, w) <= work_dim:
        return image.copy()
    scale = work_dim / max(h, w)
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


# ─────────────────────────────────────────────────────────────────────
# Phát hiện 4 góc trang
# ─────────────────────────────────────────────────────────────────────


def order_points(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def _approx_quad(contour: np.ndarray) -> np.ndarray | None:
    peri = cv2.arcLength(contour, True)
    for eps in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.04, 0.06, 0.08, 0.10, 0.12):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)


def _largest_quad(binary: np.ndarray, img_area: float, min_area: float) -> np.ndarray | None:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(cnt) < min_area * img_area:
            break
        quad = _approx_quad(cnt)
        if quad is not None and cv2.contourArea(quad.astype(np.int32)) >= min_area * img_area:
            return quad
    return None


def _edges_from_channel(channel: np.ndarray, p: Params) -> np.ndarray:
    k = max(1, p.close_ksize)
    closed = cv2.morphologyEx(channel, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)), iterations=3)
    blur = cv2.GaussianBlur(closed, (5, 5), 0)
    edges = cv2.Canny(blur, p.canny_low, p.canny_high)
    return cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)


def _grabcut_mask(image_bgr: np.ndarray) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    m = int(min(h, w) * 0.05)
    cv2.grabCut(image_bgr, mask, (m, m, w - 2 * m, h - 2 * m), bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))


def detect_page_quad(image_bgr: np.ndarray, p: Params) -> tuple[np.ndarray | None, str, np.ndarray]:
    """Trả về (quad|None, tên_phương_pháp, ảnh_trung_gian)."""
    img_area = image_bgr.shape[0] * image_bgr.shape[1]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    channels = {"gray": gray, "value": hsv[:, :, 2], "sat": hsv[:, :, 1]}

    if p.channel == "grabcut":
        mask = _grabcut_mask(image_bgr)
        quad = _largest_quad(mask, img_area, p.min_area)
        return (order_points(quad) if quad is not None else None), "grabcut", mask

    if p.channel in channels:
        edges = _edges_from_channel(channels[p.channel], p)
        quad = _largest_quad(edges, img_area, p.min_area)
        return (order_points(quad) if quad is not None else None), f"edge-{p.channel}", edges

    # auto: thử lần lượt gray → value → sat → grabcut
    for name, ch in channels.items():
        edges = _edges_from_channel(ch, p)
        quad = _largest_quad(edges, img_area, p.min_area)
        if quad is not None:
            return order_points(quad), f"edge-{name}", edges
    mask = _grabcut_mask(image_bgr)
    quad = _largest_quad(mask, img_area, p.min_area)
    if quad is not None:
        return order_points(quad), "grabcut", mask
    return None, "none", gray


# ─────────────────────────────────────────────────────────────────────
# CẢI TIẾN 1: nới góc quad để không cắt hụt mép
# ─────────────────────────────────────────────────────────────────────


def expand_quad(quad: np.ndarray, px: float, shape: tuple[int, int]) -> np.ndarray:
    """Đẩy 4 góc ra xa tâm ``px`` pixel (dọc hướng góc↔tâm), kẹp trong ảnh."""
    if px <= 0:
        return quad
    center = quad.mean(axis=0)
    out = quad.copy()
    for i, pt in enumerate(quad):
        v = pt - center
        n = np.linalg.norm(v)
        if n > 1e-6:
            out[i] = pt + v / n * px
    h, w = shape
    out[:, 0] = np.clip(out[:, 0], 0, w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, h - 1)
    return out.astype(np.float32)


def warp_perspective(image_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = quad
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    width, height = max(width, 1), max(height, 1)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(image_bgr, matrix, (width, height))


# ─────────────────────────────────────────────────────────────────────
# CẢI TIẾN 2a: chuẩn hóa ánh sáng (flat-field)
# ─────────────────────────────────────────────────────────────────────


def normalize_illumination(gray: np.ndarray, bg_ksize: int) -> np.ndarray:
    """Ước lượng nền sáng bằng CLOSE (xóa chữ tối) rồi chia để triệt bóng/ánh màu."""
    k = max(3, bg_ksize | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    norm = gray.astype(np.float32) / (background.astype(np.float32) + 1e-6)
    return np.clip(norm * 255.0, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────
# Nhị phân hóa
# ─────────────────────────────────────────────────────────────────────


def binarize(gray: np.ndarray, p: Params) -> np.ndarray:
    if p.bin_method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary
    if p.bin_method == "adaptive":
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    window = max(3, p.sauvola_window | 1)
    thresh_map = threshold_sauvola(gray, window_size=window, k=p.sauvola_k)
    return ((gray < thresh_map) * 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────
# CẢI TIẾN 2b: khử dòng kẻ ngang
# ─────────────────────────────────────────────────────────────────────


def remove_ruled_lines(binary: np.ndarray, hlen: int) -> np.ndarray:
    """Bắt các nét ngang dài bằng OPEN kernel ngang rồi trừ khỏi ảnh nhị phân."""
    if hlen <= 1:
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hlen, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return cv2.subtract(binary, horizontal)


# ─────────────────────────────────────────────────────────────────────
# Deskew tinh chỉnh & crop
# ─────────────────────────────────────────────────────────────────────


def _projection_score(binary: np.ndarray, angle: float) -> float:
    h, w = binary.shape
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(binary, matrix, (w, h), flags=cv2.INTER_NEAREST)
    return float(np.sum(np.diff(rotated.sum(axis=1, dtype=np.float64)) ** 2))


def estimate_skew_angle(binary: np.ndarray, rng: float = 8.0, step: float = 0.2) -> float:
    work = binary
    if binary.shape[0] > 600:
        s = 600 / binary.shape[0]
        work = cv2.resize(binary, (max(1, round(binary.shape[1] * s)), 600), interpolation=cv2.INTER_NEAREST)
    angles = np.arange(-rng, rng + step, step)
    scores = [_projection_score(work, a) for a in angles]
    return float(angles[int(np.argmax(scores))])


def rotate_image(image: np.ndarray, angle: float, border_value: int = 0) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    matrix[0, 2] += (new_w - w) / 2
    matrix[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


def crop_to_content(binary: np.ndarray, *others: np.ndarray, margin: int = 15):
    ys, xs = np.where(binary > 0)
    if len(ys) == 0:
        return (binary, *others)
    h, w = binary.shape
    y0, y1 = max(0, ys.min() - margin), min(h, ys.max() + margin + 1)
    x0, x1 = max(0, xs.min() - margin), min(w, xs.max() + margin + 1)
    return tuple(img[y0:y1, x0:x1] for img in (binary, *others))


# ─────────────────────────────────────────────────────────────────────
# Pipeline (chạy trên ảnh làm việc để UI phản hồi nhanh)
# ─────────────────────────────────────────────────────────────────────


def run_pipeline(small_bgr: np.ndarray, p: Params) -> dict:
    quad, method, detail = detect_page_quad(small_bgr, p)
    if quad is not None:
        quad_exp = expand_quad(quad, p.expand_px, small_bgr.shape[:2])
        warped = warp_perspective(small_bgr, quad_exp)
    else:
        quad_exp = None
        warped = small_bgr.copy()

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    illum = normalize_illumination(gray, p.illum_bg) if p.illum_on else gray

    binary = binarize(illum, p)
    binary = remove_ruled_lines(binary, p.ruled_len) if p.ruled_on else binary

    if p.deskew_on:
        angle = estimate_skew_angle(binary)
        gray_desk = rotate_image(illum, angle, border_value=255)
        binary_desk = (rotate_image(binary, angle, border_value=0) > 127).astype(np.uint8) * 255
    else:
        angle = 0.0
        gray_desk, binary_desk = illum, binary

    final_bin, final_gray = crop_to_content(binary_desk, gray_desk)

    return {
        "small": small_bgr, "quad": quad_exp, "method": method, "detail": detail,
        "warped": warped, "illum": illum, "binary": binary,
        "angle": angle, "final_bin": final_bin, "final_gray": final_gray,
    }


# ─────────────────────────────────────────────────────────────────────
# UI matplotlib tương tác
# ─────────────────────────────────────────────────────────────────────


def _draw_panels(axes, res: dict, p: Params) -> None:
    for ax in axes.ravel():
        ax.clear()
        ax.axis("off")

    def show(ax, img, title, gray=True):
        ax.imshow(img if gray else cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap="gray" if gray else None)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    vis = res["small"].copy()
    if res["quad"] is not None:
        cv2.polylines(vis, [res["quad"].astype(np.int32)], True, (0, 0, 255), 3)
        for pt in res["quad"].astype(np.int32):
            cv2.circle(vis, tuple(pt), 7, (0, 255, 0), -1)
        show(axes[0, 0], vis, f"1. Phát hiện trang [{res['method']}] + nới {p.expand_px}px", gray=False)
    else:
        show(axes[0, 0], vis, "1. KHÔNG thấy trang → dùng cả ảnh", gray=False)

    show(axes[0, 1], res["detail"], f"2. Trung gian dò trang [{res['method']}]")
    wh, ww = res["warped"].shape[:2]
    show(axes[0, 2], res["warped"], f"3. Nắn phối cảnh ({ww}×{wh})", gray=False)
    show(axes[1, 0], res["illum"], "4. Xám" + (" + chuẩn hóa sáng" if p.illum_on else ""))
    ruled = f" − kẻ({p.ruled_len})" if p.ruled_on else ""
    show(axes[1, 1], res["binary"], f"5. Nhị phân [{p.bin_method}]{ruled}")
    show(axes[1, 2], res["final_bin"], f"6. Deskew {res['angle']:+.2f}° + crop")


def show_interactive(small_bgr: np.ndarray, p: Params, title: str = "") -> None:
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"Tiền xử lí (document-scanner) — {title}", fontsize=13)

    gs = fig.add_gridspec(2, 3, left=0.26, right=0.99, top=0.93, bottom=0.05, hspace=0.14, wspace=0.08)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])

    state = {"busy": False}

    # ── Sliders (cột trái) ──
    def add_slider(y, label, lo, hi, val, step, color):
        ax = fig.add_axes([0.055, y, 0.16, 0.022])
        return Slider(ax, label, lo, hi, valinit=val, valstep=step, color=color)

    s_cl = add_slider(0.94, "canny_low", 0, 200, p.canny_low, 1, "tab:green")
    s_ch = add_slider(0.90, "canny_high", 50, 400, p.canny_high, 1, "tab:green")
    s_ck = add_slider(0.86, "close_ksize", 1, 25, p.close_ksize, 1, "tab:green")
    s_ma = add_slider(0.82, "min_area", 0.05, 0.6, p.min_area, 0.01, "tab:green")
    s_ex = add_slider(0.78, "expand_px", 0, 60, p.expand_px, 1, "tab:purple")
    s_ib = add_slider(0.74, "illum_bg", 5, 121, p.illum_bg, 2, "tab:orange")
    s_sw = add_slider(0.70, "sauvola_win", 3, 61, p.sauvola_window, 2, "tab:blue")
    s_sk = add_slider(0.66, "sauvola_k", 0.05, 0.5, p.sauvola_k, 0.01, "tab:blue")
    s_rl = add_slider(0.62, "ruled_len", 1, 120, p.ruled_len, 1, "tab:red")

    # ── Radio: kênh phát hiện trang ──
    ax_ch = fig.add_axes([0.03, 0.40, 0.09, 0.17])
    ax_ch.set_title("detect channel", fontsize=9)
    r_channel = RadioButtons(ax_ch, ["auto", "gray", "value", "sat", "grabcut"],
                             active=["auto", "gray", "value", "sat", "grabcut"].index(p.channel))

    # ── Radio: phương pháp nhị phân ──
    ax_bm = fig.add_axes([0.13, 0.44, 0.09, 0.13])
    ax_bm.set_title("binarize", fontsize=9)
    r_bin = RadioButtons(ax_bm, ["sauvola", "adaptive", "otsu"],
                         active=["sauvola", "adaptive", "otsu"].index(p.bin_method))

    # ── Checkboxes ──
    ax_ck = fig.add_axes([0.03, 0.24, 0.19, 0.12])
    checks = CheckButtons(ax_ck, ["illum norm", "remove ruled", "deskew"],
                          [p.illum_on, p.ruled_on, p.deskew_on])

    def read_params() -> Params:
        return Params(
            channel=r_channel.value_selected,
            canny_low=int(s_cl.val), canny_high=int(s_ch.val), close_ksize=int(s_ck.val),
            min_area=float(s_ma.val), expand_px=int(s_ex.val),
            illum_on=checks.get_status()[0], illum_bg=int(s_ib.val),
            bin_method=r_bin.value_selected,
            sauvola_window=int(s_sw.val), sauvola_k=float(s_sk.val),
            ruled_on=checks.get_status()[1], ruled_len=int(s_rl.val),
            deskew_on=checks.get_status()[2],
        )

    def update(_=None):
        if state["busy"]:
            return
        state["busy"] = True
        try:
            res = run_pipeline(small_bgr, read_params())
            _draw_panels(axes, res, read_params())
            fig.canvas.draw_idle()
        finally:
            state["busy"] = False

    for s in (s_cl, s_ch, s_ck, s_ma, s_ex, s_ib, s_sw, s_sk, s_rl):
        s.on_changed(update)
    r_channel.on_clicked(update)
    r_bin.on_clicked(update)
    checks.on_clicked(update)

    update()
    plt.show()


def render_static(small_bgr: np.ndarray, p: Params, title: str, save: str) -> None:
    res = run_pipeline(small_bgr, p)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"Tiền xử lí (document-scanner) — {title}", fontsize=13)
    _draw_panels(axes, res, p)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(save, dpi=120, bbox_inches="tight")
    print(f"Đã lưu figure: {save}")


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiền xử lí ảnh chụp giấy chữ viết tay (UI chỉnh tham số)")
    parser.add_argument("input", help="Đường dẫn ảnh")
    parser.add_argument("--save", default=None, help="Render tĩnh ra file (không mở UI)")
    args = parser.parse_args()

    original = load_image(args.input)
    small = resize_work(original)
    p = Params()
    name = Path(args.input).name

    if args.save:
        render_static(small, p, name, args.save)
    else:
        show_interactive(small, p, title=name)


if __name__ == "__main__":
    main()
