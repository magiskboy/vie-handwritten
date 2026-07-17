"""Main content: image preview + OCR prediction (+ optional GT compare).

Supports two modes:
    - Single-line: original flow (image + one prediction).
    - Paragraph: segmented lines display with seam overlay toggle.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GdkPixbuf, GLib, GObject, Gtk, Pango  # noqa: E402


class ContentView(Gtk.Box):
    """Picture (top) + prediction card (bottom), optional ground-truth compare."""

    __gsignals__ = {
        "rerun": (GObject.SignalFlags.RUN_FIRST, None, ()),
    }

    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.set_margin_start(16)
        self.set_margin_end(16)
        self.set_margin_top(12)
        self.set_margin_bottom(16)

        # --- image pane ---
        self._filename = Gtk.Label(xalign=0.0)
        self._filename.add_css_class("vie-filename")
        self._filename.set_ellipsize(Pango.EllipsizeMode.MIDDLE)
        self.append(self._filename)

        self._picture = Gtk.Picture()
        self._picture.set_can_shrink(True)
        self._picture.set_content_fit(Gtk.ContentFit.CONTAIN)
        self._picture.set_halign(Gtk.Align.CENTER)
        self._picture.set_valign(Gtk.Align.CENTER)
        self._picture.set_hexpand(True)
        self._picture.set_vexpand(True)

        self._placeholder = Gtk.Label(
            label="Load model và chọn ảnh để bắt đầu",
            wrap=True,
            justify=Gtk.Justification.CENTER,
            halign=Gtk.Align.CENTER,
            valign=Gtk.Align.CENTER,
        )
        self._placeholder.add_css_class("dim-label")

        self._image_stack = Gtk.Stack()
        self._image_stack.set_vexpand(True)
        self._image_stack.set_hexpand(True)
        self._image_stack.add_named(self._placeholder, "empty")
        self._image_stack.add_named(self._picture, "image")
        self._image_stack.set_visible_child_name("empty")

        image_pane = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        image_pane.add_css_class("vie-image-pane")
        image_pane.set_vexpand(True)
        image_pane.set_hexpand(True)
        image_pane.append(self._image_stack)
        self.append(image_pane)

        # --- result pane ---
        result = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        result.add_css_class("vie-result-pane")
        result.set_hexpand(True)

        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        title = Gtk.Label(label="Kết quả nhận dạng", xalign=0.0)
        title.add_css_class("vie-result-title")
        header.append(title)

        self._latency = Gtk.Label(label="", xalign=0.0, hexpand=True)
        self._latency.add_css_class("dim-label")
        self._latency.add_css_class("vie-latency")
        header.append(self._latency)

        self._spinner = Gtk.Spinner()
        header.append(self._spinner)

        self._copy_btn = Gtk.Button()
        self._copy_btn.set_icon_name("edit-copy-symbolic")
        self._copy_btn.set_tooltip_text("Copy prediction")
        self._copy_btn.add_css_class("flat")
        self._copy_btn.set_sensitive(False)
        self._copy_btn.connect("clicked", self._on_copy)
        header.append(self._copy_btn)

        self._rerun_btn = Gtk.Button()
        self._rerun_btn.set_icon_name("view-refresh-symbolic")
        self._rerun_btn.set_tooltip_text("Re-run")
        self._rerun_btn.add_css_class("flat")
        self._rerun_btn.set_sensitive(False)
        self._rerun_btn.connect("clicked", lambda *_: self.emit("rerun"))
        header.append(self._rerun_btn)
        result.append(header)

        # Prediction row
        pred_key = Gtk.Label(label="Pred", xalign=0.0, yalign=0.0)
        pred_key.add_css_class("vie-kv-key")
        self._text = Gtk.Label(
            label="Chọn ảnh để xem kết quả OCR.",
            xalign=0.0,
            wrap=True,
            wrap_mode=Pango.WrapMode.WORD_CHAR,
            selectable=True,
            hexpand=True,
        )
        self._text.add_css_class("vie-result-text")
        pred_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        pred_row.append(pred_key)
        pred_row.append(self._text)
        result.append(pred_row)

        # Ground-truth + metrics (hidden unless label.json present)
        self._compare = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self._compare.set_visible(False)

        gt_key = Gtk.Label(label="GT", xalign=0.0, yalign=0.0)
        gt_key.add_css_class("vie-kv-key")
        self._gt_text = Gtk.Label(
            label="",
            xalign=0.0,
            wrap=True,
            wrap_mode=Pango.WrapMode.WORD_CHAR,
            selectable=True,
            hexpand=True,
        )
        self._gt_text.add_css_class("vie-result-text")
        gt_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        gt_row.append(gt_key)
        gt_row.append(self._gt_text)
        self._compare.append(gt_row)

        self._metrics = Gtk.Label(label="", xalign=0.0, selectable=True)
        self._metrics.add_css_class("vie-metrics")
        self._metrics.add_css_class("dim-label")
        self._compare.append(self._metrics)

        result.append(self._compare)
        self.append(result)

        self._current_path: str | None = None
        self._prediction: str = ""
        self._ground_truth: str | None = None

    def show_image(self, path: str | Path, *, ground_truth: str | None = None) -> None:
        p = Path(path)
        self._current_path = str(p)
        self._ground_truth = ground_truth
        self._filename.set_label(p.name)
        self._filename.set_tooltip_text(str(p))
        self._picture.set_filename(str(p))
        self._image_stack.set_visible_child_name("image")
        self._rerun_btn.set_sensitive(True)
        if ground_truth is not None:
            self._gt_text.set_label(ground_truth)
            self._compare.set_visible(True)
            self._metrics.set_label("Chưa có prediction để so sánh.")
        else:
            self._compare.set_visible(False)
            self._gt_text.set_label("")
            self._metrics.set_label("")

    def set_prediction(
        self,
        text: str,
        *,
        elapsed_ms: float | None = None,
        comparison: dict[str, Any] | None = None,
    ) -> None:
        self._prediction = (text or "").strip()
        self._text.set_label(self._prediction if self._prediction else "(trống)")
        self._copy_btn.set_sensitive(bool(self._prediction))
        self.set_latency(elapsed_ms)
        self._apply_comparison(comparison)

    def set_message(self, text: str) -> None:
        self._prediction = ""
        self._text.set_label(text)
        self._copy_btn.set_sensitive(False)
        self.set_latency(None)
        if self._ground_truth is not None:
            self._compare.set_visible(True)
            self._metrics.set_label("Chưa có prediction để so sánh.")
        else:
            self._compare.set_visible(False)

    def set_latency(self, elapsed_ms: float | None) -> None:
        if elapsed_ms is None:
            self._latency.set_label("")
            return
        self._latency.set_label(f"{elapsed_ms:.0f} ms")

    def set_busy(self, busy: bool) -> None:
        if busy:
            self._spinner.start()
            self.set_latency(None)
        else:
            self._spinner.stop()
        self._rerun_btn.set_sensitive(not busy and self._current_path is not None)

    def clear(self) -> None:
        self._current_path = None
        self._prediction = ""
        self._ground_truth = None
        self._filename.set_label("")
        self._picture.set_paintable(None)
        self._image_stack.set_visible_child_name("empty")
        self._text.set_label("Chọn ảnh để xem kết quả OCR.")
        self._copy_btn.set_sensitive(False)
        self._rerun_btn.set_sensitive(False)
        self.set_latency(None)
        self._compare.set_visible(False)
        self._gt_text.set_label("")
        self._metrics.set_label("")

    def _apply_comparison(self, comparison: dict[str, Any] | None) -> None:
        if comparison is None:
            if self._ground_truth is not None:
                self._compare.set_visible(True)
                self._metrics.set_label("Chưa có prediction để so sánh.")
            else:
                self._compare.set_visible(False)
            return

        self._gt_text.set_label(str(comparison.get("reference") or ""))
        dist = int(comparison.get("levenshtein", 0))
        cer = float(comparison.get("cer", 0.0))
        wer = float(comparison.get("wer", 0.0))
        exact = bool(comparison.get("exact"))
        match = "exact" if exact else "diff"
        self._metrics.set_label(
            f"Levenshtein {dist}  ·  CER {cer:.1%}  ·  WER {wer:.1%}  ·  {match}"
        )
        if exact:
            self._metrics.remove_css_class("vie-metrics-bad")
            self._metrics.add_css_class("vie-metrics-ok")
        else:
            self._metrics.remove_css_class("vie-metrics-ok")
            self._metrics.add_css_class("vie-metrics-bad")
        self._compare.set_visible(True)

    def _on_copy(self, _btn: Gtk.Button) -> None:
        if not self._prediction:
            return
        self.get_display().get_clipboard().set(self._prediction)

    # ─── Paragraph mode ────────────────────────────────────────────────

    def show_paragraph(
        self,
        image_path: str | Path,
        seg_result: Any,
        ocr_results: list[tuple[str, float]] | None = None,
        overlay_image: "np.ndarray | None" = None,
        *,
        ground_truth: str | None = None,
        comparison: dict[str, Any] | None = None,
    ) -> None:
        """Display paragraph segmentation results.

        Args:
            image_path: Original image path.
            seg_result: SegmentationResult from segment_lines().
            ocr_results: Optional list of (text, elapsed_ms) per line.
            overlay_image: BGR image with seam lines drawn (from render_seam_overlay).
            ground_truth: Optional GT text for comparison.
            comparison: Optional pre-computed comparison dict.
        """
        p = Path(image_path)
        self._current_path = str(p)
        self._ground_truth = ground_truth
        self._filename.set_label(f"\U0001f4c4 {p.name} \u2014 {seg_result.n_lines} d\u00f2ng")
        self._filename.set_tooltip_text(str(p))

        if overlay_image is not None:
            self._set_picture_from_array(overlay_image)
        else:
            self._picture.set_filename(str(p))

        self._image_stack.set_visible_child_name("image")
        self._rerun_btn.set_sensitive(True)

        if ocr_results:
            lines_text = [text for text, _ in ocr_results]
            combined = "\n".join(lines_text)
            total_ms = sum(ms for _, ms in ocr_results)
            self._prediction = combined
            self._text.set_label(combined if combined.strip() else "(trống)")
            self._copy_btn.set_sensitive(bool(combined.strip()))
            self.set_latency(total_ms)
            self._apply_comparison(comparison)
        else:
            self._prediction = ""
            self._text.set_label(f"Đã phân tách {seg_result.n_lines} dòng. Load model để OCR.")
            self._copy_btn.set_sensitive(False)
            self.set_latency(None)
            if ground_truth is not None:
                self._gt_text.set_label(ground_truth)
                self._compare.set_visible(True)
                self._metrics.set_label("Chưa có prediction để so sánh.")
            else:
                self._compare.set_visible(False)

    def show_paragraph_with_seams(self, image_path: str | Path, seg_result: Any) -> None:
        """Update display to show/hide seam overlay."""
        from gui.segment_service import render_seam_overlay

        overlay = render_seam_overlay(image_path, seg_result)
        self._set_picture_from_array(overlay)

    def show_paragraph_without_seams(self, image_path: str | Path) -> None:
        """Revert to original image (no seam overlay)."""
        self._picture.set_filename(str(image_path))

    def _set_picture_from_array(self, bgr_image: "np.ndarray") -> None:
        """Set Gtk.Picture from a BGR numpy array via temporary file."""
        import cv2

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(tmp.name, bgr_image)
        self._picture.set_filename(tmp.name)
        self._image_stack.set_visible_child_name("image")
