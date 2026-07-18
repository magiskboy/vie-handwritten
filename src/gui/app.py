"""GTK4 + libadwaita OCR viewer application."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk  # noqa: E402

from gui.content_view import ContentView  # noqa: E402
from gui.image_list import ImageList  # noqa: E402
from gui.info_panel import InfoPanel  # noqa: E402
from gui.model_service import (  # noqa: E402
    ModelService,
    compare_prediction,
    load_folder_labels,
    lookup_label,
)
from gui.segment_service import SegmentService, is_multiline, render_seam_overlay  # noqa: E402

logger = logging.getLogger(__name__)

APP_ID = "io.github.vie_handwritten.ocr"
_CSS_PATH = Path(__file__).with_name("style.css")


def _load_css() -> None:
    if not _CSS_PATH.is_file():
        return
    display = Gdk.Display.get_default()
    if display is None:
        return
    provider = Gtk.CssProvider()
    provider.load_from_path(str(_CSS_PATH))
    Gtk.StyleContext.add_provider_for_display(
        display,
        provider,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
    )


class MainWindow(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application) -> None:
        super().__init__(application=app, title="vie-OCR")
        self.set_default_size(1120, 740)

        self.service = ModelService()
        self.seg_service = SegmentService()
        self._cache: dict[str, tuple[str, float]] = {}
        self._labels: dict[str, str] = {}
        self._current_image: str | None = None
        self._current_seg_result = None  # SegmentationResult for current paragraph
        self._seam_visible = False
        self._toast_overlay = Adw.ToastOverlay()

        self._image_list = ImageList()
        self._image_list.connect("image-selected", self._on_image_selected)
        self._info = InfoPanel()
        self._content = ContentView()
        self._content.connect("rerun", self._on_rerun)

        # One continuous sidebar surface (list + compact model footer).
        sidebar_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        sidebar_box.add_css_class("vie-sidebar")
        sidebar_box.append(self._image_list)
        sidebar_box.append(self._info)

        sidebar_page = Adw.NavigationPage(title="Ảnh", child=sidebar_box)
        content_page = Adw.NavigationPage(title="Nhận dạng", child=self._content)

        split = Adw.NavigationSplitView()
        split.set_sidebar(sidebar_page)
        split.set_content(content_page)
        split.set_min_sidebar_width(240)
        split.set_max_sidebar_width(340)
        split.set_sidebar_width_fraction(0.24)

        toolbar = Adw.ToolbarView()
        header = Adw.HeaderBar()

        load_model_btn = Gtk.Button(label="Load Model")
        load_model_btn.add_css_class("suggested-action")
        load_model_btn.connect("clicked", self._on_load_model)
        header.pack_start(load_model_btn)

        load_folder_btn = Gtk.Button(label="Load Images")
        load_folder_btn.connect("clicked", self._on_load_folder)
        header.pack_start(load_folder_btn)

        self._seam_btn = Gtk.ToggleButton(label="Seams")
        self._seam_btn.set_icon_name("view-grid-symbolic")
        self._seam_btn.set_tooltip_text("Hiện/ẩn đường phân tách dòng (seam carving)")
        self._seam_btn.set_sensitive(False)
        self._seam_btn.connect("toggled", self._on_seam_toggled)
        header.pack_end(self._seam_btn)

        toolbar.add_top_bar(header)
        toolbar.set_content(split)
        self._toast_overlay.set_child(toolbar)
        self.set_content(self._toast_overlay)

        bp = Adw.Breakpoint.new(Adw.BreakpointCondition.parse("max-width: 720sp"))
        bp.add_setter(split, "collapsed", True)
        self.add_breakpoint(bp)

    def toast(self, message: str) -> None:
        self._toast_overlay.add_toast(Adw.Toast(title=message, timeout=3))

    def _on_load_model(self, _btn: Gtk.Button) -> None:
        dialog = Gtk.FileDialog(
            title="Chọn thư mục model (Keras checkpoint hoặc OpenVINO artifact)",
            initial_folder=Gio.File.new_for_path(GLib.get_current_dir()),
        )
   
        dialog.select_folder(self, None, self._on_model_chosen)

    def _on_model_chosen(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        try:
            folder = dialog.select_folder_finish(result)
        except GLib.Error as exc:
            if exc.matches(Gio.io_error_quark(), Gio.IOErrorEnum.CANCELLED):
                return
            self.toast(f"Không mở được thư mục: {exc.message}")
            return
        if folder is None:
            return
        path = folder.get_path()
        if not path:
            self.toast("Đường dẫn checkpoint không hợp lệ")
            return

        self._info.set_status("Đang load model…")
        self._content.set_busy(True)
        self._cache.clear()

        def _done(info, err) -> None:
            def _ui() -> bool:
                self._content.set_busy(False)
                if err is not None:
                    self._info.set_status("Load thất bại")
                    self.toast(f"Load model lỗi: {err}")
                    logger.exception("load model failed", exc_info=err)
                    return False
                assert info is not None
                self._info.update(info)
                note = info.get("decode_note") or ""
                backend = {"keras": "Keras", "openvino": "OpenVINO"}.get(
                    str(info.get("backend") or ""), "model"
                )
                device = str(info.get("device") or "")
                detail = backend + (f" · {device}" if device else "")
                if note:
                    detail = f"{detail}; {note}"
                self.toast(f"Đã load {detail}")
                if self._current_image:
                    self._recognize(self._current_image)
                return False

            GLib.idle_add(_ui)

        if not self.service.load_async(path, _done):
            self._content.set_busy(False)
            self.toast("Đang bận, thử lại sau")

    def _on_load_folder(self, _btn: Gtk.Button) -> None:
        dialog = Gtk.FileDialog(title="Chọn thư mục ảnh", initial_folder=Gio.File.new_for_path(GLib.get_current_dir()))
        dialog.select_folder(self, None, self._on_folder_chosen)

    def _on_folder_chosen(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        try:
            folder = dialog.select_folder_finish(result)
        except GLib.Error as exc:
            if exc.matches(Gio.io_error_quark(), Gio.IOErrorEnum.CANCELLED):
                return
            self.toast(f"Không mở được thư mục: {exc.message}")
            return
        if folder is None:
            return
        path = folder.get_path()
        if not path:
            self.toast("Đường dẫn thư mục không hợp lệ")
            return

        self._cache.clear()
        self._content.clear()
        self._current_image = None
        self._labels = load_folder_labels(path)
        n = self._image_list.set_folder(path)
        if self._labels:
            matched = sum(1 for p in Path(path).iterdir() if p.name in self._labels)
            self.toast(f"Đã tải {n} ảnh · {len(self._labels)} labels ({matched} khớp tên)")
        else:
            self.toast(f"Đã tải {n} ảnh (không có label.json)")

    def _gt_for(self, path: str) -> str | None:
        return lookup_label(self._labels, path)

    def _comparison_for(self, path: str, prediction: str) -> dict | None:
        gt = self._gt_for(path)
        if gt is None:
            return None
        return compare_prediction(gt, prediction)

    def _on_image_selected(self, _list: ImageList, path: str) -> None:
        self._current_image = path
        self._current_seg_result = None
        self._seam_visible = False
        self._seam_btn.set_active(False)
        self._seam_btn.set_sensitive(False)

        # Auto-detect: multi-line paragraph?
        if is_multiline(path):
            self._run_paragraph_flow(path)
        else:
            self._content.show_image(path, ground_truth=self._gt_for(path))
            if path in self._cache:
                text, elapsed_ms = self._cache[path]
                self._content.set_prediction(
                    text,
                    elapsed_ms=elapsed_ms,
                    comparison=self._comparison_for(path, text),
                )
                return
            if not self.service.ready:
                self._content.set_message("Hãy load model trước.")
                return
            self._recognize(path)

    def _on_rerun(self, _view: ContentView) -> None:
        if self._current_image:
            self._cache.pop(self._current_image, None)
            self._current_seg_result = None
            if is_multiline(self._current_image):
                self._run_paragraph_flow(self._current_image)
            else:
                self._recognize(self._current_image)

    def _recognize(self, path: str) -> None:
        if not self.service.ready:
            self._content.set_message("Hãy load model trước.")
            return

        self._content.set_busy(True)
        self._content.set_message("Đang nhận dạng…")
        self._info.set_status("Đang nhận dạng…")

        def _done(text, elapsed_ms, err) -> None:
            def _ui() -> bool:
                self._content.set_busy(False)
                if err is not None:
                    self._info.set_status("Lỗi nhận dạng")
                    self._content.set_message(f"Lỗi: {err}")
                    self.toast(f"OCR lỗi: {err}")
                    logger.exception("recognize failed", exc_info=err)
                    return False
                assert text is not None and elapsed_ms is not None
                self._cache[path] = (text, elapsed_ms)
                self._image_list.mark_recognized(path, recognized=True)
                if self._current_image == path:
                    self._content.set_prediction(
                        text,
                        elapsed_ms=elapsed_ms,
                        comparison=self._comparison_for(path, text),
                    )
                self._info.set_status("Sẵn sàng")
                return False

            GLib.idle_add(_ui)

        if not self.service.recognize_async(path, _done):
            self._content.set_busy(False)
            self.toast("Đang bận, thử lại sau")


    def _run_paragraph_flow(self, path: str) -> None:
        """Segment image into lines, then OCR each line."""
        self._content.set_busy(True)
        self._content.set_message("Đang phân tách dòng…")
        self._info.set_status("Đang phân tách…")

        def _on_segment_done(seg_result, elapsed_ms, err) -> None:
            def _ui() -> bool:
                if err is not None:
                    self._content.set_busy(False)
                    self._content.set_message(f"Lỗi phân tách: {err}")
                    self._info.set_status("Lỗi phân tách")
                    self.toast(f"Segment lỗi: {err}")
                    logger.exception("segment failed", exc_info=err)
                    return False

                self._current_seg_result = seg_result
                self._seam_btn.set_sensitive(bool(seg_result.seams))

                if self.service.ready:
                    self._recognize_paragraph(path, seg_result)
                else:
                    self._content.set_busy(False)
                    self._content.show_paragraph(
                        path, seg_result, ground_truth=self._gt_for(path)
                    )
                    self._info.set_status("Sẵn sàng")
                    self.toast(f"Phân tách {seg_result.n_lines} dòng. Load model để OCR.")
                return False

            GLib.idle_add(_ui)

        if not self.seg_service.segment_async(path, _on_segment_done):
            self._content.set_busy(False)
            self.toast("Đang bận, thử lại sau")

    def _recognize_paragraph(self, path: str, seg_result) -> None:
        """OCR all segmented lines in a background thread."""
        self._content.set_message("Đang nhận dạng từng dòng…")
        self._info.set_status("Đang nhận dạng…")

        import threading

        def _run() -> None:
            ocr_results = None
            err = None
            try:
                ocr_results = self.service.recognize_lines(seg_result.lines_gray)
            except BaseException as exc:  # noqa: BLE001
                err = exc
            finally:
                self.service._busy = False

            def _ui() -> bool:
                self._content.set_busy(False)
                if err is not None:
                    self._content.set_message(f"Lỗi OCR: {err}")
                    self._info.set_status("Lỗi OCR")
                    self.toast(f"OCR lỗi: {err}")
                    return False

                # Cache combined text
                combined = "\n".join(text for text, _ in ocr_results)
                total_ms = sum(ms for _, ms in ocr_results)
                self._cache[path] = (combined, total_ms)
                self._image_list.mark_recognized(path, recognized=True)

                # Show with seam overlay if toggled
                overlay = None
                if self._seam_visible and seg_result.seams:
                    overlay = render_seam_overlay(path, seg_result)

                self._content.show_paragraph(
                    path,
                    seg_result,
                    ocr_results=ocr_results,
                    overlay_image=overlay,
                    ground_truth=self._gt_for(path),
                    comparison=self._comparison_for(path, combined),
                )
                self._info.set_status("Sẵn sàng")
                return False

            GLib.idle_add(_ui)

        self.service._busy = True
        threading.Thread(target=_run, name="ocr-paragraph", daemon=True).start()

    def _on_seam_toggled(self, btn: Gtk.ToggleButton) -> None:
        """Toggle seam line visibility on the paragraph image."""
        self._seam_visible = btn.get_active()
        if self._current_image is None or self._current_seg_result is None:
            return
        if self._seam_visible:
            self._content.show_paragraph_with_seams(
                self._current_image, self._current_seg_result
            )
        else:
            self._content.show_paragraph_without_seams(self._current_image)


class OCRApplication(Adw.Application):
    def __init__(self) -> None:
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.FLAGS_NONE)
        self.connect("startup", self._on_startup)
        self.connect("activate", self._on_activate)

    def _on_startup(self, _app: Adw.Application) -> None:
        _load_css()

    def _on_activate(self, app: Adw.Application) -> None:
        win = app.get_active_window()
        if win is None:
            win = MainWindow(app)
        win.present()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app = OCRApplication()
    return app.run(argv if argv is not None else sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
