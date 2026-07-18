"""GTK4 + libadwaita form-field extraction demo."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk  # noqa: E402

from form_ocr.domain.records import ExtractionResult
from form_ocr.domain.template import FormTemplate
from form_ocr.services.batch import BatchExtractor
from form_ocr.services.export import export_csv, export_excel
from form_ocr.services.ov_service import OvService, default_ov_dir
from form_ocr.services.recognize import SingleLineRecognizer
from form_ocr.ui.wizard import Wizard

logger = logging.getLogger(__name__)

APP_ID = "io.github.vie_handwritten.form_ocr"
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


def _safe_folder(*candidates: str | Path | None) -> Gio.File:
    """First existing directory as Gio.File (avoids broken recently-used paths)."""
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.is_dir():
            return Gio.File.new_for_path(str(path.resolve()))
    return Gio.File.new_for_path(GLib.get_home_dir())


def _image_filters() -> tuple[Gio.ListStore, Gtk.FileFilter]:
    filters = Gio.ListStore.new(Gtk.FileFilter)
    img = Gtk.FileFilter(name="Images")
    for mime in ("image/png", "image/jpeg", "image/bmp", "image/tiff", "image/webp"):
        img.add_mime_type(mime)
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp"):
        img.add_pattern(pattern)
    filters.append(img)
    return filters, img


def _make_open_dialog(title: str) -> Gtk.FileDialog:
    """File open dialog pinned to a valid folder (not stale Recent entries)."""
    pictures = GLib.get_user_special_dir(GLib.UserDirectory.DIRECTORY_PICTURES)
    dialog = Gtk.FileDialog(
        title=title,
        initial_folder=_safe_folder(pictures, GLib.get_home_dir(), Path.cwd()),
    )
    filters, default = _image_filters()
    dialog.set_filters(filters)
    dialog.set_default_filter(default)
    return dialog


class MainWindow(Adw.ApplicationWindow):
    def __init__(self, app: Adw.Application) -> None:
        super().__init__(application=app, title="Form OCR Demo")
        self.set_default_size(1180, 780)

        self.template = FormTemplate()
        self.result = ExtractionResult()
        self.ov_service = OvService()
        self.extractor = BatchExtractor()
        self._recognizer: SingleLineRecognizer | None = None

        self._toast_overlay = Adw.ToastOverlay()
        self.wizard = Wizard()

        toolbar = Adw.ToolbarView()
        header = Adw.HeaderBar()
        header.set_title_widget(self.wizard.switcher)

        load_tpl = Gtk.Button(label="Load Template")
        load_tpl.add_css_class("suggested-action")
        load_tpl.connect("clicked", self._on_load_template)
        header.pack_start(load_tpl)

        self._model_lbl = Gtk.Label(label="Model: …")
        self._model_lbl.add_css_class("dim-label")
        header.pack_end(self._model_lbl)

        toolbar.add_top_bar(header)
        toolbar.set_content(self.wizard)
        self._toast_overlay.set_child(toolbar)
        self.set_content(self._toast_overlay)

        # Wire UI signals
        self.wizard.canvas.connect("request-label", self._on_request_label)
        self.wizard.canvas.connect("selection-changed", self._on_canvas_selection)
        self.wizard.field_list.connect("field-selected", self._on_field_selected)
        self.wizard.field_list.connect("field-delete", self._on_field_delete)
        self.wizard.field_list.connect("field-rename", self._on_field_rename)

        self.wizard.batch.connect("pick-folder-requested", self._on_pick_folder)
        self.wizard.batch.connect("pick-files-requested", self._on_pick_files)
        self.wizard.batch.connect("extract-requested", self._on_extract)

        self.wizard.results.connect("export-csv", self._on_export_csv)
        self.wizard.results.connect("export-excel", self._on_export_excel)

        self._autoload_model()

    def toast(self, message: str) -> None:
        self._toast_overlay.add_toast(Adw.Toast(title=message, timeout=3))

    def _autoload_model(self) -> None:
        ov_dir = default_ov_dir()
        if not ov_dir.is_dir():
            self._model_lbl.set_text("Model: chưa có (chạy make demo-ov)")
            self.toast(f"Chưa có OpenVINO model tại {ov_dir}")
            return
        self._model_lbl.set_text("Model: đang load…")

        def _done(info, err) -> None:
            def _ui() -> bool:
                if err is not None:
                    self._model_lbl.set_text("Model: lỗi")
                    self.toast(f"Load model lỗi: {err}")
                    logger.exception("OV load failed", exc_info=err)
                    return False
                assert info is not None
                assert self.ov_service.ov is not None
                self._recognizer = SingleLineRecognizer(self.ov_service.ov)
                self._model_lbl.set_text(
                    f"Model: {info['precision']} · {info['decode']}"
                )
                self.toast("Đã load OpenVINO model")
                return False

            GLib.idle_add(_ui)

        if not self.ov_service.load_async(_done):
            self.toast("Đang bận load model")

    # --- Template ---

    def _on_load_template(self, _btn: Gtk.Button) -> None:
        dialog = _make_open_dialog("Chọn ảnh biểu mẫu (template)")
        dialog.open(self, None, self._on_template_chosen)

    def _on_template_chosen(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        try:
            f = dialog.open_finish(result)
        except GLib.Error as exc:
            if exc.matches(Gio.io_error_quark(), Gio.IOErrorEnum.CANCELLED):
                return
            self.toast(f"Không mở được ảnh: {exc.message}")
            return
        if f is None or not f.get_path():
            return
        path = Path(f.get_path())
        self.template.clear()
        self.template.image_path = path
        try:
            self.wizard.canvas.set_template(self.template)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to display template")
            self.toast(f"Không hiển thị được ảnh: {exc}")
            return
        self.wizard.field_list.set_template(self.template)
        self.wizard.show_page("template")
        self.toast(f"Template: {path.name}")

    def _on_request_label(
        self, _canvas, x: float, y: float, w: float, h: float
    ) -> None:
        dialog = Adw.AlertDialog(
            heading="Label cho field",
            body="Nhập tên trường dữ liệu (unique).",
        )
        entry = Gtk.Entry()
        entry.set_activates_default(True)
        entry.set_hexpand(True)
        dialog.set_extra_child(entry)
        dialog.add_response("cancel", "Hủy")
        dialog.add_response("ok", "Thêm")
        dialog.set_response_appearance("ok", Adw.ResponseAppearance.SUGGESTED)
        dialog.set_default_response("ok")
        dialog.set_close_response("cancel")

        def _on_response(_d: Adw.AlertDialog, response: str) -> None:
            if response != "ok":
                return
            label = entry.get_text().strip()
            try:
                fd = self.template.add_field(label, (x, y, w, h))
            except ValueError as exc:
                self.toast(str(exc))
                return
            self.wizard.field_list.refresh()
            self.wizard.field_list.select_field(fd.id)
            self.wizard.canvas.set_selected(fd.id)
            self.wizard.canvas.refresh()
            self.toast(f"Đã thêm field: {label}")

        dialog.connect("response", _on_response)
        dialog.present(self)
        entry.grab_focus()

    def _on_canvas_selection(self, _canvas, field_id: str) -> None:
        self.wizard.field_list.select_field(field_id or None)

    def _on_field_selected(self, _list, field_id: str) -> None:
        self.wizard.canvas.set_selected(field_id)

    def _on_field_delete(self, _list, field_id: str) -> None:
        self.template.remove_field(field_id)
        self.wizard.field_list.refresh()
        self.wizard.canvas.set_selected(None)
        self.wizard.canvas.refresh()

    def _on_field_rename(self, _list, field_id: str, label: str) -> None:
        try:
            self.template.update_label(field_id, label)
        except (ValueError, KeyError) as exc:
            self.toast(str(exc))
            return
        self.wizard.field_list.refresh()
        self.wizard.field_list.select_field(field_id)
        self.wizard.canvas.refresh()

    # --- Batch ---

    def _on_pick_folder(self, _view) -> None:
        pictures = GLib.get_user_special_dir(GLib.UserDirectory.DIRECTORY_PICTURES)
        dialog = Gtk.FileDialog(
            title="Chọn thư mục ảnh tờ khai",
            initial_folder=_safe_folder(pictures, GLib.get_home_dir(), Path.cwd()),
        )
        dialog.select_folder(self, None, self._on_folder_chosen)

    def _on_folder_chosen(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        try:
            folder = dialog.select_folder_finish(result)
        except GLib.Error as exc:
            if exc.matches(Gio.io_error_quark(), Gio.IOErrorEnum.CANCELLED):
                return
            self.toast(f"Không mở được thư mục: {exc.message}")
            return
        if folder is None or not folder.get_path():
            return
        n = self.wizard.batch.set_folder(folder.get_path())
        self.toast(f"Đã chọn {n} ảnh")

    def _on_pick_files(self, _view) -> None:
        dialog = _make_open_dialog("Chọn ảnh tờ khai")
        dialog.open_multiple(self, None, self._on_files_chosen)

    def _on_files_chosen(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        try:
            files = dialog.open_multiple_finish(result)
        except GLib.Error as exc:
            if exc.matches(Gio.io_error_quark(), Gio.IOErrorEnum.CANCELLED):
                return
            self.toast(f"Không mở được file: {exc.message}")
            return
        if files is None:
            return
        paths: list[Path] = []
        n = files.get_n_items()
        for i in range(n):
            f = files.get_item(i)
            if f is None:
                continue
            p = f.get_path()
            if p:
                paths.append(Path(p))
        self.wizard.batch.add_files(paths)
        self.toast(f"Đã chọn {len(self.wizard.batch.paths)} ảnh")

    def _on_extract(self, _view) -> None:
        if not self.template.ready:
            self.toast("Cần ảnh template và ít nhất 1 field")
            return
        if self._recognizer is None or not self.ov_service.ready:
            self.toast("Model chưa sẵn sàng")
            return
        paths = self.wizard.batch.paths
        if not paths:
            self.toast("Chưa chọn ảnh scan")
            return

        self.wizard.batch.set_busy(True)
        self.toast(f"Đang extract {len(paths)} ảnh…")

        def _progress(done: int, total: int, filename: str) -> None:
            GLib.idle_add(self.wizard.batch.set_progress, done, total, filename)

        def _done(result, err) -> None:
            def _ui() -> bool:
                self.wizard.batch.set_busy(False)
                if err is not None:
                    self.toast(f"Extract lỗi: {err}")
                    logger.exception("extract failed", exc_info=err)
                    return False
                assert result is not None
                self.result = result
                self.wizard.results.set_result(result)
                self.wizard.show_page("results")
                self.toast(f"Xong {len(result.records)} records")
                return False

            GLib.idle_add(_ui)

        if not self.extractor.run_async(
            paths,
            self.template,
            self._recognizer,
            on_progress=_progress,
            on_done=_done,
        ):
            self.wizard.batch.set_busy(False)
            self.toast("Đang bận extract")

    # --- Export ---

    def _on_export_csv(self, _view) -> None:
        result = self.wizard.results.get_result()
        if result is None or not result.records:
            self.toast("Không có dữ liệu để export")
            return
        dialog = Gtk.FileDialog(
            title="Lưu CSV",
            initial_name="records.csv",
            initial_folder=_safe_folder(
                GLib.get_user_special_dir(GLib.UserDirectory.DIRECTORY_DOCUMENTS),
                GLib.get_home_dir(),
            ),
        )
        dialog.save(self, None, lambda d, r: self._finish_export(d, r, result, "csv"))

    def _on_export_excel(self, _view) -> None:
        result = self.wizard.results.get_result()
        if result is None or not result.records:
            self.toast("Không có dữ liệu để export")
            return
        dialog = Gtk.FileDialog(
            title="Lưu Excel",
            initial_name="records.xlsx",
            initial_folder=_safe_folder(
                GLib.get_user_special_dir(GLib.UserDirectory.DIRECTORY_DOCUMENTS),
                GLib.get_home_dir(),
            ),
        )
        dialog.save(self, None, lambda d, r: self._finish_export(d, r, result, "xlsx"))

    def _finish_export(
        self,
        dialog: Gtk.FileDialog,
        result_async: Gio.AsyncResult,
        data: ExtractionResult,
        kind: str,
    ) -> None:
        try:
            f = dialog.save_finish(result_async)
        except GLib.Error as exc:
            if exc.matches(Gio.io_error_quark(), Gio.IOErrorEnum.CANCELLED):
                return
            self.toast(f"Không lưu được: {exc.message}")
            return
        if f is None or not f.get_path():
            return
        path = Path(f.get_path())
        try:
            if kind == "csv":
                export_csv(data, path)
            else:
                if path.suffix.lower() != ".xlsx":
                    path = path.with_suffix(".xlsx")
                export_excel(data, path)
        except Exception as exc:  # noqa: BLE001
            self.toast(f"Export lỗi: {exc}")
            logger.exception("export failed")
            return
        self.toast(f"Đã export: {path.name}")


class FormOcrApp(Adw.Application):
    def __init__(self) -> None:
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.FLAGS_NONE)
        self._window: MainWindow | None = None

    def do_activate(self) -> None:  # noqa: N802
        _load_css()
        if self._window is None:
            self._window = MainWindow(self)
        self._window.present()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = FormOcrApp()
    return app.run(argv if argv is not None else sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
