"""Batch step: pick scan images and run extraction."""

from __future__ import annotations

from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GObject, Gtk  # noqa: E402

from form_ocr.services.batch import IMAGE_EXTS, list_images


class BatchView(Gtk.Box):
    """Image list + progress for batch OCR."""

    __gsignals__ = {
        "extract-requested": (GObject.SignalFlags.RUN_FIRST, None, ()),
        "pick-folder-requested": (GObject.SignalFlags.RUN_FIRST, None, ()),
        "pick-files-requested": (GObject.SignalFlags.RUN_FIRST, None, ()),
    }

    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.set_margin_start(12)
        self.set_margin_end(12)
        self.set_margin_top(12)
        self.set_margin_bottom(12)
        self.paths: list[Path] = []

        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        folder_btn = Gtk.Button(label="Chọn thư mục")
        folder_btn.connect("clicked", lambda _b: self.emit("pick-folder-requested"))
        toolbar.append(folder_btn)
        files_btn = Gtk.Button(label="Chọn ảnh…")
        files_btn.connect("clicked", lambda _b: self.emit("pick-files-requested"))
        toolbar.append(files_btn)
        self._extract_btn = Gtk.Button(label="Extract")
        self._extract_btn.add_css_class("suggested-action")
        self._extract_btn.set_sensitive(False)
        self._extract_btn.connect("clicked", lambda _b: self.emit("extract-requested"))
        toolbar.append(self._extract_btn)
        self.append(toolbar)

        self._status = Gtk.Label(label="Chưa chọn ảnh scan", xalign=0.0)
        self._status.add_css_class("dim-label")
        self.append(self._status)

        self._list = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE)
        scroll = Gtk.ScrolledWindow(hexpand=True, vexpand=True)
        scroll.set_child(self._list)
        self.append(scroll)

        self._progress = Gtk.ProgressBar(show_text=True, fraction=0.0)
        self._progress.set_visible(False)
        self.append(self._progress)

    def set_paths(self, paths: list[Path]) -> None:
        self.paths = list(paths)
        while (row := self._list.get_row_at_index(0)) is not None:
            self._list.remove(row)
        for p in self.paths:
            row = Gtk.ListBoxRow()
            row.set_child(Gtk.Label(label=p.name, xalign=0.0, margin_start=8, margin_top=4, margin_bottom=4))
            self._list.append(row)
        n = len(self.paths)
        self._status.set_text(f"{n} ảnh đã chọn" if n else "Chưa chọn ảnh scan")
        self._extract_btn.set_sensitive(n > 0)
        self._progress.set_visible(False)
        self._progress.set_fraction(0.0)

    def set_folder(self, folder: str | Path) -> int:
        paths = list_images(folder)
        self.set_paths(paths)
        return len(paths)

    def add_files(self, files: list[Path]) -> None:
        existing = {p.resolve() for p in self.paths}
        for f in files:
            if f.suffix.lower() not in IMAGE_EXTS:
                continue
            if f.resolve() not in existing:
                self.paths.append(f)
                existing.add(f.resolve())
        self.set_paths(self.paths)

    def set_can_extract(self, enabled: bool) -> None:
        self._extract_btn.set_sensitive(enabled and bool(self.paths))

    def set_busy(self, busy: bool) -> None:
        self._extract_btn.set_sensitive(not busy and bool(self.paths))
        self._progress.set_visible(busy or self._progress.get_fraction() > 0)

    def set_progress(self, done: int, total: int, filename: str = "") -> None:
        self._progress.set_visible(True)
        frac = (done / total) if total else 0.0
        self._progress.set_fraction(frac)
        text = f"{done}/{total}"
        if filename:
            text += f" · {filename}"
        self._progress.set_text(text)
