"""Results table + export actions."""

from __future__ import annotations

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GObject, Gtk  # noqa: E402

from form_ocr.domain.records import ExtractionResult


class ResultsView(Gtk.Box):
    """Editable grid of extraction results."""

    __gsignals__ = {
        "export-csv": (GObject.SignalFlags.RUN_FIRST, None, ()),
        "export-excel": (GObject.SignalFlags.RUN_FIRST, None, ()),
    }

    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.set_margin_start(12)
        self.set_margin_end(12)
        self.set_margin_top(12)
        self.set_margin_bottom(12)
        self._result: ExtractionResult | None = None
        self._labels: list[str] = []
        self._entries: dict[tuple[int, str], Gtk.Entry] = {}

        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        csv_btn = Gtk.Button(label="Export CSV")
        csv_btn.add_css_class("suggested-action")
        csv_btn.connect("clicked", lambda _b: self.emit("export-csv"))
        toolbar.append(csv_btn)
        xlsx_btn = Gtk.Button(label="Export Excel")
        xlsx_btn.connect("clicked", lambda _b: self.emit("export-excel"))
        toolbar.append(xlsx_btn)
        self._export_btns = (csv_btn, xlsx_btn)
        for b in self._export_btns:
            b.set_sensitive(False)
        self.append(toolbar)

        self._status = Gtk.Label(label="Chưa có kết quả", xalign=0.0)
        self._status.add_css_class("dim-label")
        self.append(self._status)

        self._grid = Gtk.Grid(column_spacing=6, row_spacing=4)
        self._grid.set_margin_top(4)
        scroll = Gtk.ScrolledWindow(hexpand=True, vexpand=True)
        scroll.set_child(self._grid)
        self.append(scroll)

    def set_result(self, result: ExtractionResult | None) -> None:
        self._result = result
        self._rebuild()

    def get_result(self) -> ExtractionResult | None:
        """Return result with values synced from editable entries."""
        if self._result is None:
            return None
        for (row_i, label), entry in self._entries.items():
            if row_i < len(self._result.records):
                rec = self._result.records[row_i]
                for fv in rec.fields:
                    if fv.label == label:
                        fv.value = entry.get_text()
                        break
        return self._result

    def _rebuild(self) -> None:
        while True:
            child = self._grid.get_first_child()
            if child is None:
                break
            self._grid.remove(child)
        self._entries.clear()
        self._labels = []

        if not self._result or not self._result.records:
            self._status.set_text("Chưa có kết quả")
            for b in self._export_btns:
                b.set_sensitive(False)
            return

        labels: list[str] = []
        for rec in self._result.records:
            for fv in rec.fields:
                if fv.label not in labels:
                    labels.append(fv.label)
        self._labels = labels

        # Header
        self._grid.attach(Gtk.Label(label="filename", xalign=0.0), 0, 0, 1, 1)
        for col, lab in enumerate(labels, start=1):
            self._grid.attach(Gtk.Label(label=lab, xalign=0.0), col, 0, 1, 1)

        for row_i, rec in enumerate(self._result.records, start=1):
            name = Gtk.Label(label=rec.filename, xalign=0.0)
            if rec.error:
                name.set_tooltip_text(rec.error)
            self._grid.attach(name, 0, row_i, 1, 1)
            by_label = {fv.label: fv.value for fv in rec.fields}
            for col, lab in enumerate(labels, start=1):
                entry = Gtk.Entry(text=by_label.get(lab, ""))
                entry.set_width_chars(16)
                self._grid.attach(entry, col, row_i, 1, 1)
                self._entries[(row_i - 1, lab)] = entry

        n = len(self._result.records)
        errs = sum(1 for r in self._result.records if r.error)
        msg = f"{n} records"
        if errs:
            msg += f" ({errs} lỗi đọc ảnh)"
        self._status.set_text(msg)
        for b in self._export_btns:
            b.set_sensitive(True)
