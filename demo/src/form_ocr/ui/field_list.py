"""Sidebar list of template fields."""

from __future__ import annotations

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GObject, Gtk, Pango  # noqa: E402

from form_ocr.domain.template import FormTemplate


class FieldList(Gtk.Box):
    """Shows fields; emits selection / delete / rename requests."""

    __gsignals__ = {
        "field-selected": (GObject.SignalFlags.RUN_FIRST, None, (str,)),
        "field-delete": (GObject.SignalFlags.RUN_FIRST, None, (str,)),
        "field-rename": (GObject.SignalFlags.RUN_FIRST, None, (str, str)),
    }

    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.set_hexpand(True)
        self.set_vexpand(True)
        self._template: FormTemplate | None = None
        self._rows: dict[str, Gtk.ListBoxRow] = {}

        header = Gtk.Label(label="Fields", xalign=0.0)
        header.add_css_class("form-sidebar-header")
        self.append(header)

        self._list = Gtk.ListBox(selection_mode=Gtk.SelectionMode.SINGLE)
        self._list.add_css_class("navigation-sidebar")
        self._list.connect("row-selected", self._on_row_selected)
        scroll = Gtk.ScrolledWindow(hexpand=True, vexpand=True, vexpand_set=True)
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scroll.set_child(self._list)
        self.append(scroll)

        hint = Gtk.Label(
            label="Kéo chuột trên ảnh để thêm ROI",
            wrap=True,
            xalign=0.0,
        )
        hint.add_css_class("dim-label")
        hint.set_margin_start(12)
        hint.set_margin_end(12)
        hint.set_margin_bottom(8)
        self.append(hint)

    def set_template(self, template: FormTemplate | None) -> None:
        self._template = template
        self.refresh()

    def refresh(self) -> None:
        while (child := self._list.get_row_at_index(0)) is not None:
            self._list.remove(child)
        self._rows.clear()
        if not self._template:
            return
        for fd in self._template.fields:
            row = self._make_row(fd.id, fd.label)
            self._rows[fd.id] = row
            self._list.append(row)

    def select_field(self, field_id: str | None) -> None:
        if not field_id or field_id not in self._rows:
            self._list.unselect_all()
            return
        self._list.select_row(self._rows[field_id])

    def _make_row(self, field_id: str, label: str) -> Gtk.ListBoxRow:
        row = Gtk.ListBoxRow()
        row._field_id = field_id  # type: ignore[attr-defined]

        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        box.set_margin_start(8)
        box.set_margin_end(4)
        box.set_margin_top(4)
        box.set_margin_bottom(4)

        lab = Gtk.Label(label=label, xalign=0.0, hexpand=True)
        lab.set_ellipsize(Pango.EllipsizeMode.END)
        box.append(lab)

        edit_btn = Gtk.Button(icon_name="document-edit-symbolic")
        edit_btn.add_css_class("flat")
        edit_btn.set_tooltip_text("Đổi label")
        edit_btn.connect("clicked", lambda _b: self._prompt_rename(field_id, lab.get_text()))
        box.append(edit_btn)

        del_btn = Gtk.Button(icon_name="user-trash-symbolic")
        del_btn.add_css_class("flat")
        del_btn.set_tooltip_text("Xóa field")
        del_btn.connect("clicked", lambda _b: self.emit("field-delete", field_id))
        box.append(del_btn)

        row.set_child(box)
        return row

    def _on_row_selected(self, _lb: Gtk.ListBox, row: Gtk.ListBoxRow | None) -> None:
        if row is None:
            return
        fid = getattr(row, "_field_id", "")
        if fid:
            self.emit("field-selected", fid)

    def _prompt_rename(self, field_id: str, current: str) -> None:
        dialog = Gtk.Dialog(title="Đổi label", modal=True)
        dialog.add_button("Hủy", Gtk.ResponseType.CANCEL)
        dialog.add_button("OK", Gtk.ResponseType.OK)
        dialog.set_default_response(Gtk.ResponseType.OK)

        entry = Gtk.Entry(text=current)
        entry.set_activates_default(True)
        entry.set_margin_start(12)
        entry.set_margin_end(12)
        entry.set_margin_top(12)
        entry.set_margin_bottom(12)
        dialog.get_content_area().append(entry)

        def _on_response(dlg: Gtk.Dialog, response: int) -> None:
            if response == Gtk.ResponseType.OK:
                new = entry.get_text().strip()
                if new:
                    self.emit("field-rename", field_id, new)
            dlg.destroy()

        dialog.connect("response", _on_response)
        dialog.present()
