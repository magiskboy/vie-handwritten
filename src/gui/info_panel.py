"""Sidebar footer: compact model metadata."""

from __future__ import annotations

from typing import Any

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk, Pango  # noqa: E402


class InfoPanel(Gtk.Box):
    """Compact key/value block (not PreferencesGroup — too bulky for a sidebar)."""

    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add_css_class("vie-sidebar-footer")

        title = Gtk.Label(label="Model", xalign=0.0)
        title.add_css_class("heading")
        self.append(title)

        self._grid = Gtk.Grid(column_spacing=10, row_spacing=4)
        self._vals: dict[str, Gtk.Label] = {}
        for i, (key, label) in enumerate(
            (
                ("checkpoint", "Checkpoint"),
                ("decode", "Decode"),
                ("device", "GPU"),
                ("status", "Status"),
            )
        ):
            key_lbl = Gtk.Label(label=label, xalign=0.0, yalign=0.0)
            key_lbl.add_css_class("vie-kv-key")
            val_lbl = Gtk.Label(label="—", xalign=0.0, yalign=0.0, hexpand=True)
            val_lbl.add_css_class("vie-kv-val")
            val_lbl.set_ellipsize(Pango.EllipsizeMode.MIDDLE)
            val_lbl.set_selectable(True)
            self._grid.attach(key_lbl, 0, i, 1, 1)
            self._grid.attach(val_lbl, 1, i, 1, 1)
            self._vals[key] = val_lbl

        self.append(self._grid)
        self.set_status("Chưa load model")

    def set_status(self, text: str) -> None:
        self._vals["status"].set_label(text or "—")

    def update(self, info: dict[str, Any]) -> None:
        ckpt = info.get("checkpoint") or ""
        self._vals["checkpoint"].set_label(self._basename(ckpt) if ckpt else "—")
        self._vals["checkpoint"].set_tooltip_text(str(ckpt) if ckpt else "")

        decode = str(info.get("decode") or "—")
        note = info.get("decode_note") or ""
        classes = info.get("num_classes")
        extra = f" · {classes} classes" if classes not in ("", None) else ""
        self._vals["decode"].set_label(decode + extra)
        if note:
            self._vals["decode"].set_tooltip_text(note)

        gpu_names = [n for n in (info.get("gpu_names") or []) if n]
        tf_ver = info.get("tensorflow") or ""
        if gpu_names:
            device = gpu_names[0]
            if len(gpu_names) > 1:
                device = f"{device} (+{len(gpu_names) - 1})"
            tip = " · ".join(gpu_names)
            if tf_ver:
                tip = f"{tip} · TF {tf_ver}"
            self._vals["device"].set_label(device)
            self._vals["device"].set_tooltip_text(tip)
            self._vals["device"].set_ellipsize(Pango.EllipsizeMode.END)
        else:
            device = f"CPU · TF {tf_ver}" if tf_ver else "CPU"
            self._vals["device"].set_label(device)
            self._vals["device"].set_tooltip_text("")

        if info.get("ready"):
            self.set_status("Sẵn sàng")
        else:
            self.set_status("Chưa load model")

    @staticmethod
    def _basename(path: str) -> str:
        return path.rsplit("/", 1)[-1] if path else "—"
