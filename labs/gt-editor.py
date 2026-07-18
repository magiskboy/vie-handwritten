#!/usr/bin/env python3
"""HWDB_line ground-truth editor (GTK4).

Browse line images in a sample directory, edit the GT string, and save
in-place to label.json / labels.json.

Usage:
    python labs/gt-editor.py data/images/HWDB_line/test_data/250
    python labs/gt-editor.py   # open folder dialog
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")

from gi.repository import Gdk, GLib, Gtk  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_CANDIDATES = ("label.json", "labels.json")


def natural_key(name: str) -> tuple:
    parts = re.split(r"(\d+)", name)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


def find_label_path(directory: Path) -> Path | None:
    for name in LABEL_CANDIDATES:
        path = directory / name
        if path.is_file():
            return path
    return None


def load_labels(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return {str(k): "" if v is None else str(v) for k, v in data.items()}


def save_labels(path: Path, labels: dict[str, str]) -> None:
    text = json.dumps(labels, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


def discover_entries(directory: Path, labels: dict[str, str]) -> list[str]:
    """Prefer label keys that have an image; include orphan images last."""
    disk_images = {
        p.name
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }
    from_labels = [k for k in labels if k in disk_images or (directory / k).is_file()]
    orphans = sorted(disk_images - set(from_labels), key=natural_key)
    return sorted(from_labels, key=natural_key) + orphans


class GtEditor(Gtk.ApplicationWindow):
    def __init__(self, app: Gtk.Application, initial_dir: Path | None) -> None:
        super().__init__(application=app, title="HWDB GT Editor")
        self.set_default_size(1100, 420)

        self.directory: Path | None = None
        self.label_path: Path | None = None
        self.labels: dict[str, str] = {}
        self.saved_labels: dict[str, str] = {}
        self.entries: list[str] = []
        self.index = 0
        self._loading = False

        self._build_ui()
        self._install_shortcuts()
        self.connect("close-request", self._on_close_request)

        if initial_dir is not None:
            GLib.idle_add(self.open_directory, initial_dir)

    def _build_ui(self) -> None:
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        root.set_margin_top(8)
        root.set_margin_bottom(8)
        root.set_margin_start(8)
        root.set_margin_end(8)
        self.set_child(root)

        toolbar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        root.append(toolbar)

        self.open_btn = Gtk.Button(label="Open…")
        self.open_btn.connect("clicked", self._on_open_clicked)
        toolbar.append(self.open_btn)

        self.save_btn = Gtk.Button(label="Save")
        self.save_btn.connect("clicked", lambda *_: self.save())
        toolbar.append(self.save_btn)

        toolbar.append(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL))

        self.prev_btn = Gtk.Button(label="◀ Prev")
        self.prev_btn.connect("clicked", lambda *_: self.goto(self.index - 1))
        toolbar.append(self.prev_btn)

        self.next_btn = Gtk.Button(label="Next ▶")
        self.next_btn.connect("clicked", lambda *_: self.goto(self.index + 1))
        toolbar.append(self.next_btn)

        toolbar.append(Gtk.Label(label="Go"))
        self.goto_entry = Gtk.Entry()
        self.goto_entry.set_width_chars(5)
        self.goto_entry.set_placeholder_text("#")
        self.goto_entry.connect("activate", self._on_goto_activate)
        toolbar.append(self.goto_entry)

        self.pos_label = Gtk.Label(label="— / —")
        self.pos_label.set_margin_start(8)
        toolbar.append(self.pos_label)

        self.file_label = Gtk.Label(label="")
        self.file_label.set_hexpand(True)
        self.file_label.set_xalign(0.0)
        self.file_label.set_margin_start(8)
        self.file_label.add_css_class("dim-label")
        toolbar.append(self.file_label)

        self.dirty_label = Gtk.Label(label="")
        toolbar.append(self.dirty_label)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_vexpand(True)
        scrolled.set_hexpand(True)
        root.append(scrolled)

        self.image = Gtk.Picture()
        self.image.set_can_shrink(True)
        self.image.set_content_fit(Gtk.ContentFit.CONTAIN)
        scrolled.set_child(self.image)

        gt_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        root.append(gt_row)

        gt_row.append(Gtk.Label(label="GT"))
        self.gt_entry = Gtk.Entry()
        self.gt_entry.set_hexpand(True)
        self.gt_entry.connect("changed", self._on_gt_changed)
        self.gt_entry.connect("activate", lambda *_: self.save_and_next())
        gt_row.append(self.gt_entry)

        hint = Gtk.Label(
            label="PgUp/PgDn navigate · Ctrl+S save · Enter save+next · Go # then Enter · Ctrl+O open"
        )
        hint.add_css_class("dim-label")
        hint.set_xalign(0.0)
        root.append(hint)

        self.status = Gtk.Label(label="Open a sample directory to begin.")
        self.status.set_xalign(0.0)
        root.append(self.status)

        self._set_nav_sensitive(False)

    def _install_shortcuts(self) -> None:
        controller = Gtk.EventControllerKey()
        controller.connect("key-pressed", self._on_key_pressed)
        self.add_controller(controller)

    def _on_key_pressed(self, _ctrl, keyval, _keycode, state) -> bool:
        ctrl = bool(state & Gdk.ModifierType.CONTROL_MASK)
        if ctrl and keyval == Gdk.KEY_s:
            self.save()
            return True
        if ctrl and keyval == Gdk.KEY_o:
            self._on_open_clicked()
            return True
        if keyval in (Gdk.KEY_Page_Up, Gdk.KEY_KP_Page_Up):
            self.goto(self.index - 1)
            return True
        if keyval in (Gdk.KEY_Page_Down, Gdk.KEY_KP_Page_Down):
            self.goto(self.index + 1)
            return True
        return False

    def _set_nav_sensitive(self, enabled: bool) -> None:
        self.save_btn.set_sensitive(enabled)
        self.prev_btn.set_sensitive(enabled)
        self.next_btn.set_sensitive(enabled)
        self.goto_entry.set_sensitive(enabled)
        self.gt_entry.set_sensitive(enabled)

    def _on_goto_activate(self, *_args) -> None:
        raw = self.goto_entry.get_text().strip()
        if not raw or not self.entries:
            return
        try:
            n = int(raw)
        except ValueError:
            self._set_status("Go expects a 1-based index", error=True)
            return
        self.goto(n - 1)

    def _on_close_request(self, *_args) -> bool:
        if not self.is_dirty():
            return False  # allow close
        if self._confirm_discard():
            return False
        return True  # block close

    def _on_open_clicked(self, *_args) -> None:
        dialog = Gtk.FileDialog(title="Open sample directory")
        dialog.select_folder(self, None, self._on_folder_selected)

    def _on_folder_selected(self, dialog: Gtk.FileDialog, result) -> None:
        try:
            file = dialog.select_folder_finish(result)
        except GLib.Error:
            return
        if file is None:
            return
        path = Path(file.get_path())
        self.open_directory(path)

    def open_directory(self, directory: Path) -> bool:
        directory = directory.resolve()
        if not directory.is_dir():
            self._set_status(f"Not a directory: {directory}", error=True)
            return False

        label_path = find_label_path(directory)
        if label_path is None:
            self._set_status(
                f"No label.json / labels.json in {directory}",
                error=True,
            )
            return False

        if self.is_dirty() and not self._confirm_discard():
            return False

        try:
            labels = load_labels(label_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            self._set_status(f"Failed to load labels: {exc}", error=True)
            return False

        entries = discover_entries(directory, labels)
        if not entries:
            self._set_status(f"No images found in {directory}", error=True)
            return False

        self.directory = directory
        self.label_path = label_path
        self.labels = labels
        self.saved_labels = dict(labels)
        self.entries = entries
        self.index = 0
        self._set_nav_sensitive(True)
        self.set_title(f"HWDB GT Editor — {directory.name}")
        self.show_current()
        self._set_status(f"Loaded {len(entries)} samples from {label_path.name}")
        return False  # for GLib.idle_add

    def is_dirty(self) -> bool:
        return self.labels != self.saved_labels

    def _confirm_discard(self) -> bool:
        try:
            dialog = Gtk.AlertDialog(
                message="Unsaved changes",
                detail="Discard unsaved GT edits?",
                buttons=["Cancel", "Discard"],
                cancel_button=0,
                default_button=1,
            )
        except (AttributeError, TypeError):
            return True

        chosen = {"ok": True}

        def on_choose(_d, result):
            try:
                idx = dialog.choose_finish(result)
                chosen["ok"] = idx == 1
            except GLib.Error:
                chosen["ok"] = False
            loop.quit()

        loop = GLib.MainLoop()
        dialog.choose(self, None, on_choose)
        loop.run()
        return chosen["ok"]

    def show_current(self) -> None:
        if not self.entries or self.directory is None:
            return

        name = self.entries[self.index]
        image_path = self.directory / name
        text = self.labels.get(name, "")

        self._loading = True
        self.gt_entry.set_text(text)
        self._loading = False
        self._update_dirty_ui()

        self.pos_label.set_text(f"{self.index + 1} / {len(self.entries)}")
        self.file_label.set_text(name)

        if image_path.is_file():
            self.image.set_filename(str(image_path))
        else:
            self.image.set_paintable(None)
            self._set_status(f"Missing image file: {name}", error=True)

        self.prev_btn.set_sensitive(self.index > 0)
        self.next_btn.set_sensitive(self.index < len(self.entries) - 1)
        self.gt_entry.grab_focus()
        self.gt_entry.set_position(-1)

    def _on_gt_changed(self, *_args) -> None:
        if self._loading or not self.entries:
            return
        name = self.entries[self.index]
        self.labels[name] = self.gt_entry.get_text()
        self._update_dirty_ui()

    def _update_dirty_ui(self) -> None:
        dirty = self.is_dirty()
        self.dirty_label.set_text("● modified" if dirty else "")
        base = (
            f"HWDB GT Editor — {self.directory.name}"
            if self.directory is not None
            else "HWDB GT Editor"
        )
        self.set_title(f"*{base}" if dirty else base)

    def goto(self, index: int) -> None:
        if not self.entries:
            return
        if index < 0 or index >= len(self.entries):
            return
        # Flush current entry text, keep all edits in memory until Save.
        if self.entries:
            self.labels[self.entries[self.index]] = self.gt_entry.get_text()
        self.index = index
        self.show_current()

    def save(self) -> bool:
        if self.label_path is None or not self.entries:
            return False
        name = self.entries[self.index]
        self.labels[name] = self.gt_entry.get_text()
        try:
            save_labels(self.label_path, self.labels)
        except OSError as exc:
            self._set_status(f"Save failed: {exc}", error=True)
            return False
        self.saved_labels = dict(self.labels)
        self._update_dirty_ui()
        self._set_status(f"Saved → {self.label_path}")
        return True

    def save_and_next(self) -> None:
        if self.save() and self.index < len(self.entries) - 1:
            self.goto(self.index + 1)

    def _set_status(self, message: str, error: bool = False) -> None:
        self.status.set_text(message)
        if error:
            self.status.add_css_class("error")
        else:
            self.status.remove_css_class("error")


class GtEditorApp(Gtk.Application):
    def __init__(self, initial_dir: Path | None) -> None:
        super().__init__(application_id="dev.viehandwritten.gt-editor")
        self.initial_dir = initial_dir

    def do_activate(self) -> None:
        win = self.props.active_window
        if win is None:
            win = GtEditor(self, self.initial_dir)
        win.present()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edit HWDB_line ground truth in-place")
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        help="Sample directory containing images + label.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    app = GtEditorApp(args.directory)
    # Don't pass dataset path into GApplication argv parsing.
    return app.run([sys.argv[0]])


if __name__ == "__main__":
    raise SystemExit(main())
