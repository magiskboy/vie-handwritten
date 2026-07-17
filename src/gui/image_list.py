"""Sidebar image list (Gtk.ListView + Gio.ListStore)."""

from __future__ import annotations

from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gio, GObject, Gtk, Pango  # noqa: E402

from gui.model_service import list_images  # noqa: E402


class ImageItem(GObject.Object):
    """One row in the image list."""

    __gtype_name__ = "VieImageItem"

    path = GObject.Property(type=str, default="")
    name = GObject.Property(type=str, default="")
    recognized = GObject.Property(type=bool, default=False)

    def __init__(self, path: Path, *, recognized: bool = False) -> None:
        super().__init__()
        self.path = str(path)
        self.name = path.name
        self.recognized = recognized


def _bool_to_opacity(_binding: GObject.Binding, value: bool) -> float:
    return 1.0 if value else 0.25


class ImageList(Gtk.Box):
    """Scrollable list of images; emits ``image-selected`` with the file path."""

    __gsignals__ = {
        "image-selected": (GObject.SignalFlags.RUN_FIRST, None, (str,)),
    }

    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.set_vexpand(True)

        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        header.add_css_class("vie-sidebar-header")
        self._title = Gtk.Label(label="Ảnh", xalign=0.0, hexpand=True)
        self._title.add_css_class("heading")
        self._count = Gtk.Label(label="0")
        self._count.add_css_class("dim-label")
        header.append(self._title)
        header.append(self._count)
        self.append(header)

        self._store = Gio.ListStore.new(ImageItem)
        self._selection = Gtk.SingleSelection(model=self._store)
        self._selection.connect("notify::selected", self._on_selected)

        factory = Gtk.SignalListItemFactory()
        factory.connect("setup", self._on_setup)
        factory.connect("bind", self._on_bind)
        factory.connect("unbind", self._on_unbind)

        self._view = Gtk.ListView(
            model=self._selection,
            factory=factory,
            single_click_activate=False,
        )
        self._view.add_css_class("navigation-sidebar")
        self._view.connect("activate", self._on_activate)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_vexpand(True)
        scrolled.set_child(self._view)

        self._stack = Gtk.Stack()
        self._empty = Gtk.Label(
            label="Chưa chọn thư mục ảnh",
            wrap=True,
            justify=Gtk.Justification.CENTER,
            valign=Gtk.Align.CENTER,
            margin_start=16,
            margin_end=16,
        )
        self._empty.add_css_class("dim-label")
        self._stack.add_named(self._empty, "empty")
        self._stack.add_named(scrolled, "list")
        self._stack.set_visible_child_name("empty")
        self.append(self._stack)

    def set_folder(self, folder: str | Path) -> int:
        """Replace the list with images from ``folder``. Returns count."""
        self._store.remove_all()
        paths = list_images(folder)
        for path in paths:
            self._store.append(ImageItem(path))
        self._count.set_label(str(len(paths)))
        if paths:
            self._stack.set_visible_child_name("list")
            self._selection.set_selected(0)
        else:
            self._empty.set_label("Không tìm thấy ảnh trong thư mục")
            self._stack.set_visible_child_name("empty")
        return len(paths)

    def mark_recognized(self, path: str, *, recognized: bool = True) -> None:
        for i in range(self._store.get_n_items()):
            item = self._store.get_item(i)
            if item is not None and item.path == path:
                item.recognized = recognized
                return

    def selected_path(self) -> str | None:
        item = self._selection.get_selected_item()
        return item.path if isinstance(item, ImageItem) else None

    @staticmethod
    def _on_setup(_factory: Gtk.SignalListItemFactory, list_item: Gtk.ListItem) -> None:
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=10)
        box.add_css_class("vie-list-row")
        box.set_margin_start(6)
        box.set_margin_end(10)
        box.set_margin_top(2)
        box.set_margin_bottom(2)

        check = Gtk.Image(icon_name="object-select-symbolic")
        check.set_opacity(0.25)
        check.set_pixel_size(14)

        label = Gtk.Label(xalign=0.0, hexpand=True)
        label.set_ellipsize(Pango.EllipsizeMode.END)

        box.append(check)
        box.append(label)
        list_item.set_child(box)
        list_item._check = check  # type: ignore[attr-defined]
        list_item._label = label  # type: ignore[attr-defined]
        list_item._binding = None  # type: ignore[attr-defined]

    @staticmethod
    def _on_bind(_factory: Gtk.SignalListItemFactory, list_item: Gtk.ListItem) -> None:
        item = list_item.get_item()
        if not isinstance(item, ImageItem):
            return
        list_item._label.set_label(item.name)  # type: ignore[attr-defined]
        list_item._binding = item.bind_property(  # type: ignore[attr-defined]
            "recognized",
            list_item._check,  # type: ignore[attr-defined]
            "opacity",
            GObject.BindingFlags.SYNC_CREATE,
            _bool_to_opacity,
            None,
        )

    @staticmethod
    def _on_unbind(_factory: Gtk.SignalListItemFactory, list_item: Gtk.ListItem) -> None:
        binding = getattr(list_item, "_binding", None)
        if binding is not None:
            binding.unbind()
            list_item._binding = None  # type: ignore[attr-defined]

    def _on_selected(self, selection: Gtk.SingleSelection, _pspec: GObject.ParamSpec) -> None:
        item = selection.get_selected_item()
        if isinstance(item, ImageItem):
            self.emit("image-selected", item.path)

    def _on_activate(self, _view: Gtk.ListView, position: int) -> None:
        item = self._store.get_item(position)
        if isinstance(item, ImageItem):
            self.emit("image-selected", item.path)
