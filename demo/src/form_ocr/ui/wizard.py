"""Three-step ViewStack: Template | Batch | Results."""

from __future__ import annotations

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gtk  # noqa: E402

from form_ocr.ui.batch_view import BatchView
from form_ocr.ui.canvas import RoiCanvas
from form_ocr.ui.field_list import FieldList
from form_ocr.ui.results_view import ResultsView


class Wizard(Gtk.Box):
    """Adw.ViewStack with Template / Batch / Results pages."""

    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.set_hexpand(True)
        self.set_vexpand(True)

        self.stack = Adw.ViewStack(hexpand=True, vexpand=True)
        self.switcher = Adw.ViewSwitcher(stack=self.stack, policy=Adw.ViewSwitcherPolicy.WIDE)

        # --- Template page ---
        self.canvas = RoiCanvas()
        self.field_list = FieldList()
        split = Adw.NavigationSplitView()
        sidebar = Adw.NavigationPage(title="Fields", child=self.field_list)
        content = Adw.NavigationPage(title="Template", child=self.canvas)
        split.set_sidebar(sidebar)
        split.set_content(content)
        split.set_min_sidebar_width(220)
        split.set_max_sidebar_width(320)
        split.set_sidebar_width_fraction(0.22)
        self.stack.add_titled_with_icon(split, "template", "Template", "document-edit-symbolic")

        # --- Batch page ---
        self.batch = BatchView()
        self.stack.add_titled_with_icon(self.batch, "batch", "Batch", "folder-symbolic")

        # --- Results page ---
        self.results = ResultsView()
        self.stack.add_titled_with_icon(
            self.results, "results", "Results", "x-office-spreadsheet-symbolic"
        )

        self.append(self.stack)

    def show_page(self, name: str) -> None:
        page = self.stack.get_child_by_name(name)
        if page is not None:
            self.stack.set_visible_child(page)
