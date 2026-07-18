"""Gtk.DrawingArea canvas for drawing labeled ROI rectangles."""

from __future__ import annotations

import logging

import cairo
import cv2
import gi
import numpy as np

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gdk, GObject, Gtk  # noqa: E402

from form_ocr.domain.template import FieldDef, FormTemplate

logger = logging.getLogger(__name__)


class RoiCanvas(Gtk.Box):
    """Image preview with drag-to-draw ROIs; emits ``selection-changed`` / ``request-label``."""

    __gsignals__ = {
        "field-added": (GObject.SignalFlags.RUN_FIRST, None, (str,)),
        "selection-changed": (GObject.SignalFlags.RUN_FIRST, None, (str,)),
        "request-label": (GObject.SignalFlags.RUN_FIRST, None, (float, float, float, float)),
    }

    def __init__(self) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.set_hexpand(True)
        self.set_vexpand(True)

        self._template: FormTemplate | None = None
        self._surface: cairo.ImageSurface | None = None
        self._img_w = 0
        self._img_h = 0
        self._selected_id: str | None = None

        self._drag_start: tuple[float, float] | None = None
        self._drag_cur: tuple[float, float] | None = None
        self._drew_roi = False

        self._fit_x = 0.0
        self._fit_y = 0.0
        self._fit_w = 0.0
        self._fit_h = 0.0

        # DrawingArea as direct child (not inside ScrolledWindow) so it receives
        # a real allocation for fit-to-view painting.
        self._area = Gtk.DrawingArea(hexpand=True, vexpand=True)
        self._area.set_draw_func(self._on_draw)
        self._area.set_cursor(Gdk.Cursor.new_from_name("crosshair"))
        self._area.add_css_class("form-canvas-area")
        self.append(self._area)

        drag = Gtk.GestureDrag()
        drag.set_button(Gdk.BUTTON_PRIMARY)
        drag.connect("drag-begin", self._on_drag_begin)
        drag.connect("drag-update", self._on_drag_update)
        drag.connect("drag-end", self._on_drag_end)
        self._area.add_controller(drag)

        click = Gtk.GestureClick()
        click.set_button(Gdk.BUTTON_PRIMARY)
        click.connect("released", self._on_click)
        self._area.add_controller(click)

    def set_template(self, template: FormTemplate | None) -> None:
        self._template = template
        self._selected_id = None
        self._surface = None
        self._img_w = self._img_h = 0
        if template and template.image_path:
            bgr = cv2.imread(str(template.image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                logger.error("Cannot load template image: %s", template.image_path)
            else:
                self._img_h, self._img_w = bgr.shape[:2]
                template.image_width = self._img_w
                template.image_height = self._img_h
                self._surface = self._make_surface(bgr)
                logger.info(
                    "Template loaded %s (%dx%d)",
                    template.image_path.name,
                    self._img_w,
                    self._img_h,
                )
        self._area.queue_draw()

    def set_selected(self, field_id: str | None) -> None:
        self._selected_id = field_id
        self._area.queue_draw()

    def refresh(self) -> None:
        self._area.queue_draw()

    def _make_surface(self, bgr: np.ndarray) -> cairo.ImageSurface:
        """BGR uint8 → owned Cairo ARGB32 surface (no create_for_data / setattr)."""
        h, w = bgr.shape[:2]
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        # Little-endian ARGB32 memory layout is B, G, R, A.
        dest = np.ndarray(shape=(h, w, 4), dtype=np.uint8, buffer=surface.get_data())
        dest[:, :, 0] = bgr[:, :, 0]
        dest[:, :, 1] = bgr[:, :, 1]
        dest[:, :, 2] = bgr[:, :, 2]
        dest[:, :, 3] = 255
        surface.mark_dirty()
        return surface

    def _compute_fit(self, area_w: int, area_h: int) -> None:
        if self._img_w <= 0 or self._img_h <= 0 or area_w <= 0 or area_h <= 0:
            self._fit_x = self._fit_y = self._fit_w = self._fit_h = 0.0
            return
        scale = min(area_w / self._img_w, area_h / self._img_h)
        self._fit_w = self._img_w * scale
        self._fit_h = self._img_h * scale
        self._fit_x = (area_w - self._fit_w) / 2
        self._fit_y = (area_h - self._fit_h) / 2

    def _widget_to_norm(self, wx: float, wy: float) -> tuple[float, float] | None:
        if self._fit_w <= 0 or self._fit_h <= 0:
            return None
        nx = (wx - self._fit_x) / self._fit_w
        ny = (wy - self._fit_y) / self._fit_h
        if nx < 0 or ny < 0 or nx > 1 or ny > 1:
            return None
        return nx, ny

    def _norm_rect_to_widget(
        self, bbox: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        x, y, w, h = bbox
        return (
            self._fit_x + x * self._fit_w,
            self._fit_y + y * self._fit_h,
            w * self._fit_w,
            h * self._fit_h,
        )

    def _on_draw(
        self,
        _area: Gtk.DrawingArea,
        cr: cairo.Context,
        width: int,
        height: int,
    ) -> None:
        self._compute_fit(width, height)
        cr.set_source_rgb(0.12, 0.12, 0.14)
        cr.paint()

        if self._surface is not None and self._fit_w > 0 and self._img_w > 0:
            cr.save()
            cr.translate(self._fit_x, self._fit_y)
            scale = self._fit_w / self._img_w
            cr.scale(scale, scale)
            cr.set_source_surface(self._surface, 0, 0)
            cr.paint()
            cr.restore()

        if self._template:
            for fd in self._template.fields:
                self._draw_field(cr, fd, selected=(fd.id == self._selected_id))

        if self._drag_start and self._drag_cur:
            x0, y0 = self._drag_start
            x1, y1 = self._drag_cur
            rx, ry = min(x0, x1), min(y0, y1)
            rw, rh = abs(x1 - x0), abs(y1 - y0)
            cr.set_source_rgba(0.2, 0.7, 1.0, 0.35)
            cr.rectangle(rx, ry, rw, rh)
            cr.fill_preserve()
            cr.set_source_rgba(0.2, 0.7, 1.0, 0.95)
            cr.set_line_width(2.0)
            cr.stroke()

    def _draw_field(self, cr: cairo.Context, fd: FieldDef, *, selected: bool) -> None:
        wx, wy, ww, wh = self._norm_rect_to_widget(fd.bbox)
        if selected:
            cr.set_source_rgba(1.0, 0.55, 0.1, 0.3)
        else:
            cr.set_source_rgba(0.15, 0.85, 0.45, 0.22)
        cr.rectangle(wx, wy, ww, wh)
        cr.fill_preserve()
        if selected:
            cr.set_source_rgba(1.0, 0.55, 0.1, 1.0)
            cr.set_line_width(2.5)
        else:
            cr.set_source_rgba(0.1, 0.7, 0.4, 1.0)
            cr.set_line_width(1.5)
        cr.stroke()

        cr.set_source_rgba(0.05, 0.05, 0.08, 0.75)
        cr.rectangle(wx, max(0, wy - 18), min(ww, 160), 18)
        cr.fill()
        cr.set_source_rgb(1, 1, 1)
        cr.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        cr.set_font_size(12)
        cr.move_to(wx + 4, max(12, wy - 4))
        cr.show_text(fd.label)

    def _on_drag_begin(self, _gesture: Gtk.GestureDrag, x: float, y: float) -> None:
        if self._surface is None:
            return
        if self._widget_to_norm(x, y) is None:
            self._drag_start = None
            return
        self._drag_start = (x, y)
        self._drag_cur = (x, y)

    def _on_drag_update(
        self, gesture: Gtk.GestureDrag, offset_x: float, offset_y: float
    ) -> None:
        if self._drag_start is None:
            return
        ok, sx, sy = gesture.get_start_point()
        if not ok:
            return
        self._drag_cur = (sx + offset_x, sy + offset_y)
        self._area.queue_draw()

    def _on_drag_end(
        self, gesture: Gtk.GestureDrag, offset_x: float, offset_y: float
    ) -> None:
        if self._drag_start is None or self._template is None:
            self._drag_start = self._drag_cur = None
            return
        ok, sx, sy = gesture.get_start_point()
        if not ok:
            self._drag_start = self._drag_cur = None
            return
        ex, ey = sx + offset_x, sy + offset_y
        n0 = self._widget_to_norm(sx, sy)
        n1 = self._widget_to_norm(ex, ey)
        self._drag_start = self._drag_cur = None
        self._area.queue_draw()
        if n0 is None or n1 is None:
            return
        x0, y0 = n0
        x1, y1 = n1
        nx, ny = min(x0, x1), min(y0, y1)
        nw, nh = abs(x1 - x0), abs(y1 - y0)
        if nw < 0.005 or nh < 0.005:
            return
        self._drew_roi = True
        self.emit("request-label", nx, ny, nw, nh)

    def _on_click(self, _gesture: Gtk.GestureClick, _n: int, x: float, y: float) -> None:
        if self._drew_roi:
            self._drew_roi = False
            return
        if self._template is None or self._surface is None:
            return
        n = self._widget_to_norm(x, y)
        if n is None:
            return
        px, py = n
        hit: FieldDef | None = None
        for fd in reversed(self._template.fields):
            bx, by, bw, bh = fd.bbox
            if bx <= px <= bx + bw and by <= py <= by + bh:
                hit = fd
                break
        self._selected_id = hit.id if hit else None
        self.emit("selection-changed", self._selected_id or "")
        self._area.queue_draw()
