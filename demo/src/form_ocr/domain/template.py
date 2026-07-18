"""In-memory form template and field definitions."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FieldDef:
    """One labeled ROI on the template image.

    ``bbox`` is normalized ``(x, y, w, h)`` in ``[0, 1]`` relative to the
    template (and scan) image size.
    """

    label: str
    bbox: tuple[float, float, float, float]
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass
class FormTemplate:
    """Session-only template: image path + field ROIs (never persisted)."""

    image_path: Path | None = None
    image_width: int = 0
    image_height: int = 0
    fields: list[FieldDef] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return (
            self.image_path is not None
            and self.image_width > 0
            and self.image_height > 0
            and len(self.fields) > 0
        )

    def labels(self) -> list[str]:
        return [f.label for f in self.fields]

    def add_field(self, label: str, bbox: tuple[float, float, float, float]) -> FieldDef:
        label = label.strip()
        if not label:
            raise ValueError("Label không được để trống")
        if any(f.label == label for f in self.fields):
            raise ValueError(f"Label đã tồn tại: {label}")
        fd = FieldDef(label=label, bbox=bbox)
        self.fields.append(fd)
        return fd

    def remove_field(self, field_id: str) -> None:
        self.fields = [f for f in self.fields if f.id != field_id]

    def update_label(self, field_id: str, label: str) -> None:
        label = label.strip()
        if not label:
            raise ValueError("Label không được để trống")
        for f in self.fields:
            if f.id != field_id and f.label == label:
                raise ValueError(f"Label đã tồn tại: {label}")
        for f in self.fields:
            if f.id == field_id:
                f.label = label
                return
        raise KeyError(field_id)

    def clear(self) -> None:
        self.image_path = None
        self.image_width = 0
        self.image_height = 0
        self.fields.clear()
