"""In-memory extraction records (export schema)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FieldValue:
    label: str
    value: str = ""


@dataclass
class Record:
    filename: str
    fields: list[FieldValue] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "fields": [{"label": f.label, "value": f.value} for f in self.fields],
        }


@dataclass
class ExtractionResult:
    """Session-only batch result matching the export JSON schema."""

    records: list[Record] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"records": [r.to_dict() for r in self.records]}

    def clear(self) -> None:
        self.records.clear()
