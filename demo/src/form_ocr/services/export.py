"""Export extraction records to CSV / Excel."""

from __future__ import annotations

import csv
from pathlib import Path

from form_ocr.domain.records import ExtractionResult


def _labels(result: ExtractionResult) -> list[str]:
    seen: list[str] = []
    for rec in result.records:
        for fv in rec.fields:
            if fv.label not in seen:
                seen.append(fv.label)
    return seen


def _rows(result: ExtractionResult, labels: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for rec in result.records:
        by_label = {fv.label: fv.value for fv in rec.fields}
        rows.append([rec.filename] + [by_label.get(lab, "") for lab in labels])
    return rows


def export_csv(result: ExtractionResult, path: str | Path) -> Path:
    out = Path(path)
    labels = _labels(result)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", *labels])
        writer.writerows(_rows(result, labels))
    return out


def export_excel(result: ExtractionResult, path: str | Path) -> Path:
    from openpyxl import Workbook

    out = Path(path)
    labels = _labels(result)
    wb = Workbook()
    ws = wb.active
    ws.title = "records"
    ws.append(["filename", *labels])
    for row in _rows(result, labels):
        ws.append(row)
    wb.save(out)
    return out
