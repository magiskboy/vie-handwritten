# Form OCR Demo

GTK4 + libadwaita app: annotate field ROIs on a blank form template, batch-OCR
handwritten values from scanned filled forms (OpenVINO), export CSV/Excel.

**Session is in-memory only** — closing the app discards template, fields, and
records. The only disk write is an explicit Export.

## Setup

System packages (Fedora):

```bash
sudo dnf install gobject-introspection-devel cairo-gobject-devel cairo-devel \
  pkgconf-pkg-config gtk4-devel libadwaita-devel
```

From the repo root:

```bash
# Install demo + OpenVINO deps, convert checkpoint, vendor IR into demo/models/
make demo-ov CKPT=checkpoints/16072026

# Or stepwise:
make sync-demo
make convert-ov CKPT=checkpoints/16072026 BATCHES=1
make demo-vendor CKPT=checkpoints/16072026
```

## Run

```bash
make demo
# or
uv run --package form-ocr-demo form-ocr
```

## Usage

1. **Template** — Load a blank form image. Drag rectangles on the image; name each field.
2. **Batch** — Pick a folder (or files) of scanned filled forms. Click **Extract**.
3. **Results** — Review/edit values; **Export CSV** or **Export Excel**.

Model default path: `demo/models/openvino` (INT8 batch-1, falls back to FP16).
