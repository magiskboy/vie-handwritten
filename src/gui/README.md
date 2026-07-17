# vie-OCR GUI (GTK4 + libadwaita)

Desktop viewer for Vietnamese handwritten OCR checkpoints.

## Setup

System packages (Fedora):

```bash
sudo dnf install gobject-introspection-devel cairo-gobject-devel cairo-devel \
  pkgconf-pkg-config gtk4-devel libadwaita-devel
```

Python deps:

```bash
uv sync --extra gui
```

## Run

```bash
uv run vie-ocr-gui
# or
uv run python -m gui
```

## Usage

1. **Load Model** — pick a checkpoint directory containing `model.weights.h5` + `config.yaml`. Decode defaults to `beam_lm` when `lm/vi.binary` exists; otherwise falls back to `greedy`.
2. **Load Images** — pick a directory of line images (`.jpg`, `.png`, …). If the folder has `label.json`, the UI compares Pred vs GT (Levenshtein / CER / WER).
3. Click an image in the left list to preview and run OCR. Results (and latency) are cached per image.

Screenshots: [`screenshots/`](../../screenshots/) — also embedded in the [root README](../../README.md#gui-gtk4--libadwaita).
