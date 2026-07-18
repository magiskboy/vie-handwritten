# Vietnamese Handwritten OCR

Line-level OCR for Vietnamese handwriting using a CRNN + CTC pipeline:

```
image → preprocess → ResNet-18 → BiLSTM → Linear → CTC
```

Train with TensorFlow/Keras, decode with optional KenLM-fused beam search, and deploy on CPU via OpenVINO. A GTK4 desktop viewer and a form-field extraction demo are included.

## Features

- **CRNN + CTC** — ResNet-18 backbone (ImageNet), BiLSTM context encoder, CTC training/decoding
- **LM-fused decoding** — KenLM syllable n-gram + `pyctcdecode` (`greedy` / `beam` / `beam_lm`)
- **OpenVINO export** — FP16 / INT8 IR for TF-free CPU inference



## Stack


| Component     | Library                                                   |
| ------------- | --------------------------------------------------------- |
| Deep learning | TensorFlow / Keras (ResNet-18 + vendored ImageNet weights) |
| Images        | `opencv-python` (CLAHE), `scikit-image` (deskew / resize) |
| Metrics       | `editdistance` (CER / WER)                                |
| LM decode     | KenLM + `pyctcdecode`                                     |
| Deploy        | OpenVINO (+ NNCF for INT8)                                |
| GUI           | GTK4 + libadwaita (PyGObject)                             |




## Architecture

1. **Preprocess** — grayscale, CLAHE, deskew, aspect-preserving resize (height=64), batch pad, ImageNet normalize
2. **ResNet-18** — HTR backbone (width-preserving strides, ~1/8 downsample)
3. **BiLSTM** — left↔right context over the feature sequence
4. **Linear (Dense)** — per-timestep logits of size `|charset|` (includes blank)
5. **CTC** — `tf.nn.ctc_loss` at train time; greedy / beam / beam_lm at inference



## Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Optional: CUDA for GPU training; Boost + CMake for building KenLM tools; GTK4 for the GUI



## Installation

```bash
git clone https://github.com/magiskboy/vie-handwritten.git
cd vie-handwritten
make sync          # uv sync
# or: pip install -e ".[dev]"
```

Vendored third-party trees (committed in-repo, not submodules):

- `third_party/kenlm/` — [kpu/kenlm](https://github.com/kpu/kenlm) source; see `SOURCE.md`
- `third_party/resnet_18_imagenet/` — ImageNet ResNet-18 backbone weights; see `SOURCE.md`

Extras:

```bash
make sync-gui      # GTK viewer (pygobject)
make sync-ov       # OpenVINO + NNCF
make sync-demo     # form-ocr demo workspace package
```



## Quick start

```bash
make help                                    # list all targets
make train                                   # → checkpoints/<name>/
make evaluate CKPT=checkpoints/<name> SPLIT=test
make infer IMAGE=path/to/line.png CKPT=checkpoints/<name>
```

Or call the CLI directly:

```bash
vie-ocr train --config configs/default.yaml
vie-ocr evaluate --checkpoint checkpoints/<name> --split test
vie-ocr infer --image path/to/line.png --checkpoint checkpoints/<name>
# equivalent: uv run python -m vie_handwritten.cli <command>
```



### Checkpoint layout

Checkpoints are **self-contained directories**. Evaluate, infer, and the GUI load only from this folder — not from `configs/*.yaml`.

```
checkpoints/<name>/
  model.weights.h5
  config.yaml          # paths rewritten relative to this dir
  charset.txt
  build_info.yaml
  lm/                  # copied when LM files exist
    vi.binary
    unigrams.txt
    vi_syllables.txt
```

OpenVINO artifacts (after `make convert-ov`) live under `<checkpoint>/openvino/` with `meta.yaml` and IR variants (`fp16_b*/`, `int8_b*/`). Default decode for OV is `beam_lm`.

## Dataset

Expected HWDB_line layout (official **writer** splits):

```
data/images/HWDB_line/
  train_data/<writer_id>/{1.jpg, 2.jpg, ..., label.json}
  val_data/<writer_id>/...     # held-out writers (writer-independent)
  test_data/<writer_id>/...
```

The pipeline discovers samples from these three directories. Set `drop_oov: true` to skip labels with characters outside the charset.

## Training



### Two-phase schedule


| Phase | What trains                 | Data                                            | LR     |
| ----- | --------------------------- | ----------------------------------------------- | ------ |
| **1** | BiLSTM + Dense (CNN frozen) | small subset (`train.phase1.max_train_samples`) | `1e-3` |
| **2** | Full model                  | full train set                                  | `1e-4` |


Both phases write into the same checkpoint directory. `model.weights.h5` keeps the best `val_loss`, along with `config.yaml`, `charset.txt`, `build_info.yaml`, and `lm/`.

Curriculum (word → line) is also available:

```bash
make curriculum    # train-word then train-line
```



### Sanity-check: overfit a tiny set

Before a full run, confirm the model can converge (loss and CER → ~0 on a small in-distribution subset):

```bash
make train CONFIG=configs/debug.yaml   # overfit 32 samples
tensorboard --logdir runs/debug
```

`configs/debug.yaml` disables dropout / early-stopping / LR reduction and uses exactly 32 samples for train and decode eval.

## Post-processing (no retraining)

Applied after the model emits logits:

1. **Beam search + KenLM** (`ctc.decode: beam_lm`) — syllable n-gram LM + Vietnamese syllable lexicon via `pyctcdecode` (helps with tone marks and near-confusable characters).
2. **Text normalization** (`postprocess`) — [Underthesea](https://github.com/undertheseanlp/underthesea) `text_normalize` by default (NFC, closed-syllable tone placement, `Ð/Đ`, etc.), then punctuation spacing cleanup. Disable with `postprocess.underthesea: false` for a local open-syllable fallback.



### Build KenLM tools

KenLM needs **Boost** and **CMake** from your package manager (not pip):

```bash
# Fedora
sudo dnf install cmake boost-devel zlib-devel bzip2-devel xz-devel
# Debian/Ubuntu
sudo apt install cmake libboost-all-dev zlib1g-dev libbz2-dev liblzma-dev

make build-kenlm    # → third_party/kenlm/build/bin/{lmplz,build_binary}
```

Python `kenlm` bindings and `pyctcdecode` are installed by `make sync` (bindings build from the vendored `third_party/kenlm` tree).

### Train LM and evaluate

```bash
make build-lm
make evaluate CKPT=checkpoints/<name> SPLIT=test DECODE=greedy
make evaluate CKPT=checkpoints/<name> SPLIT=test DECODE=beam_lm
```

Set `ctc.decode: beam_lm` in `configs/default.yaml` so it is baked into the checkpoint `config.yaml`. LM knobs (`alpha`, `beta`, `beam_width`, `token_min_logp`, …) live under `ctc`.

Tune on validation:

```bash
make tune-lm CKPT=checkpoints/<name> \
  ALPHAS=0.0,0.3,0.5,0.8,1.0 BETAS=0.0,0.5,1.0,1.5
```

Logits are cached once; the grid prints CER/WER and the best settings to copy into config.

## OpenVINO deploy

```bash
make convert-ov CKPT=checkpoints/<name>
make bench-ov-acc CKPT=checkpoints/<name> SPLIT=test
make bench-ov-perf CKPT=checkpoints/<name>
```

## Contributing

Issues and pull requests are welcome. For local development:

```bash
make sync
uv run pytest
uv run ruff check src
```

Please keep changes focused, match existing style, and document new CLI flags or config keys in this README when relevant.

## License

License information will be added to the repository when published. Until then, all rights reserved by the author(s).