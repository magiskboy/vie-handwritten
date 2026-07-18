# Vietnamese Handwritten OCR (CRNN + CTC)

Nhận dạng chữ viết tay tiếng Việt (ảnh **dòng** chữ) với pipeline:

```
image → preprocess → ResNet-18 → BiLSTM → Linear → CTC
```

## Stack

| Thành phần | Thư viện |
|---|---|
| Deep learning | TensorFlow / Keras, keras-hub (ResNet-18 ImageNet) |
| Ảnh | `opencv-python` (CLAHE), `scikit-image` (deskew / resize) |
| Metrics | `editdistance` (CER / WER) |

## Kiến trúc

1. **Preprocess** — grayscale, CLAHE, deskew, resize giữ tỷ lệ (height=64), pad theo batch, chuẩn hoá ImageNet.
2. **ResNet-18** — backbone HTR (stride giữ chiều rộng, downsample ≈ 1/8).
3. **BiLSTM** — mô hình ngữ cảnh trái↔phải trên chuỗi feature.
4. **Linear (Dense)** — logits mỗi timestep, kích thước = `|charset|` (đã gồm blank).
5. **CTC** — `tf.nn.ctc_loss` khi train; greedy / beam / beam_lm (KenLM) decode khi infer
   (xem mục *Post-process* bên dưới).

## Dữ liệu (HWDB_line — split chính thức theo người viết)

```
data/images/HWDB_line/
  train_data/<writer_id>/{1.jpg, 2.jpg, ..., label.json}
  test_data/<writer_id>/...
```

`build-data` sinh manifest JSONL chuẩn hoá (unify `label.json`, split theo writer, lọc OOV):

```bash
make build-data
# → data/manifests/{train,val,test}.jsonl + summary.json
```

- `test` = `test_data` chính thức; `val` = ~10% *writers* tách từ `train_data` (writer-independent); còn lại là `train`.

## Cấu trúc source

```
Makefile                  # CLI-hoá các tác vụ thường dùng (xem `make help`)
configs/default.yaml      # cấu hình train / build-data (không dùng khi infer)
screenshots/              # ảnh chụp GUI
src/vie_handwritten/
  cli.py          # entry point `vie-ocr` (build-data/train/build-lm/evaluate/infer/tune-lm)
  utils.py        # config I/O, self-contained checkpoint/OV artifact bundles
  charset.py      # bảng ký tự ↔ index
  preprocess.py   # OpenCV + scikit-image (ảnh → tensor)
  postprocess.py  # CTC decode + Underthesea/local chuẩn hoá tiếng Việt; lớp CTCDecoder
  lm_decode.py    # TF-free KenLM + pyctcdecode (dùng chung Keras / OpenVINO)
  dataset.py      # discovery + manifest + tf.data (line only)
  model.py        # ResNet-18 → BiLSTM → Linear + lớp composition OCRModel (net + postprocess)
  trainer.py      # CTC loss + OCRTrainer + DecodeMetrics + train 2 phase
  eval.py         # metrics CER/WER + evaluate/infer (dùng OCRModel)
  kenlm.py        # train KenLM n-gram LM từ transcript tập train
  tune.py         # grid-search alpha/beta trên val
src/gui/                  # GTK4 + libadwaita viewer (`vie-ocr-gui`)
src/converter/            # Keras → OpenVINO IR + TF-free deploy runtime
```

Chạy CLI qua `make <target>`, hoặc trực tiếp `vie-ocr <command>` /
`uv run python -m vie_handwritten.cli <command>`.

## Cài đặt

```bash
make sync          # = uv sync (khuyến nghị)
# hoặc: pip install -e ".[dev]"
```

## Chạy

`make help` liệt kê mọi target. Truyền tham số qua biến, ví dụ
`make evaluate CKPT=... SPLIT=test DECODE=beam_lm`.

```bash
make build-data
make train                                    # → self-contained checkpoints/<name>/
make evaluate CKPT=checkpoints/<name> SPLIT=test
make infer IMAGE=path/to/line.png CKPT=checkpoints/<name>
```

Checkpoint là **thư mục self-contained**:

```
checkpoints/<name>/
  model.weights.h5
  config.yaml          # paths rewritten relative to this dir
  charset.txt
  build_info.yaml
  lm/                  # copied when source LM files exist
    vi.binary
    unigrams.txt
    vi_syllables.txt
```

`evaluate` / `infer` / GUI chỉ load từ thư mục đó — không dùng `configs/*.yaml`.

OpenVINO artifact (sau `vie-ov convert`) tương tự, dưới `<checkpoint>/openvino/` thêm
`meta.yaml` và các IR `fp16_b*/`, `int8_b*/`; decode mặc định là `beam_lm`.

Tương đương khi không dùng make: `vie-ocr build-data`,
`vie-ocr evaluate --checkpoint checkpoints/<name> --split test`, ...

## GUI (GTK4 + libadwaita)

Viewer desktop: chọn thư mục checkpoint self-contained, duyệt folder
ảnh dòng, nhận dạng realtime (beam_lm khi có KenLM trong checkpoint), so sánh Pred ↔ GT nếu folder có
`label.json` (Levenshtein / CER / WER), hiển thị GPU và latency (ms).

```bash
# Fedora: gtk4-devel libadwaita-devel gobject-introspection-devel cairo-gobject-devel …
make sync-gui
make gui            # = uv run vie-ocr-gui
```

Chi tiết: [`src/gui/README.md`](src/gui/README.md).

| Exact match | Exact match |
|:-----------:|:-----------:|
| ![vie-OCR — exact, 608 ms](screenshots/1.png) | ![vie-OCR — exact, 583 ms](screenshots/2.png) |
| ![vie-OCR — exact, 570 ms](screenshots/3.png) | ![vie-OCR — exact, 104 ms](screenshots/4.png) |
| Diff (CER / WER) | Diff (CER / WER) |
| ![vie-OCR — Levenshtein 3](screenshots/5.png) | ![vie-OCR — Levenshtein 2](screenshots/6.png) |

## Hai pha huấn luyện

- **Phase 1** — đóng băng CNN backbone, chỉ train BiLSTM + Dense trên một **tập nhỏ**
  (`train.phase1.max_train_samples`) để head hội tụ nhanh, LR `1e-3`.
- **Phase 2** — mở băng toàn bộ, train cả CNN + BiLSTM trên **toàn bộ** dữ liệu, LR `1e-4`.

Cả hai pha dùng chung một thư mục checkpoint self-contained; `model.weights.h5` theo
`val_loss` thấp nhất, kèm `config.yaml` / `charset.txt` / `build_info.yaml` / `lm/`.
Mỗi pha là một lượt `model.fit` trên dataset hữu hạn (1 epoch = 1 lượt qua dữ liệu).

## Debug quá trình training (overfit tập nhỏ)

Theo Andrew Ng, trước khi train full hãy kiểm tra model có **hội tụ** không: cho model
overfit một tập nhỏ lấy từ **cùng phân bố** với train thật — nếu đúng, cả *loss* và
*error (CER)* phải tiến về ~0. Nếu không → có bug ở model / loss / data pipeline.

```bash
make train CONFIG=configs/debug.yaml   # overfit 32 mẫu
tensorboard --logdir runs/debug
```

`configs/debug.yaml` đã set sẵn để overfit: tắt dropout, tắt early-stopping / giảm LR,
dùng đúng 32 mẫu cho cả train lẫn decode.

## Post-process: LM-fused decoding + chuẩn hoá tiếng Việt

Hai tầng post-process nâng độ chính xác của output, **không cần train lại model**
(chạy sau khi model đã xuất `logits`):

- **Bước 1 — beam search + KenLM language model** (`ctc.decode: beam_lm`): dùng
  `pyctcdecode` fuse một n-gram LM mức âm tiết + lexicon âm tiết tiếng Việt để ưu tiên
  chuỗi hợp lý về ngôn ngữ (sửa lỗi dấu thanh / ký tự gần giống).
- **Bước 2 — chuẩn hoá văn bản** (`postprocess`): mặc định dùng
  [Underthesea](https://github.com/undertheseanlp/underthesea) `text_normalize`
  (NFC, vị trí dấu thanh gồm âm tiết đóng `sóat→soát`, `hoà→hòa`, sửa `Ð/Đ` /
  `lựơng→lượng`), rồi dọn khoảng trắng quanh dấu câu. Áp cho mọi decode method.
  Tắt bằng `postprocess.underthesea: false` để dùng fallback local (chỉ âm tiết mở).

### 1. Cài công cụ build KenLM (qua package manager, KHÔNG dùng pip)

KenLM cần **Boost** + **cmake** (prebuilt từ package manager):

```bash
# Fedora
sudo dnf install cmake boost-devel zlib-devel bzip2-devel xz-devel
# Debian/Ubuntu
sudo apt install cmake libboost-all-dev zlib1g-dev libbz2-dev liblzma-dev
```

### 2. Build KenLM từ submodule

```bash
git submodule update --init --recursive   # nếu chưa có third_party/kenlm
make build-kenlm                           # → third_party/kenlm/build/bin/{lmplz,build_binary}
```

Python binding `kenlm` + `pyctcdecode` đã được `make sync` cài sẵn (kenlm build từ submodule).

### 3. Train LM + bật beam_lm

```bash
make build-lm                              # → lm/vi.binary + lm/unigrams.txt
# So sánh nhanh trên test (override decode method):
make evaluate CKPT=checkpoints SPLIT=test DECODE=greedy
make evaluate CKPT=checkpoints SPLIT=test DECODE=beam_lm
```

Đặt `ctc.decode: beam_lm` trong `configs/default.yaml` (train) để ghi vào
`config.yaml` của checkpoint. Các tham số LM (`alpha`, `beta`, `beam_width`,
`token_min_logp`) nằm trong khối `ctc`.

### 4. Tune trọng số LM (alpha/beta) trên val

```bash
make tune-lm CKPT=checkpoints ALPHAS=0.0,0.3,0.5,0.8,1.0 BETAS=0.0,0.5,1.0,1.5
```

Cache logits một lần rồi quét lưới alpha/beta, in CER/WER và điểm tốt nhất để chép vào config.

Trên TensorBoard theo dõi:

| Signal | Ở đâu | Kỳ vọng khi model đúng |
|---|---|---|
| CTC loss | `epoch_loss` (train), `val_loss` | giảm đều về ~0 |
| Character Error Rate | `train_cer` | → ~0 |
| Word Error Rate | `train_wer` | → ~0 |
| Learning rate | `lr` | đúng như config |
| Dự đoán vs nhãn | tab **TEXT** (`train/pred_vs_true`) | pred trùng dần với ground truth |
| Ảnh input | tab **IMAGES** (`train/inputs`) | đúng ảnh + preprocess hợp lý |

Bộ đo decode này (callback `DecodeMetrics` trong `debug.py`) cũng bật mặc định khi train
thật qua `train.decode_eval_samples` (đặt `0` để tắt) — cho tín hiệu *accuracy* chứ không
chỉ loss. Nếu overfit không về ~0: kiểm tra căn chỉnh nhãn↔ảnh, `input_length` của CTC,
charset, và chiều rộng feature (`widths // 8`).
