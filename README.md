# Vietnamese Handwritten OCR (CRNN + CTC)

Nhận dạng chữ viết tay tiếng Việt với pipeline:

```
image → preprocess → CNN → BiLSTM → Linear → CTC
```

## Stack

| Thành phần | Thư viện |
|---|---|
| Deep learning | TensorFlow / Keras |
| Ảnh (OpenCV) | `opencv-python` — CLAHE, adaptive threshold, morphology |
| Ảnh (scikit-image) | deskew / resize / exposure |
| Metrics / split | scikit-learn, editdistance |

## Kiến trúc (CRNN)

Tham khảo các hệ HTR tiếng Việt (Cinnamon / CRNN+CTC) và best practices HTR:

1. **Preprocess** — grayscale, CLAHE, (tuỳ chọn) adaptive threshold + morphology, giữ tỷ lệ khung hình, pad theo batch, chuẩn hoá `[0, 1]`
2. **CNN** — trích xuất feature map theo chiều ngang (sequence)
3. **BiLSTM** — mô hình ngữ cảnh trái↔phải trên chuỗi feature
4. **Linear (Dense)** — logits theo từng timestep, kích thước = `|charset| + blank`
5. **CTC** — `tf.nn.ctc_loss` khi train; greedy / beam decode khi infer

## Dữ liệu training (HWDB — split chính thức theo người viết)

Layout tại `data/images/`, mỗi writer folder có 1 file `label.json` (`filename → text`):

```
data/images/
  HWDB_line/{train_data,test_data}/<writer_id>/   # ảnh DÒNG chữ
    1.jpg, 2.jpg, ...
    label.json          # {"1.jpg": "văn bản dòng", ...}
  HWDB_word/{train_data,test_data}/<writer_id>/   # ảnh TỪ đơn
  HWDB_paragraph/...                              # ảnh nguyên trang (chưa dùng)
```

- `line` và `word` là cùng corpus ở 2 mức chi tiết → đều feed thẳng CRNN+CTC (~120k ảnh).
- `paragraph` là ảnh đa dòng, **chưa dùng** để train CRNN.
- Charset phủ ~100% nhãn (chỉ 1 ký tự OOV bị loại).

### Chuẩn hoá thành manifest

Sinh manifest JSONL chuẩn hoá (unify `label.json`, split theo writer, lọc OOV):

```bash
python main.py build-data --config configs/default.yaml
# → data/manifests/{train,val,test}.jsonl + summary.json
```

- `test` = `test_data` chính thức; `val` = ~10% *writers* tách từ `train_data`
  (writer-independent, không rò rỉ người viết); còn lại là `train`.
- Cấu hình ở `data.*` trong `configs/default.yaml` (`sources`, `val_writers_ratio`,
  `source_weights`, ...). Train tự build manifest nếu chưa có (`--rebuild-data` để build lại).

## Cấu trúc thư mục

```
configs/          # YAML cấu hình train / model / data
data/
  vn_handwritten_images/   # ảnh + labels.json (training)
  charset/                 # bảng ký tự tiếng Việt
  processed/               # dữ liệu đã chuẩn hoá (optional)
src/vie_handwritten/
  preprocess.py   # OpenCV + scikit-image
  model.py        # CNN → BiLSTM → Linear
  ctc.py          # loss + decode
  dataset.py      # đọc labels.json + load ảnh
  postprocess.py  # decode → chuỗi tiếng Việt
  pipeline.py     # end-to-end infer
  metrics.py      # CER / WER
main.py           # CLI: build-data / train / evaluate / infer
notebooks/        # thí nghiệm
checkpoints/      # weights
tests/
```

## Cài đặt

```bash
# với uv (khuyến nghị)
uv sync

# hoặc pip
pip install -e ".[dev]"
```

## Chạy

```bash
python main.py build-data --config configs/default.yaml            # sinh manifest
python main.py train --config configs/default.yaml                 # train 2 pha A→B
python main.py evaluate --checkpoint checkpoints/best.weights.h5 --split test --source line
python main.py infer --image path/to/line.png --checkpoint checkpoints/best.weights.h5
```

### Hai pha huấn luyện

- **Pha A (warmup/head)** — đóng băng backbone ImageNet, chỉ train BiLSTM+Dense,
  LR `1e-3`, nghiêng nguồn `word` (0.7/0.3) để học nhanh ánh xạ đặc trưng→ký tự.
- **Pha B (finetune)** — mở băng từ `layer3` (giữ stem/layer1/layer2), LR `1e-4`,
  nghiêng nguồn `line` (0.85/0.15) để chuyên hoá đọc dòng.

Trọng số trộn nguồn đặt ở `train.phases[*].source_weights`; mỗi epoch chạy
`train.steps_per_epoch` bước (dataset trộn nguồn lặp vô hạn).

## Trạng thái

Skeleton / stubs only — các hàm raise `NotImplementedError`. Implement theo thứ tự gợi ý:

1. `charset` + `preprocess`
2. `dataset` (ảnh dòng chữ + nhãn)
3. `model` + `ctc`
4. `train` / `metrics`
5. `pipeline` + `infer`
