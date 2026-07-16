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
5. **CTC** — `tf.nn.ctc_loss` khi train; greedy / beam decode khi infer.

## Dữ liệu (HWDB_line — split chính thức theo người viết)

```
data/images/HWDB_line/
  train_data/<writer_id>/{1.jpg, 2.jpg, ..., label.json}
  test_data/<writer_id>/...
```

`build-data` sinh manifest JSONL chuẩn hoá (unify `label.json`, split theo writer, lọc OOV):

```bash
python main.py build-data --config configs/default.yaml
# → data/manifests/{train,val,test}.jsonl + summary.json
```

- `test` = `test_data` chính thức; `val` = ~10% *writers* tách từ `train_data` (writer-independent); còn lại là `train`.

## Cấu trúc source

```
configs/default.yaml       # toàn bộ cấu hình
src/vie_handwritten/
  utils.py        # config I/O, seed, paths, GPU runtime
  charset.py      # bảng ký tự ↔ index
  preprocess.py   # OpenCV + scikit-image
  dataset.py      # discovery + manifest + tf.data (line only)
  model.py        # ResNet-18 → BiLSTM → Linear, CTCTrainer
  ctc.py          # CTC loss + greedy/beam decode
  train.py        # train 2 phase
  evaluate.py     # metrics + postprocess + CER/WER + infer
main.py           # CLI: build-data / train / evaluate / infer
```

## Cài đặt

```bash
uv sync            # khuyến nghị
# hoặc: pip install -e ".[dev]"
```

## Chạy

```bash
python main.py build-data --config configs/default.yaml
python main.py train --config configs/default.yaml
python main.py evaluate --checkpoint checkpoints/best.weights.h5 --split test
python main.py infer --image path/to/line.png --checkpoint checkpoints/best.weights.h5
```

## Hai pha huấn luyện

- **Phase 1** — đóng băng CNN backbone, chỉ train BiLSTM + Dense trên một **tập nhỏ**
  (`train.phase1.max_train_samples`) để head hội tụ nhanh, LR `1e-3`.
- **Phase 2** — mở băng toàn bộ, train cả CNN + BiLSTM trên **toàn bộ** dữ liệu, LR `1e-4`.

Cả hai pha dùng chung `checkpoints/best.weights.h5` (theo `val_loss` thấp nhất).
Mỗi pha là một lượt `model.fit` trên dataset hữu hạn (1 epoch = 1 lượt qua dữ liệu).

## Debug quá trình training (overfit tập nhỏ)

Theo Andrew Ng, trước khi train full hãy kiểm tra model có **hội tụ** không: cho model
overfit một tập nhỏ lấy từ **cùng phân bố** với train thật — nếu đúng, cả *loss* và
*error (CER)* phải tiến về ~0. Nếu không → có bug ở model / loss / data pipeline.

```bash
python main.py train --config configs/debug.yaml   # overfit 32 mẫu
tensorboard --logdir runs/debug
```

`configs/debug.yaml` đã set sẵn để overfit: tắt dropout, tắt early-stopping / giảm LR,
dùng đúng 32 mẫu cho cả train lẫn decode.

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
