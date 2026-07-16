# Vietnamese Handwritten OCR (ResNet + Transformer + CTC)

Nhận dạng chữ viết tay tiếng Việt với pipeline:

```
image → preprocess → CNN → Transformer Encoder → Linear → CTC
```

## Stack

| Thành phần | Thư viện |
|---|---|
| Deep learning | TensorFlow / Keras |
| Ảnh (OpenCV) | `opencv-python` — CLAHE, adaptive threshold, morphology |
| Ảnh (scikit-image) | deskew / resize / exposure |
| Metrics / split | scikit-learn, editdistance |

## Kiến trúc (CRNN)

Tham khảo best practices HTR / text-line recognition (Self-Attention + CTC):

1. **Preprocess** — grayscale, CLAHE, (tuỳ chọn) adaptive threshold + morphology, giữ tỷ lệ khung hình, pad theo batch, chuẩn hoá ImageNet
2. **CNN (ResNet-18)** — trích xuất feature map theo chiều ngang (sequence); ImageNet pretrained
3. **Transformer Encoder** — self-attention trên chuỗi feature (train từ đầu; nhẹ: 4 layers / d=256)
4. **Linear (Dense)** — logits theo từng timestep, kích thước = `|charset| + blank`
5. **CTC** — `tf.nn.ctc_loss` khi train; greedy / beam decode khi infer

## Dữ liệu training

Layout thực tế tại `data/vn_handwritten_images/` (~1838 mẫu địa chỉ viết tay):

```
data/vn_handwritten_images/
  labels.json              # {"1.jpg": "Số 3 Nguyễn Ngọc Vũ, Hà Nội", ...}
  data/
    1.jpg                  # key trong labels.json = tên file trong data/
    0001_samples.png
    ...
```

- `labels.json`: object `filename → transcription` (UTF-8).
- Ảnh: `.png` / `.jpg` / `.jpeg`, cùng thư mục `data/`.
- Config trỏ tới bộ này qua `data.dataset_dir` trong `configs/default.yaml`.

## Cấu trúc thư mục

```
configs/          # YAML cấu hình train / model / data
data/
  vn_handwritten_images/   # ảnh + labels.json (training)
  charset/                 # bảng ký tự tiếng Việt
  processed/               # dữ liệu đã chuẩn hoá (optional)
src/vie_handwritten/
  preprocess.py   # OpenCV + scikit-image
  model.py        # CNN → Transformer Encoder → Linear
  ctc.py          # loss + decode
  dataset.py      # đọc labels.json + load ảnh
  postprocess.py  # decode → chuỗi tiếng Việt
  pipeline.py     # end-to-end infer
  metrics.py      # CER / WER
scripts/          # train / evaluate / infer CLI
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

## Scripts

```bash
python main.py train --config configs/default.yaml
python main.py evaluate --config configs/default.yaml --checkpoint checkpoints/best.weights.h5
python main.py infer --image path/to/line.png --checkpoint checkpoints/best.weights.h5
```
