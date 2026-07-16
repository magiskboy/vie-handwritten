"""End-to-end inference pipeline: image → text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vie_handwritten.charset import Charset
from vie_handwritten.config import load_config
from vie_handwritten.ctc import decode_predictions
from vie_handwritten.model import build_crnn, load_crnn_weights, pack_crnn_inputs
from vie_handwritten.postprocess import postprocess
from vie_handwritten.preprocess import load_image, preprocess
from vie_handwritten.utils import project_root


class OCRPipeline:
    """Wire preprocess → CRNN → CTC decode → postprocess."""

    def __init__(self, model: Any, charset: Any, config: dict[str, Any]) -> None:
        self.model = model
        self.charset = charset
        self.config = config

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        config_path: str | Path,
    ) -> OCRPipeline:
        """Load model weights + config + charset from disk."""
        from vie_handwritten.utils import configure_runtime

        configure_runtime()
        config = load_config(config_path)
        root = project_root()
        charset_path = Path(config["data"]["charset_path"])
        if not charset_path.is_absolute():
            charset_path = root / charset_path
        charset = Charset.from_file(charset_path)
        model = build_crnn(config, num_classes=charset.num_classes)
        load_crnn_weights(model, checkpoint)
        return cls(model, charset, config)

    def predict_image(self, image: Any) -> str:
        """Run OCR on an in-memory image array."""
        arr = preprocess(image, self.config["preprocess"])
        batch = np.expand_dims(arr, axis=0)
        inputs = pack_crnn_inputs(batch)
        logits = self.model.predict(inputs, verbose=0)
        ctc_cfg = self.config.get("ctc", {})
        texts = decode_predictions(
            logits,
            self.charset,
            method=ctc_cfg.get("decode", "greedy"),
            blank_index=int(ctc_cfg.get("blank_index", 0)),
            beam_width=int(ctc_cfg.get("beam_width", 10)),
        )
        return postprocess(texts[0])

    def predict_path(self, path: str | Path) -> str:
        """Load an image from disk and run OCR."""
        image = load_image(str(path))
        return self.predict_image(image)

    def predict_batch(self, images: list) -> list[str]:
        """Batched inference (per-image preprocess; pad to max width)."""
        processed = [preprocess(img, self.config["preprocess"]) for img in images]
        max_w = max(p.shape[1] for p in processed)
        batch = []
        lengths = []
        for p in processed:
            h, w, c = p.shape
            lengths.append(max(1, w // 8))
            if w < max_w:
                canvas = np.zeros((h, max_w, c), dtype=np.float32)
                canvas[:, :w] = p
                batch.append(canvas)
            else:
                batch.append(p)
        inputs = pack_crnn_inputs(
            np.stack(batch, axis=0),
            input_length=np.asarray(lengths, dtype=np.int32),
        )
        logits = self.model.predict(inputs, verbose=0)
        ctc_cfg = self.config.get("ctc", {})
        texts = decode_predictions(
            logits,
            self.charset,
            method=ctc_cfg.get("decode", "greedy"),
            blank_index=int(ctc_cfg.get("blank_index", 0)),
            beam_width=int(ctc_cfg.get("beam_width", 10)),
        )
        return [postprocess(t) for t in texts]
