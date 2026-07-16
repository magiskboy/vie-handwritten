"""Training debug tools (Andrew Ng style "overfit a small set" sanity check).

The core signal is *convergence of error*, not just loss: on a tiny slice of the
real train distribution, both CTC loss and CER should fall toward 0. This module
adds a TensorBoard callback that, each epoch, decodes a fixed sample and logs:

- ``train_cer`` / ``train_wer`` scalars (injected into ``logs`` → picked up by the
  Keras ``TensorBoard`` callback),
- ``lr`` (current learning rate),
- a ``pred vs. true`` text table and the preprocessed input images,

so you can *watch* the model's outputs converge to the labels. If they don't, the
bug is in the model / loss / data pipeline — not in the amount of data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from vie_handwritten.dataset import resolve_image_path
from vie_handwritten.evaluate import evaluate_split, predict_image_array
from vie_handwritten.preprocess import load_image, preprocess

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _denormalize(arr: np.ndarray, preprocess_cfg: dict[str, Any]) -> np.ndarray:
    """Undo normalization so the preprocessed tensor is viewable in [0, 1]."""
    x = arr.astype(np.float32)
    if preprocess_cfg.get("normalize") == "imagenet" and x.shape[-1] == 3:
        x = x * _IMAGENET_STD + _IMAGENET_MEAN
    return np.clip(x, 0.0, 1.0)


def _stack_previews(arrs: list[np.ndarray], preprocess_cfg: dict[str, Any]) -> np.ndarray | None:
    """Right-pad variable-width previews to a common width and stack for tf.summary.image."""
    if not arrs:
        return None
    max_w = max(a.shape[1] for a in arrs)
    out = []
    for a in arrs:
        d = _denormalize(a, preprocess_cfg)
        if d.shape[1] < max_w:
            pad = np.ones((d.shape[0], max_w - d.shape[1], d.shape[2]), dtype=np.float32)
            d = np.concatenate([d, pad], axis=1)
        out.append(d)
    return np.stack(out).astype(np.float32)


class DecodeMetrics(keras.callbacks.Callback):
    """Decode a fixed sample each epoch → log CER/WER + previews to TensorBoard."""

    def __init__(
        self,
        crnn: keras.Model,
        records: list[dict[str, str]],
        charset: Any,
        config: dict[str, Any],
        *,
        tag: str,
        preview_dir: str | Path,
        every: int = 1,
        num_previews: int = 6,
    ):
        super().__init__()
        self.crnn = crnn
        self.records = records
        self.charset = charset
        self.config = config
        self.tag = tag
        self.every = max(1, int(every))
        self.num_previews = num_previews
        self._writer = tf.summary.create_file_writer(str(preview_dir))

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs if logs is not None else {}

        # Always expose the current learning rate as a scalar.
        lr = self.model.optimizer.learning_rate
        logs["lr"] = float(lr.numpy()) if hasattr(lr, "numpy") else float(lr)

        if (epoch + 1) % self.every != 0:
            return

        metrics = evaluate_split(self.crnn, self.records, self.charset, self.config)
        logs[f"{self.tag}_cer"] = metrics["cer"]
        logs[f"{self.tag}_wer"] = metrics["wer"]

        rows, previews = [], []
        for rec in self.records[: self.num_previews]:
            image = load_image(str(resolve_image_path(self.config, rec)))
            previews.append(preprocess(image, self.config["preprocess"]))
            pred = predict_image_array(self.crnn, image, self.charset, self.config)
            rows.append(f"| `{rec['text']}` | `{pred}` |")

        with self._writer.as_default():
            table = "| ground truth | prediction |\n|---|---|\n" + "\n".join(rows)
            tf.summary.text(f"{self.tag}/pred_vs_true", table, step=epoch)
            imgs = _stack_previews(previews, self.config["preprocess"])
            if imgs is not None:
                tf.summary.image(f"{self.tag}/inputs", imgs, step=epoch, max_outputs=len(previews))
            self._writer.flush()

        logger.info(
            "[decode:%s] epoch %d CER=%.4f WER=%.4f (n=%d)",
            self.tag,
            epoch + 1,
            metrics["cer"],
            metrics["wer"],
            metrics["n"],
        )
