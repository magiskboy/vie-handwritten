"""Training: CTC loss, the ``OCRTrainer`` harness, debug metrics, and the 2-phase loop.

  Phase 1 — freeze the CNN backbone, train the BiLSTM head on a small subset.
  Phase 2 — unfreeze everything, train the whole network on the full dataset.

Both phases share one ``best.weights.h5`` tracking the lowest val_loss.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from vie_handwritten.charset import Charset
from vie_handwritten.dataset import build_dataset, ensure_manifests, load_manifest, resolve_image_path
from vie_handwritten.eval import evaluate_split
from vie_handwritten.model import (
    OCRModel,
    build_crnn,
    load_crnn_weights,
    set_backbone_trainable,
)
from vie_handwritten.postprocess import CTCDecoder
from vie_handwritten.preprocess import load_image, preprocess
from vie_handwritten.utils import (
    charset_path,
    configure_runtime,
    ensure_dir,
    load_config,
    project_root,
    save_config,
    set_seed,
)

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- #
# CTC loss
# --------------------------------------------------------------------------- #
def ctc_loss(
    y_true,
    y_pred,
    *,
    blank_index: int = 0,
    label_length=None,
    logit_length=None,
):
    """Compute mean CTC loss between label sequences and model logits."""
    y_true = tf.cast(y_true, tf.int32)
    if label_length is None:
        label_length = tf.reduce_sum(tf.ones_like(y_true), axis=1)
    if logit_length is None:
        logit_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])

    per_example = tf.nn.ctc_loss(
        labels=y_true,
        logits=y_pred,
        label_length=tf.cast(label_length, tf.int32),
        logit_length=tf.cast(logit_length, tf.int32),
        logits_time_major=False,
        blank_index=blank_index,
    )
    # zero_infinity: when a sample's time steps can't fit its label (T < label_length,
    # or repeats needing extra blanks) CTC returns +inf; zero it so one bad sample in a
    # batch can't blow up the mean loss / gradients (mirrors torch ``zero_infinity=True``).
    per_example = tf.where(tf.math.is_finite(per_example), per_example, tf.zeros_like(per_example))
    return tf.reduce_mean(per_example)


# --------------------------------------------------------------------------- #
# Training harness
# --------------------------------------------------------------------------- #
class OCRTrainer(keras.Model):
    """Training harness: wraps a CRNN (logits) model with a CTC loss train/test step.

    Not a distinct architecture — it delegates ``call`` to the wrapped ``crnn`` and
    only adds the CTC objective so the CRNN can be trained with ``fit``.
    """

    def __init__(self, crnn: keras.Model, blank_index: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.crnn = crnn
        self.blank_index = blank_index
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def call(self, inputs, training=False):
        images = inputs["image"] if isinstance(inputs, dict) else inputs
        return self.crnn(images, training=training)

    @property
    def metrics(self):
        return [self.loss_tracker]

    def _ctc_loss(self, x, labels, training):
        logits = self.crnn(x["image"], training=training)
        # ``input_length`` (from the dataset, = real width // WIDTH_DOWNSAMPLE) is the
        # single source of truth for valid time steps, so padded columns are excluded.
        # Clamp to the actual T in case "same"-padding rounding makes T slightly smaller.
        time_steps = tf.cast(tf.shape(logits)[1], x["input_length"].dtype)
        logit_length = tf.minimum(x["input_length"], time_steps)
        # Samples with too few time steps (T < label_length) are dropped in the data
        # pipeline; ``ctc_loss`` additionally zeroes any non-finite per-example loss so a
        # rare bad sample can never crash or poison a batch (cf. torch zero_infinity).
        return ctc_loss(
            labels,
            logits,
            blank_index=self.blank_index,
            label_length=x["label_length"],
            logit_length=logit_length,
        )

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            loss = self._ctc_loss(x, y, training=True)
        grads = tape.gradient(loss, self.crnn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.crnn.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        loss = self._ctc_loss(x, y, training=False)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
class SaveBest(keras.callbacks.Callback):
    """Save CRNN weights whenever val_loss improves (shared across phases)."""

    def __init__(self, crnn: keras.Model, path: Path):
        super().__init__()
        self.crnn = crnn
        self.path = path
        self.best = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        val_loss = (logs or {}).get("val_loss")
        if val_loss is not None and float(val_loss) < self.best:
            self.best = float(val_loss)
            self.crnn.save_weights(str(self.path))


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
    """Decode a fixed sample each epoch -> log CER/WER + previews to TensorBoard.

    Uses greedy decoding (no LM) for a fast in-loop sanity signal that both loss
    and error fall toward 0 on a tiny slice of the real train distribution.
    """

    def __init__(
        self,
        crnn: keras.Model,
        records: list[dict[str, str]],
        charset: Charset,
        config: dict[str, Any],
        *,
        tag: str,
        preview_dir: str | Path,
        every: int = 1,
        num_previews: int = 6,
    ):
        super().__init__()
        pp_cfg = config.get("postprocess", {})
        decoder = CTCDecoder(
            charset,
            method="greedy",
            blank_index=int(config.get("ctc", {}).get("blank_index", 0)),
            nfc=bool(pp_cfg.get("nfc", True)),
            tone_marks=bool(pp_cfg.get("normalize_tone_marks", True)),
            underthesea=bool(pp_cfg.get("underthesea", True)),
            underthesea_tokenizer=str(pp_cfg.get("underthesea_tokenizer", "regex")),
        )
        self.ocr = OCRModel(crnn, charset, decoder)
        self.records = records
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

        metrics = evaluate_split(self.ocr, self.records, self.config)
        logs[f"{self.tag}_cer"] = metrics["cer"]
        logs[f"{self.tag}_wer"] = metrics["wer"]

        rows, previews = [], []
        for rec in self.records[: self.num_previews]:
            image = load_image(str(resolve_image_path(self.config, rec)))
            arr = preprocess(image, self.config["preprocess"])
            previews.append(arr)
            pred = self.ocr.recognize(arr)
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


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def _make_optimizer(name: str, learning_rate: float, clipnorm: float | None = None):
    # Gradient clipping tames the occasional CTC/RNN loss spikes from hard (long) lines.
    kwargs: dict[str, Any] = {"learning_rate": learning_rate}
    if clipnorm is not None:
        kwargs["clipnorm"] = float(clipnorm)
    if name.lower() == "sgd":
        return keras.optimizers.SGD(momentum=0.9, **kwargs)
    return keras.optimizers.Adam(**kwargs)


def _subsample(records: list[dict], n: int | None, seed: int) -> list[dict]:
    if n is None or len(records) <= n:
        return records
    return random.Random(seed).sample(records, n)


def train(
    config_path: str | Path,
    *,
    resume_from: str | Path | None = None,
    rebuild_data: bool = False,
) -> OCRTrainer:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger.info("Runtime: %s", configure_runtime())

    config = load_config(config_path)
    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)

    charset = Charset.from_file(charset_path(config))
    logger.info("Charset classes: %d", charset.num_classes)

    ckpt_root = ensure_dir(project_root() / config["train"].get("checkpoint_dir", "checkpoints"))
    report_dir = ensure_dir(project_root() / config["train"].get("report_dir", "reports"))
    log_root = config["train"].get("log_dir", "runs")
    save_config(config, ckpt_root / "config_used.yaml")

    manifests = ensure_manifests(config, rebuild=rebuild_data)
    train_records = load_manifest(manifests["train"])
    val_records = load_manifest(manifests["val"])
    max_val = config["data"].get("max_val_samples")
    val_ds = build_dataset(
        _subsample(val_records, max_val, seed), charset=charset, config=config, training=False
    )
    logger.info("train=%d val=%d", len(train_records), len(val_records))

    crnn = build_crnn(config, num_classes=charset.num_classes)
    if resume_from is not None:
        logger.info("Loading weights from %s", resume_from)
        load_crnn_weights(crnn, resume_from)
    blank_index = int(config.get("ctc", {}).get("blank_index", charset.blank_index))
    trainer = OCRTrainer(crnn, blank_index=blank_index, name="ocr_trainer")

    best_path = ckpt_root / "best.weights.h5"
    save_best = SaveBest(crnn, best_path)
    optimizer_name = str(config["train"].get("optimizer", "adam"))
    clipnorm = config["train"].get("grad_clipnorm")

    phases = [
        ("phase1", config["train"]["phase1"], False),  # freeze backbone
        ("phase2", config["train"]["phase2"], True),  # train everything
    ]
    for name, pcfg, backbone_trainable in phases:
        logger.info("################ %s (backbone_trainable=%s) ################", name, backbone_trainable)
        set_backbone_trainable(crnn, backbone_trainable)
        trainer.compile(optimizer=_make_optimizer(optimizer_name, float(pcfg["learning_rate"]), clipnorm))

        records = _subsample(train_records, pcfg.get("max_train_samples"), seed)
        logger.info("[%s] training on %d samples", name, len(records))
        train_ds = build_dataset(records, charset=charset, config=config, training=True)
        phase_log_dir = project_root() / log_root / name
        callbacks = [save_best]
        # Decode-based debug metrics (CER/WER + previews) on a fixed train sample.
        n_decode = int(config["train"].get("decode_eval_samples", 0) or 0)
        if n_decode:
            callbacks.append(
                DecodeMetrics(
                    crnn,
                    _subsample(records, n_decode, seed),
                    charset,
                    config,
                    tag="train",
                    preview_dir=phase_log_dir / "decode",
                    every=int(config["train"].get("decode_eval_every", 1)),
                )
            )
        callbacks += [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(config["train"].get("early_stopping_patience", 10)),
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=float(config["train"].get("reduce_lr_factor", 0.5)),
                patience=int(config["train"].get("reduce_lr_patience", 5)),
                min_lr=1e-6,
                verbose=1,
            ),
            # Kept last so it captures scalars injected by DecodeMetrics (cer/wer/lr).
            keras.callbacks.TensorBoard(log_dir=str(phase_log_dir)),
        ]
        trainer.fit(
            train_ds,
            validation_data=val_ds,
            epochs=int(pcfg["epochs"]),
            callbacks=callbacks,
            shuffle=False,  # tf.data already shuffles
            verbose=1,
        )

    if best_path.is_file():
        load_crnn_weights(crnn, best_path)
        logger.info("Restored best weights (val_loss=%.4f)", save_best.best)

    ocr = OCRModel(crnn, charset, CTCDecoder.from_config(charset, config))
    test_metrics = evaluate_split(ocr, load_manifest(manifests["test"]), config)
    report = {"best_val_loss": save_best.best, "test": test_metrics, "weights": str(best_path)}
    (report_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "REPORT test CER=%.4f WER=%.4f (n=%d) → %s",
        test_metrics["cer"],
        test_metrics["wer"],
        test_metrics["n"],
        best_path,
    )
    return trainer
