"""Training loop helpers (Keras fit with 2-phase freeze/unfreeze)."""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Any

import tensorflow as tf
from tensorflow import keras

from vie_handwritten.charset import Charset
from vie_handwritten.config import load_config, save_config
from vie_handwritten.dataset import build_tf_dataset, discover_samples, train_val_test_split
from vie_handwritten.model import (
    CTCModel,
    build_crnn,
    set_backbone_trainable,
    set_sequence_head_trainable,
)
from vie_handwritten.utils import configure_runtime, ensure_dir, project_root, set_seed

logger = logging.getLogger(__name__)


@keras.utils.register_keras_serializable(package="vie_handwritten")
class WarmUpCosine(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup for ``warmup_steps`` then cosine decay to ``min_lr``.

    Transformers trained from scratch are unstable under Adam without warmup;
    this schedule ramps the LR up gently before annealing it.
    """

    def __init__(
        self,
        peak_lr: float,
        total_steps: int,
        warmup_steps: int,
        min_lr: float = 0.0,
    ) -> None:
        super().__init__()
        self.peak_lr = float(peak_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr = float(min_lr)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.maximum(1.0, float(self.warmup_steps))
        total = tf.maximum(warmup + 1.0, float(self.total_steps))
        peak = self.peak_lr
        warmup_lr = peak * (step / warmup)
        progress = tf.clip_by_value((step - warmup) / (total - warmup), 0.0, 1.0)
        cosine_lr = self.min_lr + 0.5 * (peak - self.min_lr) * (
            1.0 + tf.cos(math.pi * progress)
        )
        return tf.where(step < warmup, warmup_lr, cosine_lr)

    def get_config(self) -> dict[str, Any]:
        return {
            "peak_lr": self.peak_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
        }


def make_learning_rate(config: dict[str, Any], *, peak_lr: float, total_steps: int):
    """Return a float LR or a ``WarmUpCosine`` schedule based on config."""
    train_cfg = config["train"]
    kind = str(train_cfg.get("lr_schedule", "constant")).lower()
    if kind in ("constant", "none", "") or total_steps <= 1:
        return peak_lr
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.1))
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    min_lr = peak_lr * float(train_cfg.get("min_lr_ratio", 0.01))
    return WarmUpCosine(
        peak_lr=peak_lr,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=min_lr,
    )


def build_optimizer(config: dict[str, Any], *, learning_rate):
    """Build the optimizer (adam / adamw / sgd) with optional gradient clipping."""
    train_cfg = config["train"]
    opt_name = str(train_cfg.get("optimizer", "adam")).lower()
    kwargs: dict[str, Any] = {}
    clip_norm = train_cfg.get("grad_clip_norm")
    if clip_norm:
        kwargs["global_clipnorm"] = float(clip_norm)

    if opt_name == "adamw":
        return keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
            **kwargs,
        )
    if opt_name == "sgd":
        return keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, **kwargs)
    return keras.optimizers.Adam(learning_rate=learning_rate, **kwargs)


def _uses_lr_schedule(config: dict[str, Any]) -> bool:
    kind = str(config["train"].get("lr_schedule", "constant")).lower()
    return kind not in ("constant", "none", "")


def build_callbacks(
    config: dict[str, Any],
    *,
    checkpoint_path: Path,
    phase_name: str,
    crnn: keras.Model,
    global_best: dict[str, float],
) -> list:
    """EarlyStopping, ReduceLROnPlateau, TensorBoard, and CRNN weight checkpoint."""
    train_cfg = config["train"]
    log_dir = ensure_dir(project_root() / train_cfg.get("log_dir", "runs") / phase_name)
    weights_path = checkpoint_path.with_suffix(".weights.h5")

    class SaveCRNNWeights(keras.callbacks.Callback):
        """Save per-phase best weights, and update the global ``best.weights.h5``
        only when ``val_loss`` improves across *all* phases."""

        def __init__(self, model_to_save: keras.Model, path: Path, global_best: dict[str, float]):
            super().__init__()
            self.model_to_save = model_to_save
            self.path = path
            self.best_path = path.parent / "best.weights.h5"
            self.phase_best = float("inf")
            self.global_best = global_best

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            val_loss = logs.get("val_loss")
            if val_loss is None:
                return
            val_loss = float(val_loss)
            if val_loss < self.phase_best:
                self.phase_best = val_loss
                self.model_to_save.save_weights(str(self.path))
            if val_loss < self.global_best["val_loss"]:
                self.global_best["val_loss"] = val_loss
                self.model_to_save.save_weights(str(self.best_path))

    callbacks = [
        SaveCRNNWeights(crnn, weights_path, global_best),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(train_cfg.get("early_stopping_patience", 10)),
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(log_dir=str(log_dir)),
    ]
    # ReduceLROnPlateau cannot coexist with a LearningRateSchedule (the schedule
    # already anneals the LR from the optimizer's step count). Only add it when
    # the LR is a plain constant.
    if not _uses_lr_schedule(config):
        callbacks.insert(
            2,
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=float(train_cfg.get("reduce_lr_factor", 0.5)),
                patience=int(train_cfg.get("reduce_lr_patience", 5)),
                min_lr=1e-6,
                verbose=1,
            ),
        )
    return callbacks


def compile_model(model: CTCModel, config: dict[str, Any], *, learning_rate) -> CTCModel:
    """Attach optimizer and compile the CTC wrapper model.

    ``learning_rate`` may be a float or a ``LearningRateSchedule``.
    """
    optimizer = build_optimizer(config, learning_rate=learning_rate)
    model.compile(optimizer=optimizer)
    return model


def _resolve_data_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    root = project_root()
    data_cfg = config["data"]
    dataset_dir = Path(data_cfg["dataset_dir"])
    if not dataset_dir.is_absolute():
        dataset_dir = root / dataset_dir
    charset_path = Path(data_cfg["charset_path"])
    if not charset_path.is_absolute():
        charset_path = root / charset_path
    return dataset_dir, charset_path


def train(
    config_path: str | Path,
    *,
    resume_from: str | Path | None = None,
    max_samples: int | None = None,
) -> Any:
    """Full training entry: load data → build CRNN → 2-phase fit → save checkpoint."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    runtime = configure_runtime()
    logger.info("Runtime: %s", runtime)

    config = load_config(config_path)
    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)

    if config["train"].get("mixed_precision"):
        keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Mixed precision enabled")

    dataset_dir, charset_path = _resolve_data_paths(config)
    charset = Charset.from_file(charset_path)
    logger.info("Charset classes: %d", charset.num_classes)

    samples = discover_samples(
        dataset_dir,
        images_subdir=config["data"].get("images_subdir", "data"),
        labels_file=config["data"].get("labels_file", "labels.json"),
    )
    if max_samples is not None:
        if max_samples < 3:
            raise ValueError("max_samples must be >= 3 to allow train/val/test split")
        rng = random.Random(seed)
        samples = list(samples)
        rng.shuffle(samples)
        samples = samples[:max_samples]
        logger.info("Using subset of %d samples (--max-samples)", len(samples))

    # Fail fast on out-of-vocabulary characters rather than crashing mid-epoch
    # inside the tf.data map when Charset.encode raises KeyError.
    unknown = charset.unknown_characters([text for _, text in samples])
    if unknown:
        raise ValueError(
            f"{len(unknown)} character(s) in labels are missing from the charset "
            f"{charset_path}: {unknown}. Add them to the charset file (one per line) "
            "and retry."
        )

    train_s, val_s, test_s = train_val_test_split(
        samples,
        train_ratio=float(config["data"]["train_split"]),
        val_ratio=float(config["data"]["val_split"]),
        test_ratio=float(config["data"]["test_split"]),
        seed=seed,
    )
    logger.info("Splits train/val/test: %d / %d / %d", len(train_s), len(val_s), len(test_s))

    train_ds = build_tf_dataset(train_s, charset=charset, config=config, training=True)
    val_ds = build_tf_dataset(val_s, charset=charset, config=config, training=False)

    ckpt_root = ensure_dir(project_root() / config["train"].get("checkpoint_dir", "checkpoints"))
    save_config(config, ckpt_root / "config_used.yaml")

    crnn = build_crnn(config, num_classes=charset.num_classes)
    if resume_from is not None:
        logger.info("Loading weights from %s", resume_from)
        from vie_handwritten.model import load_crnn_weights

        load_crnn_weights(crnn, resume_from)

    blank_index = int(config.get("ctc", {}).get("blank_index", charset.blank_index))
    ctc_model = CTCModel(crnn, blank_index=blank_index, name="ctc_crnn")

    phases = config["train"].get("phases") or [
        {"name": "a_head", "epochs": 40, "learning_rate": 1e-3, "freeze_backbone": True},
        {
            "name": "b_finetune",
            "epochs": 60,
            "learning_rate": 1e-4,
            "freeze_backbone": False,
            "unfreeze_from": "layer3",
        },
    ]

    history_all = {}
    # Shared across phases so best.weights.h5 tracks the globally best val_loss.
    global_best = {"val_loss": float("inf")}
    batch_size = int(config["train"]["batch_size"])
    steps_per_epoch = max(1, math.ceil(len(train_s) / batch_size))

    for i, phase in enumerate(phases):
        phase_name = phase.get("name", f"phase_{i}")
        logger.info("=== Phase %s ===", phase_name)
        set_backbone_trainable(crnn, phase, config.get("model", {}))
        # Transformer sequence head + logits always trainable
        set_sequence_head_trainable(crnn, trainable=True)

        peak_lr = float(phase["learning_rate"])
        total_steps = steps_per_epoch * int(phase["epochs"])
        # A fresh optimizer is created per phase, so its step count restarts at 0
        # and the per-phase schedule (warmup + cosine) lines up with the phase.
        lr = make_learning_rate(config, peak_lr=peak_lr, total_steps=total_steps)
        compile_model(ctc_model, config, learning_rate=lr)
        logger.info(
            "Phase %s LR: peak=%.2e schedule=%s (%d steps/epoch)",
            phase_name,
            peak_lr,
            config["train"].get("lr_schedule", "constant"),
            steps_per_epoch,
        )

        trainable = sum(int(tf.size(w)) for w in crnn.trainable_weights)
        logger.info("Trainable parameters: %s", f"{trainable:,}")

        phase_ckpt = ckpt_root / f"{phase_name}.keras"
        callbacks = build_callbacks(
            config,
            checkpoint_path=phase_ckpt,
            phase_name=phase_name,
            crnn=crnn,
            global_best=global_best,
        )

        history = ctc_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=int(phase["epochs"]),
            callbacks=callbacks,
            shuffle=False,  # tf.data already shuffled
            verbose=1,
        )
        history_all[phase_name] = history.history

        # Per-phase best weights are written by SaveCRNNWeights; EarlyStopping
        # restores the best in-memory weights so the next phase continues from
        # the best of this phase. best.weights.h5 already holds the global best.
        weights_path = phase_ckpt.with_suffix(".weights.h5")
        if not weights_path.is_file():
            crnn.save_weights(str(weights_path))
        logger.info(
            "Phase %s done. Phase weights: %s (global best val_loss=%.4f)",
            phase_name,
            weights_path,
            global_best["val_loss"],
        )

    # Save test split paths for evaluate
    test_list = ckpt_root / "test_samples.txt"
    with test_list.open("w", encoding="utf-8") as f:
        for path, text in test_s:
            f.write(f"{path}\t{text}\n")

    logger.info("Training complete. Best weights: %s", ckpt_root / "best.weights.h5")
    return ctc_model
