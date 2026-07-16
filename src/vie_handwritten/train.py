"""Training loop helpers (Keras fit with 2-phase freeze/unfreeze)."""

from __future__ import annotations

import logging
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


def build_callbacks(
    config: dict[str, Any],
    *,
    checkpoint_path: Path,
    phase_name: str,
    crnn: keras.Model,
) -> list:
    """EarlyStopping, ReduceLROnPlateau, TensorBoard, and CRNN weight checkpoint."""
    train_cfg = config["train"]
    log_dir = ensure_dir(project_root() / train_cfg.get("log_dir", "runs") / phase_name)
    weights_path = checkpoint_path.with_suffix(".weights.h5")

    class SaveCRNNWeights(keras.callbacks.Callback):
        def __init__(self, model_to_save: keras.Model, path: Path):
            super().__init__()
            self.model_to_save = model_to_save
            self.path = path
            self.best = float("inf")

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            val_loss = logs.get("val_loss")
            if val_loss is None:
                return
            if val_loss < self.best:
                self.best = float(val_loss)
                self.model_to_save.save_weights(str(self.path))
                self.model_to_save.save_weights(
                    str(self.path.parent / "best.weights.h5")
                )

    callbacks = [
        SaveCRNNWeights(crnn, weights_path),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(train_cfg.get("early_stopping_patience", 10)),
            restore_best_weights=False,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=float(train_cfg.get("reduce_lr_factor", 0.5)),
            patience=int(train_cfg.get("reduce_lr_patience", 5)),
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.TensorBoard(log_dir=str(log_dir)),
    ]
    return callbacks


def compile_model(model: CTCModel, config: dict[str, Any], *, learning_rate: float) -> CTCModel:
    """Attach optimizer and compile the CTC wrapper model."""
    opt_name = str(config["train"].get("optimizer", "adam")).lower()
    if opt_name == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    elif opt_name == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
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
        rng = __import__("random").Random(seed)
        samples = list(samples)
        rng.shuffle(samples)
        samples = samples[:max_samples]
        logger.info("Using subset of %d samples (--max-samples)", len(samples))

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

    for i, phase in enumerate(phases):
        phase_name = phase.get("name", f"phase_{i}")
        logger.info("=== Phase %s ===", phase_name)
        set_backbone_trainable(crnn, phase, config.get("model", {}))
        # Transformer sequence head + logits always trainable
        set_sequence_head_trainable(crnn, trainable=True)

        lr = float(phase["learning_rate"])
        compile_model(ctc_model, config, learning_rate=lr)

        trainable = sum(int(tf.size(w)) for w in crnn.trainable_weights)
        logger.info("Trainable parameters: %s", f"{trainable:,}")

        phase_ckpt = ckpt_root / f"{phase_name}.keras"
        callbacks = build_callbacks(
            config, checkpoint_path=phase_ckpt, phase_name=phase_name, crnn=crnn
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

        weights_path = phase_ckpt.with_suffix(".weights.h5")
        crnn.save_weights(str(weights_path))
        crnn.save_weights(str(ckpt_root / "best.weights.h5"))
        logger.info("Saved phase weights: %s", weights_path)

    # Save test split paths for evaluate
    test_list = ckpt_root / "test_samples.txt"
    with test_list.open("w", encoding="utf-8") as f:
        for path, text in test_s:
            f.write(f"{path}\t{text}\n")

    logger.info("Training complete. Best weights: %s", ckpt_root / "best.weights.h5")
    return ctc_model
