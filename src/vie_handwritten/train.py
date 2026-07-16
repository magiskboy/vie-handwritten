"""Training: 2 phases on HWDB_line.

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

from tensorflow import keras

from vie_handwritten.charset import Charset
from vie_handwritten.dataset import (
    build_dataset,
    ensure_manifests,
    load_manifest,
)
from vie_handwritten.evaluate import evaluate_split
from vie_handwritten.model import CTCTrainer, build_crnn, load_crnn_weights, set_backbone_trainable
from vie_handwritten.utils import configure_runtime, ensure_dir, load_config, project_root, save_config, set_seed

logger = logging.getLogger(__name__)


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


def _make_optimizer(name: str, learning_rate: float):
    if name.lower() == "sgd":
        return keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    return keras.optimizers.Adam(learning_rate=learning_rate)


def _subsample(records: list[dict], n: int | None, seed: int) -> list[dict]:
    if n is None or len(records) <= n:
        return records
    return random.Random(seed).sample(records, n)


def _charset_path(config: dict[str, Any]) -> Path:
    p = Path(config["data"]["charset_path"])
    return p if p.is_absolute() else project_root() / p


def train(
    config_path: str | Path,
    *,
    resume_from: str | Path | None = None,
    rebuild_data: bool = False,
) -> CTCTrainer:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logger.info("Runtime: %s", configure_runtime())

    config = load_config(config_path)
    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)

    charset = Charset.from_file(_charset_path(config))
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
    trainer = CTCTrainer(crnn, blank_index=blank_index, name="ctc_trainer")

    best_path = ckpt_root / "best.weights.h5"
    save_best = SaveBest(crnn, best_path)
    optimizer_name = str(config["train"].get("optimizer", "adam"))

    phases = [
        ("phase1", config["train"]["phase1"], False),  # freeze backbone
        ("phase2", config["train"]["phase2"], True),  # train everything
    ]
    for name, pcfg, backbone_trainable in phases:
        logger.info("################ %s (backbone_trainable=%s) ################", name, backbone_trainable)
        set_backbone_trainable(crnn, backbone_trainable)
        trainer.compile(optimizer=_make_optimizer(optimizer_name, float(pcfg["learning_rate"])))

        records = _subsample(train_records, pcfg.get("max_train_samples"), seed)
        logger.info("[%s] training on %d samples", name, len(records))
        train_ds = build_dataset(records, charset=charset, config=config, training=True)
        callbacks = [
            save_best,
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
            keras.callbacks.TensorBoard(log_dir=str(project_root() / log_root / name)),
        ]
        trainer.fit(
            train_ds,
            validation_data=val_ds,
            epochs=int(pcfg["epochs"]),
            callbacks=callbacks,
            verbose=1,
        )

    if best_path.is_file():
        load_crnn_weights(crnn, best_path)
        logger.info("Restored best weights (val_loss=%.4f)", save_best.best)

    test_metrics = evaluate_split(crnn, load_manifest(manifests["test"]), charset, config)
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
