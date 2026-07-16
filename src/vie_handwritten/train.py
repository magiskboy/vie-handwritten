"""Training loop: word→line curriculum, each phase with freeze→unfreeze stages."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import tensorflow as tf
from tensorflow import keras

from vie_handwritten.charset import Charset
from vie_handwritten.config import load_config, save_config
from vie_handwritten.dataset import (
    build_eval_dataset,
    build_training_dataset,
    ensure_source_manifests,
    group_by_source,
    load_manifest,
    resolve_image_path,
)
from vie_handwritten.model import CTCModel, build_crnn, load_crnn_weights, set_backbone_trainable
from vie_handwritten.utils import configure_runtime, ensure_dir, project_root, set_seed

logger = logging.getLogger(__name__)


def build_callbacks(
    config: dict[str, Any],
    *,
    checkpoint_path: Path,
    phase_name: str,
    crnn: keras.Model,
    best_tracker: dict[str, float],
    best_path: Path,
) -> list:
    """EarlyStopping, ReduceLROnPlateau, TensorBoard, and CRNN weight checkpoint.

    ``best_path`` receives the lowest-``val_loss`` weights across the whole phase
    (shared via ``best_tracker`` between the phase's stages).
    """
    train_cfg = config["train"]
    log_dir = ensure_dir(project_root() / train_cfg.get("log_dir", "runs") / phase_name)
    weights_path = checkpoint_path.with_suffix(".weights.h5")

    class SaveCRNNWeights(keras.callbacks.Callback):
        def __init__(
            self,
            model_to_save: keras.Model,
            path: Path,
            best_path: Path,
            best_tracker: dict[str, float],
        ):
            super().__init__()
            self.model_to_save = model_to_save
            self.path = path
            self.best_path = best_path
            # Shared across phases so best.weights.h5 tracks the global best.
            self.best_tracker = best_tracker
            # Per-phase best for the phase-specific checkpoint file.
            self.phase_best = float("inf")

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            val_loss = logs.get("val_loss")
            if val_loss is None:
                return
            val_loss = float(val_loss)
            if val_loss < self.phase_best:
                self.phase_best = val_loss
                self.model_to_save.save_weights(str(self.path))
            if val_loss < self.best_tracker["val_loss"]:
                self.best_tracker["val_loss"] = val_loss
                self.model_to_save.save_weights(str(self.best_path))

    callbacks = [
        SaveCRNNWeights(crnn, weights_path, best_path, best_tracker),
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


def _resolve_charset_path(config: dict[str, Any]) -> Path:
    charset_path = Path(config["data"]["charset_path"])
    if not charset_path.is_absolute():
        charset_path = project_root() / charset_path
    return charset_path


def _report_split(
    crnn: keras.Model,
    config: dict[str, Any],
    records: list[dict[str, str]],
    charset: Charset,
) -> dict[str, float]:
    """Greedy/beam decode ``records`` with the in-memory CRNN → CER/WER metrics."""
    from vie_handwritten.ctc import decode_predictions
    from vie_handwritten.metrics import evaluate_corpus
    from vie_handwritten.postprocess import postprocess
    from vie_handwritten.preprocess import load_image, preprocess

    ctc_cfg = config.get("ctc", {})
    refs: list[str] = []
    hyps: list[str] = []
    for rec in records:
        arr = preprocess(load_image(str(resolve_image_path(config, rec))), config["preprocess"])
        logits = crnn.predict(arr[None, ...], verbose=0)
        pred = decode_predictions(
            logits,
            charset,
            method=ctc_cfg.get("decode", "greedy"),
            blank_index=int(ctc_cfg.get("blank_index", 0)),
            beam_width=int(ctc_cfg.get("beam_width", 10)),
        )[0]
        refs.append(rec["text"])
        hyps.append(postprocess(pred))
    return evaluate_corpus(refs, hyps)


def _run_phase(
    phase: dict[str, Any],
    *,
    config: dict[str, Any],
    crnn: keras.Model,
    ctc_model: CTCModel,
    charset: Charset,
    ckpt_root: Path,
    report_dir: Path,
    seed: int,
    max_samples: int | None,
    rebuild_data: bool,
) -> dict[str, Any]:
    """Train one dataset phase (all its freeze→unfreeze stages) + write a report."""
    phase_name = phase["name"]
    source = phase["source"]
    steps_per_epoch = int(config["train"].get("steps_per_epoch", 500))
    logger.info("################ PHASE %s (source=%s) ################", phase_name, source)

    # --- Transfer init: reload best weights of a previous phase if requested. ---
    init_from = phase.get("init_from")
    if init_from and init_from != "imagenet":
        src_weights = ckpt_root / f"{init_from}.best.weights.h5"
        if not src_weights.is_file():
            raise FileNotFoundError(
                f"Phase {phase_name!r} needs weights from phase {init_from!r} at "
                f"{src_weights} — run that phase first."
            )
        logger.info("Transfer init: loading %s", src_weights)
        load_crnn_weights(crnn, src_weights)

    # --- Data (this phase pins to one source's frozen manifests). ---
    manifests = ensure_source_manifests(config, source, rebuild=rebuild_data)
    train_records = load_manifest(manifests["train"])
    val_records = load_manifest(manifests["val"])
    if max_samples is not None:
        rng = __import__("random").Random(seed)
        rng.shuffle(train_records)
        train_records = train_records[:max_samples]
        logger.info("Using subset of %d train samples (--max-samples)", len(train_records))
    train_by_source = group_by_source(train_records)
    logger.info("[%s] train=%d val=%d", source, len(train_records), len(val_records))

    val_ds = build_eval_dataset(
        val_records,
        charset=charset,
        config=config,
        max_samples=config["data"].get("max_val_samples"),
        seed=seed,
    )

    phase_best_tracker = {"val_loss": float("inf")}
    phase_best_path = ckpt_root / f"{phase_name}.best.weights.h5"
    history_all: dict[str, Any] = {}

    for stage in phase["stages"]:
        stage_name = stage.get("name", "stage")
        logger.info("=== %s / stage %s ===", phase_name, stage_name)
        set_backbone_trainable(crnn, stage, config.get("model", {}))
        # BiLSTM + Dense head always trainable.
        for layer in crnn.layers:
            if layer.name.startswith("bilstm") or layer.name == "logits":
                layer.trainable = True
        # Recompile after any trainable change (required by Keras).
        compile_model(ctc_model, config, learning_rate=float(stage["learning_rate"]))
        logger.info(
            "Trainable parameters: %s",
            f"{sum(int(tf.size(w)) for w in crnn.trainable_weights):,}",
        )

        train_ds = build_training_dataset(
            train_by_source,
            weights={source: 1.0},
            charset=charset,
            config=config,
            seed=seed,
        )
        stage_ckpt = ckpt_root / f"{phase_name}_{stage_name}.keras"
        callbacks = build_callbacks(
            config,
            checkpoint_path=stage_ckpt,
            phase_name=f"{phase_name}/{stage_name}",
            crnn=crnn,
            best_tracker=phase_best_tracker,
            best_path=phase_best_path,
        )
        history = ctc_model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=int(stage["epochs"]),
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            shuffle=False,  # tf.data already shuffled
            verbose=1,
        )
        history_all[stage_name] = history.history

    # Restore this phase's best weights before reporting / handing to next phase.
    if phase_best_path.is_file():
        load_crnn_weights(crnn, phase_best_path)
        logger.info("Restored phase best (val_loss=%.4f)", phase_best_tracker["val_loss"])

    # --- Report on this source's val + test splits. ---
    # Val can be huge (word ~12k) → cap it with max_val_samples; test is full.
    logger.info("[%s] evaluating val/test for report …", phase_name)
    report_val_records = val_records
    max_val = config["data"].get("max_val_samples")
    if max_val is not None and len(report_val_records) > int(max_val):
        report_val_records = __import__("random").Random(seed).sample(
            report_val_records, int(max_val)
        )
    val_metrics = _report_split(crnn, config, report_val_records, charset)
    test_metrics = _report_split(crnn, config, load_manifest(manifests["test"]), charset)
    report = {
        "phase": phase_name,
        "source": source,
        "init_from": init_from,
        "best_val_loss": phase_best_tracker["val_loss"],
        "val": val_metrics,
        "test": test_metrics,
        "weights": str(phase_best_path),
    }
    report_path = report_dir / f"{phase_name}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "[%s] REPORT test CER=%.4f WER=%.4f (n=%d) → %s",
        phase_name,
        test_metrics["cer"],
        test_metrics["wer"],
        test_metrics["n"],
        report_path,
    )
    return report


def train(
    config_path: str | Path,
    *,
    resume_from: str | Path | None = None,
    max_samples: int | None = None,
    rebuild_data: bool = False,
    only_phase: str | None = None,
) -> Any:
    """Curriculum training: run each dataset phase (word→line) end-to-end.

    Each phase trains its freeze→unfreeze stages, restores its best weights, then
    reports CER/WER on that source's val + test splits. ``only_phase`` restricts
    the run to a single phase (its ``init_from`` weights must already exist).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    runtime = configure_runtime()
    logger.info("Runtime: %s", runtime)

    config = load_config(config_path)
    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)

    if config["train"].get("mixed_precision"):
        keras.mixed_precision.set_global_policy("mixed_float16")
        logger.info("Mixed precision enabled")

    charset = Charset.from_file(_resolve_charset_path(config))
    logger.info("Charset classes: %d", charset.num_classes)

    ckpt_root = ensure_dir(project_root() / config["train"].get("checkpoint_dir", "checkpoints"))
    report_dir = ensure_dir(project_root() / config["train"].get("report_dir", "reports"))
    save_config(config, ckpt_root / "config_used.yaml")

    crnn = build_crnn(config, num_classes=charset.num_classes)
    if resume_from is not None:
        logger.info("Loading weights from %s", resume_from)
        load_crnn_weights(crnn, resume_from)

    blank_index = int(config.get("ctc", {}).get("blank_index", charset.blank_index))
    ctc_model = CTCModel(crnn, blank_index=blank_index, name="ctc_crnn")

    phases = list(config["train"].get("phases") or [])
    if not phases:
        raise ValueError("config train.phases is empty")
    if only_phase is not None:
        phases = [p for p in phases if p.get("name") == only_phase]
        if not phases:
            raise ValueError(f"--phase {only_phase!r} not found in config train.phases")

    reports: list[dict[str, Any]] = []
    for phase in phases:
        reports.append(
            _run_phase(
                phase,
                config=config,
                crnn=crnn,
                ctc_model=ctc_model,
                charset=charset,
                ckpt_root=ckpt_root,
                report_dir=report_dir,
                seed=seed,
                max_samples=max_samples,
                rebuild_data=rebuild_data,
            )
        )

    # The last phase (line) is the deployment model → expose as best.weights.h5.
    final = reports[-1]
    final_best = ckpt_root / f"{final['phase']}.best.weights.h5"
    if final_best.is_file():
        import shutil

        shutil.copyfile(final_best, ckpt_root / "best.weights.h5")

    logger.info("==================== CURRICULUM SUMMARY ====================")
    for r in reports:
        logger.info(
            "phase=%-6s source=%-5s test CER=%.4f WER=%.4f (n=%d)",
            r["phase"],
            r["source"],
            r["test"]["cer"],
            r["test"]["wer"],
            r["test"]["n"],
        )
    logger.info("Deployment weights: %s", ckpt_root / "best.weights.h5")
    return ctc_model
