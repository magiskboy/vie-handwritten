"""CRNN model: ResNet-18 → BiLSTM → Linear (logits for CTC).

Architecture:
  Input (B, H, W, C)
    → ResNet-18 backbone (HTR strides, no classifier)
    → Map to sequence (B, T, D) along image width
    → Bidirectional LSTM × N
    → Dense(num_classes)  (logits; no softmax — CTC applies it)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from vie_handwritten.charset import Charset
from vie_handwritten.postprocess import CTCDecoder
from vie_handwritten.utils import (
    WEIGHTS_NAME,
    charset_path,
    load_checkpoint_config,
    project_root,
    resolve_checkpoint_dir,
    resolve_ctc_paths,
)

# Vendored ImageNet ResNet-18 weights (our HTR backbone layer names). See
# third_party/resnet_18_imagenet/SOURCE.md for upstream provenance.
IMAGENET_BACKBONE_WEIGHTS = Path("third_party/resnet_18_imagenet/backbone.weights.h5")

logger = logging.getLogger(__name__)

# Width downsampling of the CNN backbone = product of width strides:
#   stem_conv (2) · stem_pool (2) · layer2 width-stride (1) · layer3 (1) · layer4 (1) = 4.
# Kept at 4 (not 8) so CTC gets ~2× the time steps: long Vietnamese lines need
# T ≳ 2·label_length or the model drops characters (measured T/L < 1.7 → deletions).
# This is the single source of truth for how image width maps to CTC time steps
# (T ≈ width / WIDTH_DOWNSAMPLE); the dataset uses it to derive ``input_length``.
WIDTH_DOWNSAMPLE = 4

# Match keras_hub ResNet BN so transferred ImageNet moving stats stay valid under
# our forward pass (Keras BatchNormalization defaults are momentum=0.99, epsilon=1e-3).
_BN_MOMENTUM = 0.9
_BN_EPSILON = 1e-5


def _bn(name: str) -> layers.BatchNormalization:
    return layers.BatchNormalization(momentum=_BN_MOMENTUM, epsilon=_BN_EPSILON, name=name)


def _basic_block(x, filters: int, *, strides: tuple[int, int] = (1, 1), name: str):
    """One ResNet basic block. ``name`` is ``layer{L}_block{B}`` (1-based).

    Weighted layers (must stay stable — vendored ImageNet ``.h5`` keys off these):
      {name}_conv1, {name}_bn1, {name}_conv2, {name}_bn2
      {name}_proj_conv, {name}_proj_bn   # only when stride≠(1,1) or channel change
    Non-weighted: {name}_relu1, {name}_add, {name}_out.
    """
    shortcut = x
    y = layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    y = _bn(f"{name}_bn1")(y)
    y = layers.ReLU(name=f"{name}_relu1")(y)
    y = layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False, name=f"{name}_conv2")(y)
    y = _bn(f"{name}_bn2")(y)

    if strides != (1, 1) or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False, name=f"{name}_proj_conv")(shortcut)
        shortcut = _bn(f"{name}_proj_bn")(shortcut)

    y = layers.Add(name=f"{name}_add")([shortcut, y])
    return layers.ReLU(name=f"{name}_out")(y)


def build_cnn_backbone(input_tensor):
    """ResNet-18 feature extractor with HTR strides (width downsample ≈ 1/4).

    Width is downsampled only by the stem (stem_conv ·2, stem_pool ·2); layer2/3/4
    keep width-stride 1 so CTC retains enough time steps. Height is still downsampled
    by layer2 (2,2) and layer3 (2,1). See ``WIDTH_DOWNSAMPLE``.

    Layer-name convention (this project; used by vendored ImageNet weights)::

        stem_conv, stem_bn, stem_relu, stem_pool
        layer{L}_block{B}_{conv1|bn1|relu1|conv2|bn2|proj_conv|proj_bn|add|out}

    where L ∈ {1..4}, B ∈ {1, 2} for ResNet-18. Indexing is **1-based**.

    keras_hub ``ResNetBackbone`` (basic block) uses 0-based stacks/blocks and
    different suffixes — map when dumping a new preset::

        stem_conv / stem_bn  ←  conv1_conv / conv1_bn
        layer{L}_block{B}_conv1     ←  stack{L-1}_block{B-1}_1_conv
        layer{L}_block{B}_bn1       ←  stack{L-1}_block{B-1}_1_bn
        layer{L}_block{B}_conv2     ←  stack{L-1}_block{B-1}_2_conv
        layer{L}_block{B}_bn2       ←  stack{L-1}_block{B-1}_2_bn
        layer{L}_block{B}_proj_conv ←  stack{L-1}_block{B-1}_0_conv
        layer{L}_block{B}_proj_bn   ←  stack{L-1}_block{B-1}_0_bn

    Match by *name*, not weight order: hub interleaves projection weights inside
    downsample blocks. Full provenance: ``third_party/resnet_18_imagenet/SOURCE.md``.
    """
    x = layers.Conv2D(64, 7, strides=(2, 2), padding="same", use_bias=False, name="stem_conv")(input_tensor)
    x = _bn("stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=(2, 2), padding="same", name="stem_pool")(x)

    x = _basic_block(x, 64, strides=(1, 1), name="layer1_block1")
    x = _basic_block(x, 64, strides=(1, 1), name="layer1_block2")

    x = _basic_block(x, 128, strides=(2, 1), name="layer2_block1")
    x = _basic_block(x, 128, strides=(1, 1), name="layer2_block2")

    x = _basic_block(x, 256, strides=(2, 1), name="layer3_block1")
    x = _basic_block(x, 256, strides=(1, 1), name="layer3_block2")

    x = _basic_block(x, 512, strides=(1, 1), name="layer4_block1")
    x = _basic_block(x, 512, strides=(1, 1), name="layer4_block2")
    return x


def _imagenet_backbone_weights_path() -> Path:
    """Absolute path to the vendored ImageNet backbone weights file."""
    return project_root() / IMAGENET_BACKBONE_WEIGHTS


def _load_imagenet_weights(backbone: keras.Model) -> None:
    """Load ImageNet ResNet-18 weights from the vendored on-disk checkpoint.

    Weights were transferred once from keras_hub ``resnet_18_imagenet`` onto this
    backbone's layer names and committed under ``third_party/resnet_18_imagenet/``.
    No network access is required.
    """
    path = _imagenet_backbone_weights_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"Vendored ImageNet weights not found: {path}. "
            "Expected third_party/resnet_18_imagenet/backbone.weights.h5 in the repo."
        )
    logger.info("Loading ImageNet ResNet-18 backbone weights from %s …", path)
    backbone.load_weights(str(path))


class MapToSequence(layers.Layer):
    """Collapse height with max-pool → ``(B, W, C)`` sequence."""

    def call(self, feature_map):
        return tf.reduce_max(feature_map, axis=1)


def build_bilstm_head(model_cfg: dict[str, Any], num_classes: int, features):
    """Stack Bidirectional LSTM layers + Linear projection to ``num_classes``."""
    units = int(model_cfg.get("bilstm_units", 256))
    n_layers = int(model_cfg.get("bilstm_layers", 2))
    dropout = float(model_cfg.get("dropout", 0.2))
    x = features
    for i in range(n_layers):
        x = layers.Bidirectional(
            layers.LSTM(units, return_sequences=True), name=f"bilstm_{i + 1}"
        )(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"bilstm_drop_{i + 1}")(x)
    return layers.Dense(num_classes, name="logits")(x)


def build_crnn(config: dict[str, Any], num_classes: int) -> keras.Model:
    """Assemble the full CRNN, returning a Keras ``Model`` outputting CTC logits.

    Output shape: ``(B, T, num_classes)`` where ``num_classes`` includes blank.
    The backbone is exposed as ``model.backbone`` for freeze/unfreeze.
    """
    model_cfg = config["model"]
    inputs = keras.Input(shape=(None, None, 3), name="image")
    features = build_cnn_backbone(inputs)
    backbone = keras.Model(inputs, features, name="resnet18_backbone")

    if model_cfg.get("pretrained", "imagenet") == "imagenet":
        try:
            _load_imagenet_weights(backbone)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load ImageNet weights: %s", exc)

    seq = MapToSequence(name="map_to_sequence")(features)
    logits = build_bilstm_head(model_cfg, num_classes, seq)
    model = keras.Model(inputs, logits, name="crnn_resnet18")
    model.backbone = backbone  # type: ignore[attr-defined]
    return model


def set_backbone_trainable(model: keras.Model, trainable: bool) -> None:
    """Freeze (``trainable=False``) or unfreeze the whole ResNet backbone."""
    model.backbone.trainable = trainable
    logger.info("Backbone %s", "trainable" if trainable else "frozen")


def load_crnn_weights(model: keras.Model, weights_path: str | Path) -> None:
    """Load CRNN weights from a ``*.weights.h5`` file."""
    model.load_weights(str(weights_path))


class OCRModel:
    """A usable OCR model: composition of the deep-learning net + postprocess.

    Wraps the CRNN (produces CTC logits) together with a :class:`CTCDecoder`
    (logits -> clean Vietnamese text), exposing a single ``recognize`` step so
    callers never have to wire the net, charset and decoder together by hand.
    Images passed in are expected to be already preprocessed (see ``preprocess``);
    the train-time preprocess settings live on ``self.config["preprocess"]``.
    """

    def __init__(
        self,
        net: keras.Model,
        charset: Charset,
        decoder: CTCDecoder,
        config: dict[str, Any] | None = None,
    ):
        self.net = net
        self.charset = charset
        self.decoder = decoder
        self.config = config or {}

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        config: dict[str, Any] | None = None,
        decode: str | None = None,
    ) -> "OCRModel":
        """Load from a self-contained checkpoint directory.

        Required: ``model.weights.h5``, ``config.yaml``, ``charset.txt``,
        ``build_info.yaml`` (and ``lm/`` when using ``beam_lm``).

        ``config`` is the already-loaded checkpoint config (optional); ``decode``
        overrides ``ctc.decode`` at runtime.
        """
        root = resolve_checkpoint_dir(checkpoint)
        cfg = dict(config) if config is not None else load_checkpoint_config(root)
        if decode is not None:
            cfg = {**cfg, "ctc": {**cfg.get("ctc", {}), "decode": decode}}
        cfg = resolve_ctc_paths(cfg, root)
        charset = Charset.from_file(charset_path(cfg, artifact_root=root))
        net = build_crnn(cfg, num_classes=charset.num_classes)
        load_crnn_weights(net, root / WEIGHTS_NAME)
        decoder = CTCDecoder.from_config(charset, cfg)
        return cls(net, charset, decoder, config=cfg)

    def predict_logits(self, images: np.ndarray) -> np.ndarray:
        """Forward pass: batched preprocessed images ``(B, H, W, C)`` -> logits."""
        return self.net.predict(images, verbose=0)

    def recognize(self, image_array: np.ndarray) -> str:
        """Recognize a single preprocessed image ``(H, W, C)`` -> text."""
        return self.decoder.decode(self.predict_logits(image_array[None, ...]))[0]

    def recognize_batch(self, image_arrays: np.ndarray) -> list[str]:
        """Recognize a batch of equal-shape preprocessed images -> list of texts."""
        return self.decoder.decode(self.predict_logits(image_arrays))
