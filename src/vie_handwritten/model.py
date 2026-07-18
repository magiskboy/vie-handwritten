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
import re
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
    resolve_checkpoint_dir,
    resolve_ctc_paths,
)

logger = logging.getLogger(__name__)

# Width downsampling of the CNN backbone = product of width strides:
#   stem_conv (2) · stem_pool (2) · layer2 width-stride (1) · layer3 (1) · layer4 (1) = 4.
# Kept at 4 (not 8) so CTC gets ~2× the time steps: long Vietnamese lines need
# T ≳ 2·label_length or the model drops characters (measured T/L < 1.7 → deletions).
# This is the single source of truth for how image width maps to CTC time steps
# (T ≈ width / WIDTH_DOWNSAMPLE); the dataset uses it to derive ``input_length``.
WIDTH_DOWNSAMPLE = 4

# Match keras_hub ResNet so transferred ImageNet moving stats stay valid under our
# forward pass (Keras BatchNormalization defaults are momentum=0.99, epsilon=1e-3).
_BN_MOMENTUM = 0.9
_BN_EPSILON = 1e-5


def _bn(name: str) -> layers.BatchNormalization:
    return layers.BatchNormalization(momentum=_BN_MOMENTUM, epsilon=_BN_EPSILON, name=name)


def _basic_block(x, filters: int, *, strides: tuple[int, int] = (1, 1), name: str):
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


def _hub_layer_name(custom_name: str) -> str | None:
    """Map a layer name in our backbone → the matching keras_hub ResNet-18 layer.

    Mapping is derived from the official keras_hub ``ResNetBackbone`` source (basic
    block): the stem is ``conv1_{conv,bn}``; each residual block ``stack{s}_block{b}``
    names its main path ``_1_*`` / ``_2_*`` and its projection shortcut ``_0_*``.
    Our ``layer{L}`` (1-based) corresponds to hub ``stack{L-1}`` and ``block{B}``
    (1-based) to hub ``block{B-1}``.

    Matching by *name* (not by weight order) is required because keras_hub emits a
    downsample block's weights interleaved (``_1_conv, _1_bn, _0_conv, _2_conv, …``),
    so a positional ``zip`` silently misaligns tensors of equal shape.
    """
    if custom_name == "stem_conv":
        return "conv1_conv"
    if custom_name == "stem_bn":
        return "conv1_bn"
    m = re.match(r"layer(\d+)_block(\d+)_(conv1|bn1|conv2|bn2|proj_conv|proj_bn)$", custom_name)
    if not m:
        return None
    stack, block = int(m.group(1)) - 1, int(m.group(2)) - 1
    suffix = {
        "conv1": "1_conv",
        "bn1": "1_bn",
        "conv2": "2_conv",
        "bn2": "2_bn",
        "proj_conv": "0_conv",
        "proj_bn": "0_bn",
    }[m.group(3)]
    return f"stack{stack}_block{block}_{suffix}"


def _load_imagenet_weights(backbone: keras.Model) -> None:
    """Transfer ImageNet weights from keras_hub ResNet-18, matched by layer name.

    Every weighted layer must map and have identical shapes, otherwise we raise so a
    silent partial/misaligned transfer can never corrupt the backbone.
    """
    import keras_hub

    logger.info("Loading keras_hub ResNetBackbone preset resnet_18_imagenet …")
    pretrained = keras_hub.models.ResNetBackbone.from_preset("resnet_18_imagenet")
    hub_layers = {layer.name: layer for layer in pretrained.layers}

    transferred = 0
    weighted = [layer for layer in backbone.layers if layer.weights]
    unmatched: list[str] = []
    for layer in weighted:
        hub_name = _hub_layer_name(layer.name)
        src = hub_layers.get(hub_name) if hub_name else None
        if src is None:
            unmatched.append(layer.name)
            continue
        dst_w, src_w = layer.get_weights(), src.get_weights()
        if len(dst_w) != len(src_w) or any(d.shape != s.shape for d, s in zip(dst_w, src_w)):
            unmatched.append(f"{layer.name}→{hub_name}(shape)")
            continue
        layer.set_weights(src_w)
        transferred += 1

    logger.info("Transferred %d / %d weighted backbone layers from ImageNet", transferred, len(weighted))
    if unmatched:
        raise RuntimeError(f"ImageNet transfer incomplete; unmatched layers: {unmatched}")


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
