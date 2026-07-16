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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from vie_handwritten.ctc import ctc_loss as mean_ctc_loss

logger = logging.getLogger(__name__)


def _basic_block(x, filters: int, *, strides: tuple[int, int] = (1, 1), name: str):
    shortcut = x
    y = layers.Conv2D(filters, 3, strides=strides, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.ReLU(name=f"{name}_relu1")(y)
    y = layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False, name=f"{name}_conv2")(y)
    y = layers.BatchNormalization(name=f"{name}_bn2")(y)

    if strides != (1, 1) or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding="same", use_bias=False, name=f"{name}_proj_conv")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

    y = layers.Add(name=f"{name}_add")([shortcut, y])
    return layers.ReLU(name=f"{name}_out")(y)


def build_cnn_backbone(input_tensor):
    """ResNet-18 feature extractor with HTR strides (width downsample ≈ 1/8).

    layer3 uses stride (2, 1) and layer4 stride (1, 1) to preserve width for CTC.
    """
    x = layers.Conv2D(64, 7, strides=(2, 2), padding="same", use_bias=False, name="stem_conv")(input_tensor)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=(2, 2), padding="same", name="stem_pool")(x)

    x = _basic_block(x, 64, strides=(1, 1), name="layer1_block1")
    x = _basic_block(x, 64, strides=(1, 1), name="layer1_block2")

    x = _basic_block(x, 128, strides=(2, 2), name="layer2_block1")
    x = _basic_block(x, 128, strides=(1, 1), name="layer2_block2")

    x = _basic_block(x, 256, strides=(2, 1), name="layer3_block1")
    x = _basic_block(x, 256, strides=(1, 1), name="layer3_block2")

    x = _basic_block(x, 512, strides=(1, 1), name="layer4_block1")
    x = _basic_block(x, 512, strides=(1, 1), name="layer4_block2")
    return x


def _load_imagenet_weights(backbone: keras.Model) -> None:
    """Transfer matching weights from keras_hub ResNet-18 ImageNet preset."""
    import keras_hub

    logger.info("Loading keras_hub ResNetBackbone preset resnet_18_imagenet …")
    pretrained = keras_hub.models.ResNetBackbone.from_preset("resnet_18_imagenet")
    transferred = 0
    for sw, dw in zip(pretrained.weights, backbone.weights):
        if tuple(sw.shape) == tuple(dw.shape):
            dw.assign(sw)
            transferred += 1
    logger.info("Transferred %d / %d backbone weights from ImageNet", transferred, len(backbone.weights))


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


class CTCTrainer(keras.Model):
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
        time_steps = tf.shape(logits)[1]
        logit_length = tf.ones_like(x["input_length"]) * tf.cast(time_steps, x["input_length"].dtype)
        return mean_ctc_loss(
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


def load_crnn_weights(model: keras.Model, checkpoint: str | Path) -> None:
    """Load CRNN weights from a ``*.weights.h5`` file."""
    ckpt = Path(checkpoint)
    for path in (ckpt, ckpt.with_suffix(".weights.h5")):
        if path.is_file():
            model.load_weights(str(path))
            return
    model.load_weights(str(ckpt))
