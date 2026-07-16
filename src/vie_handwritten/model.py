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


def _basic_block(
    x,
    filters: int,
    *,
    strides: tuple[int, int] = (1, 1),
    name: str,
):
    shortcut = x
    y = layers.Conv2D(
        filters,
        3,
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"{name}_conv1",
    )(x)
    y = layers.BatchNormalization(name=f"{name}_bn1")(y)
    y = layers.ReLU(name=f"{name}_relu1")(y)
    y = layers.Conv2D(
        filters,
        3,
        strides=1,
        padding="same",
        use_bias=False,
        name=f"{name}_conv2",
    )(y)
    y = layers.BatchNormalization(name=f"{name}_bn2")(y)

    if strides != (1, 1) or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters,
            1,
            strides=strides,
            padding="same",
            use_bias=False,
            name=f"{name}_proj_conv",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_proj_bn")(shortcut)

    y = layers.Add(name=f"{name}_add")([shortcut, y])
    y = layers.ReLU(name=f"{name}_out")(y)
    return y


def build_cnn_backbone(config: dict[str, Any], input_tensor=None):
    """Build ResNet-18 feature extractor with HTR-friendly strides.

    Standard ResNet-18 spatial downsampling is reduced on width for CTC:
      stem: (2, 2), maxpool (2, 2),
      layer2: (2, 2),
      layer3: (2, 1) if stride_policy=htr,
      layer4: (1, 1) if stride_policy=htr.
    Overall width factor ≈ 1/8.
    """
    model_cfg = config.get("model", config)
    stride_policy = model_cfg.get("stride_policy", "htr")
    if stride_policy == "htr":
        s3, s4 = (2, 1), (1, 1)
    else:
        s3, s4 = (2, 2), (2, 2)

    if input_tensor is None:
        inputs = keras.Input(shape=(None, None, 3), name="image")
        x = inputs
    else:
        inputs = None
        x = input_tensor

    # Stem
    x = layers.Conv2D(
        64, 7, strides=(2, 2), padding="same", use_bias=False, name="stem_conv"
    )(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.MaxPooling2D(pool_size=3, strides=(2, 2), padding="same", name="stem_pool")(
        x
    )

    # layer1: 2× BasicBlock, 64, stride 1
    x = _basic_block(x, 64, strides=(1, 1), name="layer1_block1")
    x = _basic_block(x, 64, strides=(1, 1), name="layer1_block2")

    # layer2: 2× BasicBlock, 128, stride 2
    x = _basic_block(x, 128, strides=(2, 2), name="layer2_block1")
    x = _basic_block(x, 128, strides=(1, 1), name="layer2_block2")

    # layer3: 2× BasicBlock, 256
    x = _basic_block(x, 256, strides=s3, name="layer3_block1")
    x = _basic_block(x, 256, strides=(1, 1), name="layer3_block2")

    # layer4: 2× BasicBlock, 512
    x = _basic_block(x, 512, strides=s4, name="layer4_block1")
    x = _basic_block(x, 512, strides=(1, 1), name="layer4_block2")

    if inputs is not None:
        return keras.Model(inputs, x, name="resnet18_backbone")
    return x


def _load_imagenet_weights(backbone: keras.Model) -> None:
    """Transfer matching weights from keras_hub ResNet-18 ImageNet preset."""
    try:
        import keras_hub
    except ImportError as exc:
        raise ImportError(
            "keras-hub is required for ImageNet pretrained ResNet-18"
        ) from exc

    logger.info("Loading keras_hub ResNetBackbone preset resnet_18_imagenet …")
    pretrained = keras_hub.models.ResNetBackbone.from_preset("resnet_18_imagenet")
    src_weights = list(pretrained.weights)
    dst_weights = list(backbone.weights)
    if len(src_weights) != len(dst_weights):
        logger.warning(
            "Weight count mismatch: pretrained=%d ours=%d",
            len(src_weights),
            len(dst_weights),
        )

    transferred = 0
    for sw, dw in zip(src_weights, dst_weights):
        if tuple(sw.shape) != tuple(dw.shape):
            logger.debug(
                "Skip mismatched shapes: %s %s vs %s %s",
                sw.name,
                tuple(sw.shape),
                dw.name,
                tuple(dw.shape),
            )
            continue
        dw.assign(sw)
        transferred += 1
    logger.info(
        "Transferred %d / %d backbone weights from ImageNet",
        transferred,
        len(dst_weights),
    )


class MapToSequence(layers.Layer):
    """Collapse height with max-pool → ``(B, W, C)`` sequence."""

    def call(self, feature_map):
        return tf.reduce_max(feature_map, axis=1)


def map_to_sequence(feature_map):
    """Convert CNN feature map ``(B, H, W, C)`` → sequence ``(B, T, D)``."""
    return tf.reduce_max(feature_map, axis=1)


def build_bilstm_head(config: dict[str, Any], num_classes: int, features):
    """Stack Bidirectional LSTM layers + Linear projection to ``num_classes``."""
    model_cfg = config.get("model", config)
    units = int(model_cfg.get("bilstm_units", 256))
    n_layers = int(model_cfg.get("bilstm_layers", 2))
    dropout = float(model_cfg.get("dropout", 0.2))
    x = features
    for i in range(n_layers):
        x = layers.Bidirectional(
            layers.LSTM(units, return_sequences=True, dropout=0.0),
            name=f"bilstm_{i + 1}",
        )(x)
        if dropout > 0:
            x = layers.Dropout(dropout, name=f"bilstm_drop_{i + 1}")(x)
    logits = layers.Dense(num_classes, name="logits")(x)
    return logits


def build_crnn(config: dict[str, Any], num_classes: int) -> keras.Model:
    """Assemble full CRNN and return a Keras ``Model`` outputting CTC logits.

    Output shape: ``(B, T, num_classes)`` where ``num_classes`` includes blank.
    """
    model_cfg = config.get("model", {})
    inputs = keras.Input(shape=(None, None, 3), name="image")
    features = build_cnn_backbone(config, input_tensor=inputs)
    backbone_model = keras.Model(inputs, features, name="resnet18_backbone")

    pretrained = model_cfg.get("pretrained", "imagenet")
    if pretrained == "imagenet":
        try:
            _load_imagenet_weights(backbone_model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load ImageNet weights: %s", exc)

    seq = MapToSequence(name="map_to_sequence")(features)
    logits = build_bilstm_head(config, num_classes, seq)
    model = keras.Model(inputs, logits, name="crnn_resnet18")
    model.backbone = backbone_model  # type: ignore[attr-defined]
    return model


STAGE_PREFIXES = {
    "stem": ("stem_",),
    "layer1": ("layer1_",),
    "layer2": ("layer2_",),
    "layer3": ("layer3_",),
    "layer4": ("layer4_",),
}


def set_backbone_trainable(model: keras.Model, phase_cfg: dict[str, Any], model_cfg: dict[str, Any]) -> None:
    """Configure which backbone stages are trainable for a training phase."""
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        # Fallback: toggle layers by name on full model
        targets = model
    else:
        targets = backbone

    freeze_all = bool(phase_cfg.get("freeze_backbone", False))
    if freeze_all:
        for layer in targets.layers:
            layer.trainable = False
        logger.info("Backbone fully frozen (phase %s)", phase_cfg.get("name"))
        return

    # Unfreeze from a stage onward; keep earlier freeze_stages frozen
    unfreeze_from = phase_cfg.get("unfreeze_from", "layer3")
    freeze_stages = list(model_cfg.get("freeze_stages", ["stem", "layer1", "layer2"]))
    stage_order = ["stem", "layer1", "layer2", "layer3", "layer4"]
    if unfreeze_from not in stage_order:
        raise ValueError(f"Unknown unfreeze_from={unfreeze_from}")
    unfreeze_idx = stage_order.index(unfreeze_from)

    for layer in targets.layers:
        layer.trainable = True
    for stage in stage_order[:unfreeze_idx]:
        prefixes = STAGE_PREFIXES[stage]
        for layer in targets.layers:
            if any(layer.name.startswith(p) for p in prefixes):
                layer.trainable = False
    # Also honor freeze_stages that extend beyond unfreeze_from
    for stage in freeze_stages:
        if stage in stage_order and stage_order.index(stage) >= unfreeze_idx:
            prefixes = STAGE_PREFIXES[stage]
            for layer in targets.layers:
                if any(layer.name.startswith(p) for p in prefixes):
                    layer.trainable = False
    logger.info(
        "Backbone trainable from %s onward (frozen: %s)",
        unfreeze_from,
        stage_order[:unfreeze_idx],
    )


class CTCModel(keras.Model):
    """Wrap a logits model with CTC train/test steps."""

    def __init__(self, crnn: keras.Model, blank_index: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.crnn = crnn
        self.blank_index = blank_index
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def call(self, inputs, training=False):
        if isinstance(inputs, dict):
            images = inputs["image"]
        else:
            images = inputs
        return self.crnn(images, training=training)

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        x, y = data
        images = x["image"]
        label_length = x["label_length"]
        input_length = x["input_length"]
        labels = y

        with tf.GradientTape() as tape:
            logits = self.crnn(images, training=True)
            time_steps = tf.shape(logits)[1]
            input_length = tf.ones_like(input_length) * tf.cast(
                time_steps, input_length.dtype
            )
            # Metal: CTC dense path forced to CPU inside mean_ctc_loss
            loss = mean_ctc_loss(
                labels,
                logits,
                blank_index=self.blank_index,
                label_length=label_length,
                logit_length=input_length,
            )

        grads = tape.gradient(loss, self.crnn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.crnn.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        images = x["image"]
        label_length = x["label_length"]
        input_length = x["input_length"]
        labels = y
        logits = self.crnn(images, training=False)
        time_steps = tf.shape(logits)[1]
        input_length = tf.ones_like(input_length) * tf.cast(time_steps, input_length.dtype)
        loss = mean_ctc_loss(
            labels,
            logits,
            blank_index=self.blank_index,
            label_length=label_length,
            logit_length=input_length,
        )
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def load_crnn_weights(model: keras.Model, checkpoint: str | Path) -> None:
    """Load CRNN weights from ``*.weights.h5`` or a weights path."""
    from pathlib import Path

    ckpt = Path(checkpoint)
    candidates = [ckpt, ckpt.with_suffix(".weights.h5")]
    if ckpt.suffix == ".keras":
        candidates.insert(0, ckpt.with_name(ckpt.stem + ".weights.h5"))
    for path in candidates:
        if path.is_file() and (
            str(path).endswith(".weights.h5") or path.suffix == ".h5"
        ):
            model.load_weights(str(path))
            return
    model.load_weights(str(ckpt))
