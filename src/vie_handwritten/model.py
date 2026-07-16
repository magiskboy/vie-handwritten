"""CRNN model: ResNet-18 → Transformer Encoder → Linear (logits for CTC).

Architecture:
  Input (B, H, W, C) + input_length
    → ResNet-18 backbone (HTR strides, no classifier)
    → Map to sequence (B, T, D) along image width
    → Dense → sinusoidal PE → Transformer Encoder × N
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

# ResNet HTR stride policy downsamples width by ~8.
WIDTH_STRIDE = 8


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


class SinusoidalPositionalEncoding(layers.Layer):
    """Add fixed sinusoidal positional encodings to a sequence ``(B, T, D)``."""

    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)

    def call(self, x):
        t = tf.shape(x)[1]
        depth = self.d_model // 2
        positions = tf.cast(tf.range(t), tf.float32)[:, tf.newaxis]
        depths = tf.cast(tf.range(depth), tf.float32)[tf.newaxis, :] / float(depth)
        angle_rates = 1.0 / tf.pow(10000.0, depths)
        angle_rads = positions * angle_rates
        pe = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)
        if self.d_model % 2 == 1:
            pe = tf.pad(pe, [[0, 0], [0, 1]])
        return x + pe[tf.newaxis, :, :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model})
        return cfg


def _transformer_encoder_block(
    x,
    attention_mask,
    *,
    d_model: int,
    num_heads: int,
    ffn_dim: int,
    dropout: float,
    name: str,
):
    """Pre-norm Transformer encoder block with padding mask."""
    key_dim = max(1, d_model // num_heads)
    y = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(x)
    y = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout,
        name=f"{name}_mha",
    )(y, y, attention_mask=attention_mask)
    y = layers.Dropout(dropout, name=f"{name}_drop1")(y)
    x = layers.Add(name=f"{name}_add1")([x, y])

    y = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(x)
    y = layers.Dense(ffn_dim, activation="gelu", name=f"{name}_ffn1")(y)
    y = layers.Dropout(dropout, name=f"{name}_drop2")(y)
    y = layers.Dense(d_model, name=f"{name}_ffn2")(y)
    y = layers.Dropout(dropout, name=f"{name}_drop3")(y)
    x = layers.Add(name=f"{name}_add2")([x, y])
    return x


def build_transformer_head(
    config: dict[str, Any],
    num_classes: int,
    features,
    input_length,
):
    """Project CNN sequence → Transformer encoder → logits for CTC."""
    model_cfg = config.get("model", config)
    d_model = int(model_cfg.get("d_model", 256))
    n_layers = int(model_cfg.get("transformer_layers", 4))
    num_heads = int(model_cfg.get("transformer_heads", 4))
    ffn_dim = int(model_cfg.get("transformer_ffn_dim", 4 * d_model))
    dropout = float(model_cfg.get("dropout", 0.1))

    if d_model % num_heads != 0:
        raise ValueError(f"d_model={d_model} must be divisible by heads={num_heads}")

    x = layers.Dense(d_model, name="seq_proj")(features)
    x = SinusoidalPositionalEncoding(d_model, name="pos_encoding")(x)

    # (B, 1, T): each query attends only to valid (non-pad) keys
    def _attention_mask(args):
        seq, lengths = args
        t = tf.shape(seq)[1]
        mask = tf.sequence_mask(lengths, maxlen=t)  # (B, T), True = valid
        return mask[:, tf.newaxis, :]

    attention_mask = layers.Lambda(_attention_mask, name="attention_mask")(
        [x, input_length]
    )

    for i in range(n_layers):
        x = _transformer_encoder_block(
            x,
            attention_mask,
            d_model=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            name=f"transformer_{i + 1}",
        )

    x = layers.LayerNormalization(epsilon=1e-6, name="transformer_out_ln")(x)
    logits = layers.Dense(num_classes, name="logits")(x)
    return logits


def estimate_input_length(image_width, *, dtype=tf.int32):
    """Approximate CTC/attention length from image width (HTR stride ≈ 8)."""
    return tf.maximum(tf.cast(image_width, dtype) // WIDTH_STRIDE, 1)


def pack_crnn_inputs(images, input_length=None):
    """Build the dict expected by ``build_crnn`` from a batch of images.

    ``images``: ``(B, H, W, C)`` numpy/tensor. If ``input_length`` is None,
    use ``W // 8`` for every sample (no width padding).
    """
    images = tf.convert_to_tensor(images)
    if input_length is None:
        w = tf.shape(images)[2]
        batch = tf.shape(images)[0]
        input_length = tf.fill([batch], estimate_input_length(w))
    else:
        input_length = tf.convert_to_tensor(input_length, dtype=tf.int32)
        if input_length.shape.rank == 0:
            input_length = tf.fill([tf.shape(images)[0]], input_length)
    return {"image": images, "input_length": input_length}


def build_crnn(config: dict[str, Any], num_classes: int) -> keras.Model:
    """Assemble full CRNN and return a Keras ``Model`` outputting CTC logits.

    Inputs: ``{"image": (B,H,W,3), "input_length": (B,)}``.
    Output shape: ``(B, T, num_classes)`` where ``num_classes`` includes blank.
    """
    model_cfg = config.get("model", {})
    image = keras.Input(shape=(None, None, 3), name="image")
    input_length = keras.Input(shape=(), dtype="int32", name="input_length")

    features = build_cnn_backbone(config, input_tensor=image)
    backbone_model = keras.Model(image, features, name="resnet18_backbone")

    pretrained = model_cfg.get("pretrained", "imagenet")
    if pretrained == "imagenet":
        try:
            _load_imagenet_weights(backbone_model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load ImageNet weights: %s", exc)

    seq = MapToSequence(name="map_to_sequence")(features)
    logits = build_transformer_head(config, num_classes, seq, input_length)
    model = keras.Model(
        inputs={"image": image, "input_length": input_length},
        outputs=logits,
        name="crnn_resnet18_transformer",
    )
    model.backbone = backbone_model  # type: ignore[attr-defined]
    return model


STAGE_PREFIXES = {
    "stem": ("stem_",),
    "layer1": ("layer1_",),
    "layer2": ("layer2_",),
    "layer3": ("layer3_",),
    "layer4": ("layer4_",),
}

# Sequence head layers kept trainable during backbone freeze phases.
SEQUENCE_HEAD_PREFIXES = (
    "seq_proj",
    "pos_encoding",
    "attention_mask",
    "transformer_",
    "logits",
)


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


def set_sequence_head_trainable(model: keras.Model, trainable: bool = True) -> None:
    """Ensure Transformer sequence head layers are trainable."""
    for layer in model.layers:
        if any(layer.name.startswith(p) for p in SEQUENCE_HEAD_PREFIXES):
            layer.trainable = trainable


class CTCModel(keras.Model):
    """Wrap a logits model with CTC train/test steps."""

    def __init__(self, crnn: keras.Model, blank_index: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.crnn = crnn
        self.blank_index = blank_index
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def call(self, inputs, training=False):
        if isinstance(inputs, dict):
            return self.crnn(inputs, training=training)
        return self.crnn(pack_crnn_inputs(inputs), training=training)

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        x, y = data
        label_length = x["label_length"]
        input_length = x["input_length"]
        labels = y
        crnn_inputs = {"image": x["image"], "input_length": input_length}

        with tf.GradientTape() as tape:
            logits = self.crnn(crnn_inputs, training=True)
            time_steps = tf.shape(logits)[1]
            logit_length = tf.minimum(
                tf.cast(input_length, tf.int32), time_steps
            )
            # Metal: CTC dense path forced to CPU inside mean_ctc_loss
            loss = mean_ctc_loss(
                labels,
                logits,
                blank_index=self.blank_index,
                label_length=label_length,
                logit_length=logit_length,
            )

        grads = tape.gradient(loss, self.crnn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.crnn.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        label_length = x["label_length"]
        input_length = x["input_length"]
        labels = y
        crnn_inputs = {"image": x["image"], "input_length": input_length}
        logits = self.crnn(crnn_inputs, training=False)
        time_steps = tf.shape(logits)[1]
        logit_length = tf.minimum(tf.cast(input_length, tf.int32), time_steps)
        loss = mean_ctc_loss(
            labels,
            logits,
            blank_index=self.blank_index,
            label_length=label_length,
            logit_length=logit_length,
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
