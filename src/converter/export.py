"""Keras CRNN -> OpenVINO IR (static shape). Requires TensorFlow + openvino.

Only the ``convert`` command touches this module; the deploy/runtime path stays
TF-free. Weights are loaded from the checkpoint and ImageNet pretrained loading is
skipped (weights are overwritten anyway), so conversion needs no network access.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from vie_handwritten.charset import Charset
from vie_handwritten.utils import (
    charset_path,
    load_checkpoint_config,
    checkpoint_weights_path,
)

from converter.config import ShapeSpec

logger = logging.getLogger(__name__)


def force_cpu() -> None:
    """Hide GPUs from TensorFlow so LSTM traces the standard (non-cuDNN) path.

    The cuDNN LSTM kernel is emitted as ``CudnnRNNV3``, which the OpenVINO TF
    frontend cannot convert. Tracing on CPU keeps the portable op set. Must run
    before TensorFlow initializes its devices.
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def build_keras_net(checkpoint: str | Path) -> tuple[Any, dict[str, Any], Charset]:
    """Load the trained CRNN from a checkpoint dir (no ImageNet download)."""
    force_cpu()
    from vie_handwritten.model import build_crnn, load_crnn_weights

    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:  # devices already initialized
        pass

    config = load_checkpoint_config(checkpoint)
    # Skip ImageNet transfer: the checkpoint weights are loaded right after.
    build_cfg = {**config, "model": {**config.get("model", {}), "pretrained": "none"}}
    charset = Charset.from_file(charset_path(config))
    net = build_crnn(build_cfg, num_classes=charset.num_classes)
    net.trainable = False
    load_crnn_weights(net, checkpoint_weights_path(checkpoint))
    logger.info("Loaded Keras CRNN from %s (%d classes)", checkpoint, charset.num_classes)
    return net, config, charset


def static_input_model(net: Any, shape: ShapeSpec) -> Any:
    """Wrap ``net`` behind a fully static input ``[B, H, W, C]``.

    A concrete batch/height/width removes the dynamic dimensions that otherwise
    break OpenVINO's LSTM transpose-sinking passes, while reusing the trained
    weights (the layers are shared, not copied).
    """
    from tensorflow import keras

    static_in = keras.Input(batch_shape=tuple(shape.input_shape), name="image")
    return keras.Model(static_in, net(static_in, training=False), name="crnn_static")


def to_ov_model(net: Any, shape: ShapeSpec) -> Any:
    """Convert a Keras model to an OpenVINO model with a static input shape."""
    import openvino as ov

    logger.info("Converting to OpenVINO IR with input shape %s", shape.input_shape)
    static_model = static_input_model(net, shape)
    return ov.convert_model(static_model, share_weights=False)


def save_ir(ov_model: Any, xml_path: str | Path, *, compress_to_fp16: bool = True) -> Path:
    """Serialize an OpenVINO model to ``model.xml`` (+ ``.bin``)."""
    import openvino as ov

    xml_path = Path(xml_path)
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=compress_to_fp16)
    logger.info("Saved IR -> %s", xml_path)
    return xml_path
