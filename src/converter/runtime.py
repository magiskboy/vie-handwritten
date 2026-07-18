"""TF-free OpenVINO inference for the CRNN OCR model.

Loads a static-shape IR (``fp16``/``int8``, batch 1 or 16), runs it on CPU, and
turns logits into text. Depends only on ``openvino`` + NumPy (+ OpenCV/underthesea
via preprocess/decode) — never TensorFlow or Keras.

Static shape means every image is right-padded to width ``W`` before inference;
each sample's logits are trimmed back to ``ceil(true_width / width_downsample)``
so padded columns cannot leak into the CTC decode (mirrors training's
``input_length``).

OpenVINO artifact layout (self-contained):

  <ov_dir>/
    charset.txt, config.yaml, meta.yaml, build_info.yaml
    lm/…  (when available; decode defaults to beam_lm)
    <precision>_b<batch>/model.xml|.bin
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from vie_handwritten.charset import Charset
from vie_handwritten.utils import (
    BUILD_INFO_NAME,
    CHARSET_NAME,
    CHECKPOINT_CONFIG_NAME,
    load_config,
    resolve_ctc_paths,
)

from converter.config import ArtifactPaths, META_NAME
from converter.decode import ArtifactDecoder

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_OV_REQUIRED = (CHARSET_NAME, CHECKPOINT_CONFIG_NAME, META_NAME, BUILD_INFO_NAME)


def imagenet_pad_value(preprocess_cfg: dict[str, Any]) -> float:
    """Scalar pad value matching training (``dataset.build_dataset``)."""
    if preprocess_cfg.get("normalize") == "imagenet":
        white = (1.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        return float(white.mean())
    return 0.0


def pad_width(arr: np.ndarray, width: int, pad_value: float) -> np.ndarray:
    """Right-pad ``(H, W, C)`` to ``width`` (or crop if wider)."""
    h, w = arr.shape[0], arr.shape[1]
    if w == width:
        return arr
    if w > width:
        return arr[:, :width]
    c = arr.shape[2]
    out = np.full((h, width, c), pad_value, dtype=np.float32)
    out[:, :w] = arr
    return out


def resolve_openvino_dir(ov_dir: str | Path) -> Path:
    """Validate a self-contained OpenVINO artifact directory."""
    root = Path(ov_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"OpenVINO dir must be a directory: {ov_dir}")
    missing = [name for name in _OV_REQUIRED if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"OpenVINO artifact {root} is incomplete; missing: {', '.join(missing)}"
        )
    return root.resolve()


class OpenVINOCR:
    """Compiled OpenVINO CRNN + decoder (default beam_lm), batch-fixed, CPU-only."""

    def __init__(
        self,
        compiled_model: Any,
        *,
        charset: Charset,
        config: dict[str, Any],
        batch: int,
        height: int,
        width: int,
        width_downsample: int,
        decoder: ArtifactDecoder | None = None,
    ):
        self.compiled_model = compiled_model
        self.charset = charset
        self.config = config
        self.batch = batch
        self.height = height
        self.width = width
        self.width_downsample = width_downsample
        self.pad_value = imagenet_pad_value(config.get("preprocess", {}))
        self.decoder = decoder or ArtifactDecoder.from_config(
            charset, config, prefer_beam_lm=True
        )
        self._output = compiled_model.output(0)

    @classmethod
    def from_dir(
        cls,
        ov_dir: str | Path,
        *,
        batch: int = 1,
        precision: str = "int8",
        device: str = "CPU",
    ) -> "OpenVINOCR":
        """Load a compiled model from ``<ov_dir>/<precision>_b<batch>/model.xml``."""
        import openvino as ov

        root = resolve_openvino_dir(ov_dir)
        paths = ArtifactPaths.for_dir(root)
        xml = paths.model_xml(precision, batch)
        if not xml.is_file():
            raise FileNotFoundError(f"IR not found: {xml}")

        meta = cls._load_meta(paths.meta_path)
        config = resolve_ctc_paths(load_config(root / CHECKPOINT_CONFIG_NAME), root)
        charset = Charset.from_file(root / CHARSET_NAME)

        if meta.get("num_classes") is not None and int(meta["num_classes"]) != charset.num_classes:
            raise ValueError(
                f"Charset num_classes={charset.num_classes} != meta.num_classes={meta['num_classes']}"
            )

        pp = config.get("preprocess", {})
        height = int(pp.get("target_height", 64))
        width = int(pp.get("max_width", 1536))
        wds = int(config.get("model", {}).get("width_downsample", 4))

        core = ov.Core()
        model = core.read_model(str(xml))
        compiled = core.compile_model(model, device)
        decoder = ArtifactDecoder.from_config(charset, config, prefer_beam_lm=True)
        logger.info(
            "Loaded OpenVINO IR %s on %s (batch=%d, decode=%s)",
            xml,
            device,
            batch,
            decoder.method,
        )
        return cls(
            compiled,
            charset=charset,
            config=config,
            batch=batch,
            height=height,
            width=width,
            width_downsample=wds,
            decoder=decoder,
        )

    @staticmethod
    def _load_meta(meta_path: Path) -> dict[str, Any]:
        if meta_path.is_file():
            with meta_path.open(encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        return {}

    def _run_batch(self, batch_arr: np.ndarray) -> np.ndarray:
        """Run one full padded batch ``(B, H, W, C)`` -> logits ``(B, T, C)``."""
        result = self.compiled_model(batch_arr)[self._output]
        return np.asarray(result)

    def predict_logits(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Preprocessed arrays ``(H, W_i, C)`` -> per-sample trimmed logits ``(T_i, C)``.

        Pads each image to the static width, packs them into fixed-size batches
        (zero-padding the final partial batch), then trims each output to its valid
        time-step budget derived from the true image width.
        """
        if not images:
            return []
        widths = [int(a.shape[1]) for a in images]
        padded = [pad_width(a, self.width, self.pad_value) for a in images]

        logits: list[np.ndarray] = []
        for start in range(0, len(padded), self.batch):
            chunk = padded[start : start + self.batch]
            k = len(chunk)
            batch_arr = np.zeros(
                (self.batch, self.height, self.width, chunk[0].shape[2]),
                dtype=np.float32,
            )
            for i, arr in enumerate(chunk):
                batch_arr[i] = arr
            out = self._run_batch(batch_arr)
            t_full = out.shape[1]
            for i in range(k):
                true_w = widths[start + i]
                t_valid = max(true_w // self.width_downsample, 1)
                t_valid = min(t_valid, t_full)
                logits.append(out[i, :t_valid])
        return logits

    def recognize_batch(self, images: list[np.ndarray]) -> list[str]:
        """Preprocessed arrays -> decoded, normalized Vietnamese text."""
        return [self.decoder.decode(lg[None, ...])[0] for lg in self.predict_logits(images)]

    def recognize(self, image: np.ndarray) -> str:
        """Single preprocessed image ``(H, W, C)`` -> text."""
        return self.recognize_batch([image])[0]

    def recognize_file(self, image_path: str | Path) -> str:
        """Load + preprocess an image file, then recognize (uses OpenCV, no TF)."""
        from vie_handwritten.preprocess import load_image, preprocess

        arr = preprocess(load_image(str(image_path)), self.config["preprocess"])
        return self.recognize(arr)
