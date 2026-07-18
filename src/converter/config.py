"""Convert/bench configuration: defaults + checkpoint-derived shape.

The OpenVINO settings live in ``configs/openvino.yaml`` (see
:data:`DEFAULT_CONFIG`); ``H``/``W`` for the static IR come from the checkpoint's
own ``preprocess`` block so the IR always matches how the model was trained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vie_handwritten.utils import load_checkpoint_config, load_config

DEFAULT_CONFIG: dict[str, Any] = {
    "openvino": {
        "batches": [1, 16],
        "calib_split": "val",
        "calib_samples": 300,
        "subset_size": 300,
        "precision_out": ["fp16", "int8"],
        "max_cer_drop": 0.01,
    },
    "bench": {
        "split": "val",
        "max_samples": 500,
        "warmup": 20,
        "iters": 100,
        "decode": "greedy",
    },
}

# Default IR sub-directory under a checkpoint. Sidecars live at the ov root:
# charset.txt, config.yaml, meta.yaml, build_info.yaml, lm/, <precision>_b<batch>/.
OPENVINO_SUBDIR = "openvino"
META_NAME = "meta.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_ov_config(path: str | Path | None) -> dict[str, Any]:
    """Load ``configs/openvino.yaml`` merged over the built-in defaults."""
    cfg = DEFAULT_CONFIG
    if path is not None:
        cfg = _deep_merge(cfg, load_config(path))
    return cfg


@dataclass
class ShapeSpec:
    """Static input shape ``[batch, height, width, channels]`` for one IR."""

    batch: int
    height: int
    width: int
    channels: int = 3
    width_downsample: int = 4

    @property
    def input_shape(self) -> list[int]:
        return [self.batch, self.height, self.width, self.channels]

    @property
    def max_time_steps(self) -> int:
        return max(self.width // self.width_downsample, 1)


def shape_from_checkpoint(checkpoint: str | Path, *, batch: int) -> ShapeSpec:
    """Derive the static shape from a checkpoint's ``preprocess`` config."""
    cfg = load_checkpoint_config(checkpoint)
    pp = cfg.get("preprocess", {})
    height = int(pp.get("target_height", 64))
    width = int(pp.get("max_width", 1536))
    channels = int(pp.get("channels", 3))
    wds = int(cfg.get("model", {}).get("width_downsample", 4))
    return ShapeSpec(
        batch=batch,
        height=height,
        width=width,
        channels=channels,
        width_downsample=wds,
    )


@dataclass
class ArtifactPaths:
    """Resolves the ``<checkpoint>/openvino/<precision>_b<batch>/`` layout."""

    root: Path
    checkpoint: Path = field(default_factory=Path)

    @classmethod
    def for_checkpoint(cls, checkpoint: str | Path) -> "ArtifactPaths":
        ckpt = Path(checkpoint).resolve()
        return cls(root=ckpt / OPENVINO_SUBDIR, checkpoint=ckpt)

    @classmethod
    def for_dir(cls, ov_dir: str | Path) -> "ArtifactPaths":
        return cls(root=Path(ov_dir).resolve())

    def variant_dir(self, precision: str, batch: int) -> Path:
        return self.root / f"{precision}_b{batch}"

    def model_xml(self, precision: str, batch: int) -> Path:
        return self.variant_dir(precision, batch) / "model.xml"

    @property
    def meta_path(self) -> Path:
        return self.root / META_NAME
