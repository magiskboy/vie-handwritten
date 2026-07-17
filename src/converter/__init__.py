"""Keras CRNN -> OpenVINO converter (INT8, batch1/16) + CPU benchmarks.

Public API:

* :func:`converter.quantize.convert_checkpoint` - full convert pipeline.
* :class:`converter.runtime.OpenVINOCR` - TF-free CPU inference on an IR.
* :func:`converter.bench_accuracy.bench_accuracy` / :func:`converter.bench_perf.bench_perf`.

Importing this package does not pull in TensorFlow; the heavy deps
(``openvino``, ``nncf``, ``tensorflow``) are imported lazily where needed.
"""

from converter.config import ArtifactPaths, ShapeSpec, load_ov_config
from converter.runtime import OpenVINOCR

__all__ = [
    "ArtifactPaths",
    "ShapeSpec",
    "load_ov_config",
    "OpenVINOCR",
]
