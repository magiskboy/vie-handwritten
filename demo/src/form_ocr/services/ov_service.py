"""OpenVINO OCR service for the form demo."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable

from converter.runtime import OpenVINOCR

logger = logging.getLogger(__name__)


def default_ov_dir() -> Path:
    """Resolve vendored OpenVINO artifact directory.

    Order: ``FORM_OCR_OV_DIR`` env → ``demo/models/openvino`` next to the
    package source tree → ``./demo/models/openvino`` / ``./models/openvino`` cwd.
    """
    env = os.environ.get("FORM_OCR_OV_DIR")
    if env:
        return Path(env).expanduser().resolve()

    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "models" / "openvino",  # demo/src/form_ocr/services → demo/
        Path.cwd() / "demo" / "models" / "openvino",
        Path.cwd() / "models" / "openvino",
    ]
    for path in candidates:
        if path.is_dir():
            return path.resolve()
    return candidates[0].resolve()


def load_openvino(
    ov_dir: str | Path | None = None,
    *,
    precision: str = "int8",
    batch: int = 1,
) -> OpenVINOCR:
    """Load OpenVINOCR; fall back to fp16 if the requested precision IR is missing."""
    root = Path(ov_dir) if ov_dir else default_ov_dir()
    try:
        return OpenVINOCR.from_dir(root, batch=batch, precision=precision)
    except FileNotFoundError:
        if precision != "fp16":
            logger.warning("IR %s not found under %s; trying fp16", precision, root)
            return OpenVINOCR.from_dir(root, batch=batch, precision="fp16")
        raise


class OvService:
    """Thread-safe wrapper around OpenVINOCR load + access."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._busy = False
        self.ov: OpenVINOCR | None = None
        self.ov_dir: Path | None = None
        self.precision: str | None = None

    @property
    def ready(self) -> bool:
        return self.ov is not None

    def load(self, ov_dir: str | Path | None = None, *, precision: str = "int8") -> dict[str, Any]:
        root = Path(ov_dir) if ov_dir else default_ov_dir()
        used = precision
        try:
            ov = OpenVINOCR.from_dir(root, batch=1, precision=precision)
        except FileNotFoundError:
            if precision == "fp16":
                raise
            used = "fp16"
            ov = OpenVINOCR.from_dir(root, batch=1, precision="fp16")
        with self._lock:
            self.ov = ov
            self.ov_dir = root.resolve()
            self.precision = used
        return {
            "ov_dir": str(self.ov_dir),
            "precision": self.precision,
            "batch": ov.batch,
            "decode": ov.decoder.method,
            "height": ov.height,
            "width": ov.width,
        }

    def load_async(
        self,
        callback: Callable[[dict[str, Any] | None, BaseException | None], None],
        ov_dir: str | Path | None = None,
        *,
        precision: str = "int8",
    ) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        def _run() -> None:
            try:
                info = self.load(ov_dir, precision=precision)
                callback(info, None)
            except BaseException as exc:  # noqa: BLE001 — surface to UI
                callback(None, exc)
            finally:
                with self._lock:
                    self._busy = False

        threading.Thread(target=_run, daemon=True).start()
        return True
