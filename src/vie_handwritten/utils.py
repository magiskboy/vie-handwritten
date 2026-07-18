"""Shared helpers: config I/O, seeding, paths, GPU runtime, artifact bundles.

Standard Keras checkpoint layout (written by train, read by infer/eval/GUI):

  <checkpoint>/
    model.weights.h5   # Keras 3 requires the ``.weights.h5`` suffix
    config.yaml        # train config with paths rewritten relative to this dir
    charset.txt        # CTC vocabulary (self-contained)
    build_info.yaml    # git / runtime versions at save time
    lm/                # KenLM artifacts (copied when source files exist)
      vi.binary
      unigrams.txt
      vi_syllables.txt
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger(__name__)

_RUNTIME_CONFIGURED = False

WEIGHTS_NAME = "model.weights.h5"
CHECKPOINT_CONFIG_NAME = "config.yaml"
CHARSET_NAME = "charset.txt"
BUILD_INFO_NAME = "build_info.yaml"
LM_DIR = "lm"
LM_BINARY_NAME = "vi.binary"
LM_UNIGRAMS_NAME = "unigrams.txt"
LM_LEXICON_NAME = "vi_syllables.txt"

# Relative paths written into artifact ``config.yaml``.
ARTIFACT_CHARSET_REL = CHARSET_NAME
ARTIFACT_LM_BINARY_REL = f"{LM_DIR}/{LM_BINARY_NAME}"
ARTIFACT_LM_UNIGRAMS_REL = f"{LM_DIR}/{LM_UNIGRAMS_NAME}"
ARTIFACT_LM_LEXICON_REL = f"{LM_DIR}/{LM_LEXICON_NAME}"

_CHECKPOINT_REQUIRED = (WEIGHTS_NAME, CHECKPOINT_CONFIG_NAME, CHARSET_NAME, BUILD_INFO_NAME)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a nested dict."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Write config dict back to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)


def resolve_artifact_path(root: str | Path | None, relative: str | Path) -> Path:
    """Resolve ``relative`` under ``root`` when set; otherwise against project root.

    Absolute paths are returned as-is. When ``root`` is given and the file exists
    under it, that path wins (artifact-local). Otherwise fall back to ``abs_path``.
    """
    p = Path(relative)
    if p.is_absolute():
        return p
    if root is not None:
        candidate = Path(root) / p
        if candidate.is_file() or candidate.is_dir():
            return candidate.resolve()
        # Prefer artifact-relative even when missing so callers get clear errors.
        if (Path(root) / CHARSET_NAME).is_file() or (Path(root) / CHECKPOINT_CONFIG_NAME).is_file():
            return candidate.resolve()
    return abs_path(p)


def resolve_checkpoint_dir(checkpoint: str | Path) -> Path:
    """Validate a self-contained checkpoint directory."""
    root = Path(checkpoint)
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint must be a directory: {checkpoint}")
    missing = [name for name in _CHECKPOINT_REQUIRED if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint {root} is incomplete; missing: {', '.join(missing)}"
        )
    return root.resolve()


def checkpoint_weights_path(checkpoint: str | Path) -> Path:
    """Absolute path to ``model.weights.h5`` inside a checkpoint directory."""
    return resolve_checkpoint_dir(checkpoint) / WEIGHTS_NAME


def load_checkpoint_config(checkpoint: str | Path) -> dict[str, Any]:
    """Load ``config.yaml`` from a checkpoint directory."""
    return load_config(resolve_checkpoint_dir(checkpoint) / CHECKPOINT_CONFIG_NAME)


def save_checkpoint_config(config: dict[str, Any], checkpoint_dir: str | Path) -> Path:
    """Write config dict into ``checkpoint_dir/config.yaml`` (no path rewrite)."""
    path = Path(checkpoint_dir) / CHECKPOINT_CONFIG_NAME
    save_config(config, path)
    return path


def charset_path(config: dict[str, Any], *, artifact_root: str | Path | None = None) -> Path:
    """Absolute path to the charset file.

    When ``artifact_root`` is set, prefer ``<root>/charset.txt`` then the
    rewritten ``data.charset_path`` under that root.
    """
    if artifact_root is not None:
        local = Path(artifact_root) / CHARSET_NAME
        if local.is_file():
            return local.resolve()
        declared = config.get("data", {}).get("charset_path")
        if declared:
            return resolve_artifact_path(artifact_root, declared)
    return abs_path(config["data"]["charset_path"])


def file_sha256(path: str | Path) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_build_info(**extra: Any) -> dict[str, Any]:
    """Gather git / runtime metadata for ``build_info.yaml``."""
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "git_commit": None,
        "git_dirty": None,
    }
    try:
        root = project_root()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if commit.returncode == 0:
            info["git_commit"] = commit.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if dirty.returncode == 0:
            info["git_dirty"] = bool(dirty.stdout.strip())
    except OSError:
        pass

    try:
        import tensorflow as tf

        info["tensorflow_version"] = tf.__version__
        info["keras_version"] = getattr(tf.keras, "__version__", None)
    except Exception:  # noqa: BLE001
        pass

    try:
        import underthesea

        info["underthesea_version"] = getattr(underthesea, "__version__", None)
    except Exception:  # noqa: BLE001
        pass

    for key, value in extra.items():
        if value is not None:
            info[key] = value
    return info


def save_build_info(path: str | Path, info: dict[str, Any] | None = None, **extra: Any) -> Path:
    """Write ``build_info.yaml`` (merges ``info`` with :func:`collect_build_info`)."""
    path = Path(path)
    payload = collect_build_info(**extra)
    if info:
        payload.update(info)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, allow_unicode=True, sort_keys=False)
    return path


def _resolve_source_file(
    config_path: str | Path | None,
    *,
    source_root: Path | None,
    fallback_name: str,
) -> Path | None:
    """Find a sidecar source file under ``source_root`` or project paths."""
    if source_root is not None:
        local = source_root / fallback_name
        if local.is_file():
            return local.resolve()
        if config_path:
            via_root = resolve_artifact_path(source_root, config_path)
            if via_root.is_file():
                return via_root
    if config_path:
        via_proj = abs_path(config_path)
        if via_proj.is_file():
            return via_proj
    return None


def _copy_lm_into_artifact(
    config: dict[str, Any],
    root: Path,
    *,
    source_root: Path | None,
) -> dict[str, bool]:
    """Copy LM files into ``root/lm/`` when source files exist."""
    ctc = config.get("ctc", {})
    lm_dir = ensure_dir(root / LM_DIR)
    copied = {"lm_binary": False, "unigrams": False, "lexicon": False}

    sources = [
        ("lm_binary", ctc.get("lm_path"), f"{LM_DIR}/{LM_BINARY_NAME}", LM_BINARY_NAME),
        ("unigrams", ctc.get("unigrams_path"), f"{LM_DIR}/{LM_UNIGRAMS_NAME}", LM_UNIGRAMS_NAME),
        ("lexicon", ctc.get("lexicon_path"), f"{LM_DIR}/{LM_LEXICON_NAME}", LM_LEXICON_NAME),
    ]
    for key, cfg_path, artifact_rel, dest_name in sources:
        src_path = _resolve_source_file(cfg_path, source_root=source_root, fallback_name=artifact_rel)
        if src_path is None:
            if cfg_path:
                logger.warning("LM source missing, skip copy: %s", cfg_path)
            continue
        dest = lm_dir / dest_name
        if src_path.resolve() != dest.resolve():
            shutil.copyfile(src_path, dest)
        copied[key] = True
        logger.info("Copied %s → %s", src_path, dest)
    return copied


def rewrite_config_for_artifact(config: dict[str, Any], *, lm_copied: dict[str, bool]) -> dict[str, Any]:
    """Return a deep-ish copy of config with artifact-relative paths."""
    import copy

    out = copy.deepcopy(config)
    data = out.setdefault("data", {})
    data["charset_path"] = ARTIFACT_CHARSET_REL
    ctc = out.setdefault("ctc", {})
    if lm_copied.get("lm_binary"):
        ctc["lm_path"] = ARTIFACT_LM_BINARY_REL
    if lm_copied.get("unigrams"):
        ctc["unigrams_path"] = ARTIFACT_LM_UNIGRAMS_REL
    if lm_copied.get("lexicon"):
        ctc["lexicon_path"] = ARTIFACT_LM_LEXICON_REL
    return out


def resolve_ctc_paths(config: dict[str, Any], artifact_root: str | Path) -> dict[str, Any]:
    """Return a config copy with CTC/charset paths resolved under ``artifact_root``."""
    import copy

    out = copy.deepcopy(config)
    root = Path(artifact_root)
    out.setdefault("data", {})["charset_path"] = str(charset_path(out, artifact_root=root))
    ctc = out.setdefault("ctc", {})
    for key in ("lm_path", "unigrams_path", "lexicon_path"):
        rel = ctc.get(key)
        if rel:
            ctc[key] = str(resolve_artifact_path(root, rel))
    return out


def artifact_has_lm(root: str | Path) -> bool:
    """True when ``<root>/lm/vi.binary`` exists."""
    return (Path(root) / LM_DIR / LM_BINARY_NAME).is_file()


def save_sidecar_bundle(
    config: dict[str, Any],
    artifact_dir: str | Path,
    *,
    source_root: str | Path | None = None,
    **build_extra: Any,
) -> dict[str, Any]:
    """Write charset + LM + rewritten config + build_info into ``artifact_dir``.

    Used for both Keras checkpoints and OpenVINO sidecars. Does not touch weights/IR.
    ``source_root`` (e.g. a Keras checkpoint) is searched first for charset/LM files.
    Returns the rewritten config that was written.
    """
    root = ensure_dir(artifact_dir)
    src_root = Path(source_root).resolve() if source_root is not None else None

    src_charset = _resolve_source_file(
        config.get("data", {}).get("charset_path"),
        source_root=src_root,
        fallback_name=CHARSET_NAME,
    )
    if src_charset is None:
        raise FileNotFoundError(
            f"Charset not found for artifact {root} (source_root={src_root})"
        )
    dest_charset = root / CHARSET_NAME
    if src_charset.resolve() != dest_charset.resolve():
        shutil.copyfile(src_charset, dest_charset)

    lm_copied = _copy_lm_into_artifact(config, root, source_root=src_root)
    if (root / LM_DIR / LM_BINARY_NAME).is_file():
        lm_copied["lm_binary"] = True
    if (root / LM_DIR / LM_UNIGRAMS_NAME).is_file():
        lm_copied["unigrams"] = True
    if (root / LM_DIR / LM_LEXICON_NAME).is_file():
        lm_copied["lexicon"] = True

    rewritten = rewrite_config_for_artifact(config, lm_copied=lm_copied)
    save_config(rewritten, root / CHECKPOINT_CONFIG_NAME)
    save_build_info(root / BUILD_INFO_NAME, **build_extra)
    logger.info("Saved sidecar bundle → %s", root)
    return rewritten


def save_checkpoint_bundle(config: dict[str, Any], checkpoint_dir: str | Path, **build_extra: Any) -> dict[str, Any]:
    """Persist a self-contained Keras checkpoint sidecar next to weights."""
    return save_sidecar_bundle(config, checkpoint_dir, **build_extra)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow RNGs for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if missing; return Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_root() -> Path:
    """Return repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[2]


def abs_path(path: str | Path) -> Path:
    """Resolve a possibly-relative path against the project root."""
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def configure_runtime(*, memory_growth: bool = True) -> dict:
    """Enable GPU memory growth once before any tensor allocation (CUDA/CPU)."""
    global _RUNTIME_CONFIGURED
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    info = {
        "tensorflow": tf.__version__,
        "gpu_count": len(gpus),
        "gpus": [gpu.name for gpu in gpus],
    }
    if _RUNTIME_CONFIGURED:
        return info

    if gpus and memory_growth:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            logger.warning("Could not set GPU memory growth: %s", exc)

    if gpus:
        logger.info("TensorFlow %s using %d GPU(s): %s", tf.__version__, len(gpus), info["gpus"])
    else:
        logger.warning("No GPU detected — training will run on CPU.")

    _RUNTIME_CONFIGURED = True
    return info
