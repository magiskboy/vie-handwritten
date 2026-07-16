"""KenLM language-model training for CTC post-processing.

Builds a syllable-level n-gram LM from the training-split transcripts so that
LM-fused beam search (see :mod:`vie_handwritten.ctc`) can favour linguistically
plausible Vietnamese output.

Artifacts (under the ``lm/`` dir, derived from ``ctc.lm_path``):
    corpus.txt   one NFC-normalized transcript per line (KenLM input)
    vi.arpa      the estimated n-gram model (text ARPA)
    vi.binary    the trie binary used at decode time
    unigrams.txt corpus tokens union the syllable lexicon (pyctcdecode unigrams)

KenLM binaries are built from the vendored submodule via
``make build-kenlm``; this module only invokes ``lmplz`` / ``build_binary``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import unicodedata
from pathlib import Path
from typing import Any

from vie_handwritten.dataset import ensure_manifests, load_manifest
from vie_handwritten.utils import project_root

logger = logging.getLogger(__name__)


def _abs(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else project_root() / p


def _kenlm_tools() -> tuple[dict[str, str], Path, Path]:
    """Locate lmplz / build_binary and the env needed to run them.

    Honors ``KENLM_BIN`` (dir of the binaries); defaults to the submodule build.
    KenLM links system Boost (from ``boost-devel``), so no extra library path
    is needed at runtime.
    """
    root = project_root()
    bin_dir = Path(os.environ.get("KENLM_BIN", root / "third_party" / "kenlm" / "build" / "bin"))
    lmplz = bin_dir / "lmplz"
    build_binary = bin_dir / "build_binary"
    if not lmplz.is_file() or not build_binary.is_file():
        raise FileNotFoundError(
            f"KenLM binaries not found in {bin_dir}. Build them with "
            "`make build-kenlm` (or set KENLM_BIN)."
        )
    return os.environ.copy(), lmplz, build_binary


def _lm_dir(config: dict[str, Any]) -> Path:
    lm_path = _abs(config.get("ctc", {}).get("lm_path", "lm/vi.binary"))
    return lm_path.parent


def build_corpus(config: dict[str, Any], out_path: Path) -> list[str]:
    """Write one NFC-normalized train transcript per line; return the lines."""
    manifests = ensure_manifests(config)
    records = load_manifest(manifests["train"])
    lines: list[str] = []
    for rec in records:
        text = unicodedata.normalize("NFC", rec["text"]).strip()
        if text:
            lines.append(text)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Corpus: %d lines -> %s", len(lines), out_path)
    return lines


def build_unigrams(corpus_lines: list[str], config: dict[str, Any], out_path: Path) -> int:
    """Merge corpus tokens with the syllable lexicon into a unigram list."""
    tokens: set[str] = set()
    for line in corpus_lines:
        tokens.update(line.split())
    lexicon_path = _abs(config.get("ctc", {}).get("lexicon_path", "data/charset/vi_syllables.txt"))
    if lexicon_path.is_file():
        for raw in lexicon_path.read_text(encoding="utf-8").splitlines():
            tok = unicodedata.normalize("NFC", raw).strip()
            if tok and not tok.startswith("##"):
                tokens.add(tok)
    else:
        logger.warning("Lexicon not found: %s (unigrams from corpus only)", lexicon_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sorted(tokens)) + "\n", encoding="utf-8")
    logger.info("Unigrams: %d tokens -> %s", len(tokens), out_path)
    return len(tokens)


def build_lm(config: dict[str, Any], *, order: int = 4, prune: int = 0) -> dict[str, Path]:
    """Train a KenLM n-gram model from the train split; return artifact paths."""
    env, lmplz, build_binary = _kenlm_tools()
    lm_dir = _lm_dir(config)
    lm_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = lm_dir / "corpus.txt"
    arpa_path = lm_dir / "vi.arpa"
    binary_path = _abs(config.get("ctc", {}).get("lm_path", "lm/vi.binary"))
    unigrams_path = _abs(config.get("ctc", {}).get("unigrams_path", str(lm_dir / "unigrams.txt")))

    lines = build_corpus(config, corpus_path)
    if not lines:
        raise RuntimeError("Empty training corpus; cannot train an LM.")
    build_unigrams(lines, config, unigrams_path)

    # --discount_fallback: small corpora often lack the higher-order n-gram counts
    # modified Kneser-Ney needs; fall back to lower discounts instead of erroring.
    lmplz_cmd = [str(lmplz), "-o", str(order), "--discount_fallback"]
    if prune:
        lmplz_cmd += ["--prune", *([str(prune)] * order)]
    logger.info("Running: %s < %s > %s", " ".join(lmplz_cmd), corpus_path, arpa_path)
    with corpus_path.open("rb") as fin, arpa_path.open("wb") as fout:
        subprocess.run(lmplz_cmd, stdin=fin, stdout=fout, env=env, check=True)

    bb_cmd = [str(build_binary), "trie", str(arpa_path), str(binary_path)]
    logger.info("Running: %s", " ".join(bb_cmd))
    subprocess.run(bb_cmd, env=env, check=True)

    logger.info("LM ready: %s", binary_path)
    return {
        "corpus": corpus_path,
        "arpa": arpa_path,
        "binary": binary_path,
        "unigrams": unigrams_path,
    }
