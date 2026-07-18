"""Grid-search KenLM shallow-fusion weights (alpha, beta) on a manifest split.

Runs the CRNN forward pass once per image (logits are cached), then re-decodes
with each (alpha, beta) via ``decoder.reset_params`` so the sweep is cheap.
Prints CER/WER per grid point and the best setting to copy into the config.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vie_handwritten.charset import Charset
from vie_handwritten.dataset import ensure_manifests, load_manifest, resolve_image_path
from vie_handwritten.eval import evaluate_corpus
from vie_handwritten.model import build_crnn, load_crnn_weights
from vie_handwritten.postprocess import build_lm_decoder, ctc_lm_decode, normalize_text
from vie_handwritten.preprocess import load_image, preprocess
from vie_handwritten.utils import (
    charset_path,
    checkpoint_weights_path,
    load_checkpoint_config,
    resolve_checkpoint_dir,
    resolve_ctc_paths,
)

logger = logging.getLogger(__name__)

DEFAULT_ALPHAS = (0.0, 0.3, 0.5, 0.8, 1.0)
DEFAULT_BETAS = (0.0, 0.5, 1.0, 1.5)


def tune_lm(
    checkpoint: str | Path,
    *,
    split: str = "val",
    max_samples: int | None = 300,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    betas: Sequence[float] = DEFAULT_BETAS,
) -> dict[str, Any]:
    """Sweep (alpha, beta) on ``split`` and return per-point + best CER/WER."""
    root = resolve_checkpoint_dir(checkpoint)
    config = resolve_ctc_paths(load_checkpoint_config(root), root)
    ctc_cfg = config.setdefault("ctc", {})
    ctc_cfg["decode"] = "beam_lm"
    pp_cfg = config.get("postprocess", {})
    charset = Charset.from_file(charset_path(config, artifact_root=root))

    records = load_manifest(ensure_manifests(config)[split])
    if max_samples and len(records) > max_samples:
        records = records[:max_samples]

    crnn = build_crnn(config, num_classes=charset.num_classes)
    load_crnn_weights(crnn, checkpoint_weights_path(root))

    logger.info("Caching logits for %d %s samples ...", len(records), split)
    cached = []  # (logits[1, T, C], ref)
    for rec in records:
        arr = preprocess(load_image(str(resolve_image_path(config, rec))), config["preprocess"])
        cached.append((crnn.predict(arr[None, ...], verbose=0), rec["text"]))

    decoder = build_lm_decoder(charset, ctc_cfg)
    beam_width = int(ctc_cfg.get("beam_width", 100))
    tmin = float(ctc_cfg.get("token_min_logp", -5.0))
    bprune = float(ctc_cfg.get("beam_prune_logp", -10.0))

    results: list[tuple[float, float, float, float]] = []
    print(f"\n{'alpha':>6} {'beta':>6} {'CER':>8} {'WER':>8}")
    for alpha in alphas:
        for beta in betas:
            decoder.reset_params(alpha=alpha, beta=beta)
            hyps, refs = [], []
            for logits, ref in cached:
                text = ctc_lm_decode(
                    logits, decoder,
                    beam_width=beam_width, token_min_logp=tmin, beam_prune_logp=bprune,
                )[0]
                hyps.append(normalize_text(text, pp_cfg))
                refs.append(ref)
            m = evaluate_corpus(refs, hyps)
            results.append((alpha, beta, m["cer"], m["wer"]))
            print(f"{alpha:6.2f} {beta:6.2f} {m['cer']:8.4f} {m['wer']:8.4f}")

    best = min(results, key=lambda r: r[2])
    print(f"\nBest (by CER): alpha={best[0]} beta={best[1]} CER={best[2]:.4f} WER={best[3]:.4f}")
    return {"results": results, "best": best}
