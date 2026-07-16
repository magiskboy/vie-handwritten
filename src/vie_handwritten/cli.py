"""CLI for Vietnamese handwritten OCR.

Entry point ``vie-ocr`` (see pyproject ``[project.scripts]``); also runnable as
``python -m vie_handwritten.cli``. The Makefile wraps common invocations.
"""

from __future__ import annotations

import argparse
import json
import logging

from vie_handwritten.utils import configure_runtime


def _floats(csv: str) -> list[float]:
    return [float(x) for x in csv.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vie-ocr",
        description="Vietnamese handwritten OCR (ResNet-18 CRNN + CTC)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    data_p = sub.add_parser("build-data", help="Build normalized JSONL manifests")
    data_p.add_argument("--config", default="configs/default.yaml")
    data_p.add_argument("--rebuild", action="store_true", help="Rebuild manifests")

    train_p = sub.add_parser("train", help="Train CRNN (2 phases: freeze CNN → train all)")
    train_p.add_argument("--config", default="configs/default.yaml")
    train_p.add_argument("--resume", default=None, help="Checkpoint weights to resume from")
    train_p.add_argument("--rebuild-data", action="store_true", help="Rebuild manifests first")

    lm_p = sub.add_parser("build-lm", help="Train KenLM syllable LM from train split")
    lm_p.add_argument("--config", default="configs/default.yaml")
    lm_p.add_argument("--order", type=int, default=4, help="n-gram order")
    lm_p.add_argument("--prune", type=int, default=0, help="prune threshold per order (0 = none)")

    eval_p = sub.add_parser("evaluate", help="Evaluate checkpoint (CER/WER)")
    eval_p.add_argument("--config", default="configs/default.yaml")
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--split", default="test", choices=["train", "val", "test"])
    eval_p.add_argument("--max-samples", type=int, default=None)
    eval_p.add_argument("--decode", default=None, choices=["greedy", "beam", "beam_lm"],
                        help="Override ctc.decode from config")

    infer_p = sub.add_parser("infer", help="Run OCR on an image")
    infer_p.add_argument("--image", required=True)
    infer_p.add_argument("--checkpoint", required=True)
    infer_p.add_argument("--config", default="configs/default.yaml")
    infer_p.add_argument("--decode", default=None, choices=["greedy", "beam", "beam_lm"],
                         help="Override ctc.decode from config")

    tune_p = sub.add_parser("tune-lm", help="Grid-search KenLM alpha/beta on a split")
    tune_p.add_argument("--config", default="configs/default.yaml")
    tune_p.add_argument("--checkpoint", required=True)
    tune_p.add_argument("--split", default="val", choices=["train", "val", "test"])
    tune_p.add_argument("--max-samples", type=int, default=300)
    tune_p.add_argument("--alphas", default="0.0,0.3,0.5,0.8,1.0")
    tune_p.add_argument("--betas", default="0.0,0.5,1.0,1.5")

    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    configure_runtime()  # GPU memory growth before any tensor allocation

    if args.command == "build-data":
        from vie_handwritten.dataset import build_manifests
        from vie_handwritten.utils import load_config

        config = load_config(args.config)
        paths = build_manifests(config)
        summary_file = paths["train"].parent / "summary.json"
        if summary_file.is_file():
            counts = json.loads(summary_file.read_text(encoding="utf-8")).get("counts", {})
            print(json.dumps(counts, ensure_ascii=False))
        for split, path in paths.items():
            print(f"  {split}: {path}")
    elif args.command == "train":
        from vie_handwritten.trainer import train

        train(args.config, resume_from=args.resume, rebuild_data=args.rebuild_data)
    elif args.command == "build-lm":
        from vie_handwritten.kenlm import build_lm
        from vie_handwritten.utils import load_config

        paths = build_lm(load_config(args.config), order=args.order, prune=args.prune)
        for name, path in paths.items():
            print(f"  {name}: {path}")
    elif args.command == "evaluate":
        from vie_handwritten.eval import evaluate

        metrics = evaluate(
            args.config,
            args.checkpoint,
            split=args.split,
            max_samples=args.max_samples,
            decode=args.decode,
        )
        print(f"split={args.split} n={metrics['n']} CER={metrics['cer']:.4f} WER={metrics['wer']:.4f}")
    elif args.command == "infer":
        from vie_handwritten.eval import infer

        print(infer(args.config, args.checkpoint, args.image, decode=args.decode))
    elif args.command == "tune-lm":
        from vie_handwritten.tune import tune_lm

        tune_lm(
            args.config,
            args.checkpoint,
            split=args.split,
            max_samples=args.max_samples,
            alphas=_floats(args.alphas),
            betas=_floats(args.betas),
        )


if __name__ == "__main__":
    main()
