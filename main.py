#!/usr/bin/env python3
"""CLI for Vietnamese handwritten OCR (train / evaluate / infer)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from vie_handwritten.utils import configure_runtime  # noqa: E402

configure_runtime()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vie-ocr",
        description="Vietnamese handwritten OCR (ResNet-18 CRNN + CTC)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train CRNN model")
    train_p.add_argument("--config", default="configs/default.yaml")
    train_p.add_argument("--resume", default=None, help="Checkpoint weights to resume from")
    train_p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit dataset to N random samples (smoke / small-set training)",
    )

    eval_p = sub.add_parser("evaluate", help="Evaluate checkpoint (CER/WER)")
    eval_p.add_argument("--config", default="configs/default.yaml")
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--split", default="test", choices=["train", "val", "test"])

    infer_p = sub.add_parser("infer", help="Run OCR on an image")
    infer_p.add_argument("--image", required=True)
    infer_p.add_argument("--checkpoint", required=True)
    infer_p.add_argument("--config", default="configs/default.yaml")

    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        from vie_handwritten.train import train

        train(args.config, resume_from=args.resume, max_samples=args.max_samples)
    elif args.command == "evaluate":
        from vie_handwritten.evaluate import evaluate

        metrics = evaluate(args.config, args.checkpoint, split=args.split)
        print(
            f"split={args.split} n={metrics['n']} "
            f"CER={metrics['cer']:.4f} WER={metrics['wer']:.4f}"
        )
    elif args.command == "infer":
        from vie_handwritten.pipeline import OCRPipeline

        pipe = OCRPipeline.from_checkpoint(args.checkpoint, args.config)
        print(pipe.predict_path(args.image))
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
