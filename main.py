#!/usr/bin/env python3
"""CLI for Vietnamese handwritten OCR (build-data / train / evaluate / infer)."""

from __future__ import annotations

import argparse
import json
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

    data_p = sub.add_parser("build-data", help="Build normalized JSONL manifests")
    data_p.add_argument("--config", default="configs/default.yaml")
    data_p.add_argument("--rebuild", action="store_true", help="Rebuild manifests")

    train_p = sub.add_parser("train", help="Train CRNN (2 phases: freeze CNN → train all)")
    train_p.add_argument("--config", default="configs/default.yaml")
    train_p.add_argument("--resume", default=None, help="Checkpoint weights to resume from")
    train_p.add_argument("--rebuild-data", action="store_true", help="Rebuild manifests first")

    eval_p = sub.add_parser("evaluate", help="Evaluate checkpoint (CER/WER)")
    eval_p.add_argument("--config", default="configs/default.yaml")
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--split", default="test", choices=["train", "val", "test"])
    eval_p.add_argument("--max-samples", type=int, default=None)

    infer_p = sub.add_parser("infer", help="Run OCR on an image")
    infer_p.add_argument("--image", required=True)
    infer_p.add_argument("--checkpoint", required=True)
    infer_p.add_argument("--config", default="configs/default.yaml")

    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

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
        from vie_handwritten.train import train

        train(args.config, resume_from=args.resume, rebuild_data=args.rebuild_data)
    elif args.command == "evaluate":
        from vie_handwritten.evaluate import evaluate

        metrics = evaluate(
            args.config, args.checkpoint, split=args.split, max_samples=args.max_samples
        )
        print(f"split={args.split} n={metrics['n']} CER={metrics['cer']:.4f} WER={metrics['wer']:.4f}")
    elif args.command == "infer":
        from vie_handwritten.evaluate import infer

        print(infer(args.config, args.checkpoint, args.image))


if __name__ == "__main__":
    main()
