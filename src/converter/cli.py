"""``vie-ov`` CLI: convert Keras -> OpenVINO IR, benchmark accuracy & performance.

Unlike ``vie-ocr``, this entry point never calls ``configure_runtime`` and only
imports TensorFlow when a command actually needs it (``convert`` and the optional
Keras baselines), so benchmarking a converted model stays TF/GPU-free.
"""

from __future__ import annotations

import argparse
import json
import logging


def _ints(csv: str) -> list[int]:
    return [int(x) for x in csv.split(",") if x.strip()]


def _strs(csv: str) -> list[str]:
    return [x.strip() for x in csv.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vie-ov",
        description="Keras -> OpenVINO converter + CPU benchmarks (INT8, batch1/16)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    conv = sub.add_parser("convert", help="Convert a checkpoint to FP16 + INT8 IRs")
    conv.add_argument(
        "--checkpoint",
        required=True,
        help="Self-contained Keras checkpoint dir (weights, config, charset, build_info, lm/)",
    )
    conv.add_argument("--config", default="configs/openvino.yaml", help="OpenVINO settings YAML")
    conv.add_argument("--batches", default=None, help="Comma list, e.g. 1,16 (default from config)")

    acc = sub.add_parser("bench-accuracy", help="CER/WER: OV variants vs Keras")
    acc.add_argument(
        "--ov-dir",
        required=True,
        help="Self-contained OpenVINO artifact dir (IRs + charset, config, meta, build_info, lm/)",
    )
    acc.add_argument("--checkpoint", default=None, help="Keras checkpoint for baseline (optional)")
    acc.add_argument("--config", default="configs/openvino.yaml")
    acc.add_argument("--split", default=None, choices=["train", "val", "test"])
    acc.add_argument("--max-samples", type=int, default=None)
    acc.add_argument("--precisions", default="fp16,int8")
    acc.add_argument("--batch", type=int, default=1)
    acc.add_argument("--json", default=None, help="Write full report JSON to this path")

    perf = sub.add_parser("bench-perf", help="CPU latency/throughput per precision x batch")
    perf.add_argument("--ov-dir", required=True)
    perf.add_argument("--checkpoint", default=None, help="Keras checkpoint for CPU baseline (optional)")
    perf.add_argument("--config", default="configs/openvino.yaml")
    perf.add_argument("--precisions", default="fp16,int8")
    perf.add_argument("--batches", default="1,16")
    perf.add_argument("--warmup", type=int, default=None)
    perf.add_argument("--iters", type=int, default=None)
    perf.add_argument("--json", default=None, help="Write full report JSON to this path")

    return parser


def _run_convert(args: argparse.Namespace) -> None:
    from converter.config import load_ov_config
    from converter.quantize import convert_checkpoint

    ov_config = load_ov_config(args.config)
    batches = _ints(args.batches) if args.batches else None
    meta = convert_checkpoint(args.checkpoint, ov_config, batches=batches)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def _run_bench_accuracy(args: argparse.Namespace) -> None:
    from converter.bench_accuracy import bench_accuracy, format_report
    from converter.config import load_ov_config

    ov_config = load_ov_config(args.config)
    bench = ov_config["bench"]
    report = bench_accuracy(
        args.ov_dir,
        checkpoint=args.checkpoint,
        split=args.split or str(bench["split"]),
        max_samples=args.max_samples if args.max_samples is not None else int(bench["max_samples"]),
        precisions=tuple(_strs(args.precisions)),
        batch=args.batch,
    )
    print(format_report(report))
    if args.json:
        _write_json(args.json, report)


def _run_bench_perf(args: argparse.Namespace) -> None:
    from converter.bench_perf import bench_perf, format_report
    from converter.config import load_ov_config

    ov_config = load_ov_config(args.config)
    bench = ov_config["bench"]
    report = bench_perf(
        args.ov_dir,
        checkpoint=args.checkpoint,
        precisions=tuple(_strs(args.precisions)),
        batches=tuple(_ints(args.batches)),
        warmup=args.warmup if args.warmup is not None else int(bench["warmup"]),
        iters=args.iters if args.iters is not None else int(bench["iters"]),
        split=str(bench["split"]),
    )
    print(format_report(report))
    if args.json:
        _write_json(args.json, report)


def _write_json(path: str, report: dict) -> None:
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote report -> {p}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    if args.command == "convert":
        _run_convert(args)
    elif args.command == "bench-accuracy":
        _run_bench_accuracy(args)
    elif args.command == "bench-perf":
        _run_bench_perf(args)


if __name__ == "__main__":
    main()
