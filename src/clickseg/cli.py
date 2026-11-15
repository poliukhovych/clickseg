import argparse
import json
import sys
from pathlib import Path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Click-based object segmentation with MobileSAM")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Inspect Python, architecture and MPS/CUDA support")
    doctor.add_argument("--output", type=Path, default=Path("runs/doctor.json"))
    download = commands.add_parser(
        "download", help="Download and verify official MobileSAM weights"
    )
    download.add_argument("--checkpoint", type=Path, default=Path("weights/mobile_sam.pt"))
    for name in ["smoke", "predict", "app"]:
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", type=Path, default=Path("weights/mobile_sam.pt"))
        command.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
        if name == "smoke":
            command.add_argument("--output", type=Path, default=Path("runs/smoke"))
        elif name == "predict":
            command.add_argument("--decoder", type=Path)
            command.add_argument("--image", type=Path, required=True)
            command.add_argument("--x", type=float, required=True)
            command.add_argument("--y", type=float, required=True)
            command.add_argument("--output", type=Path, default=Path("runs/predict"))
        else:
            command.add_argument("--decoder", type=Path)
            command.add_argument("--port", type=int, default=7860)
    prepare = commands.add_parser("prepare", help="Prepare a deterministic COCO subset")
    prepare.add_argument("--data-dir", type=Path, default=Path("data/coco2017"))
    prepare.add_argument("--output", type=Path, default=Path("data/pilot"))
    prepare.add_argument("--train-images", type=int, default=100)
    prepare.add_argument("--val-images", type=int, default=20)
    prepare.add_argument("--test-images", type=int, default=30)
    prepare.add_argument("--max-objects", type=int, default=3)
    prepare.add_argument("--seed", type=int, default=42)
    for name in ["cache", "evaluate", "train"]:
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", type=Path, default=Path("weights/mobile_sam.pt"))
        command.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
        command.add_argument("--cache", type=Path, default=Path("data/features"))
        if name == "cache":
            command.add_argument("--manifest", type=Path, nargs="+", required=True)
        elif name == "evaluate":
            command.add_argument("--manifest", type=Path, required=True)
            command.add_argument("--decoder", type=Path)
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--examples", type=int, default=6)
        else:
            command.add_argument("--train", type=Path, required=True)
            command.add_argument("--val", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--epochs", type=int, default=3)
            command.add_argument("--lr", type=float, default=1e-4)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument("--resume", type=Path)
    compare = commands.add_parser("compare", help="Compare metrics for identical masks and clicks")
    compare.add_argument("--before", type=Path, required=True)
    compare.add_argument("--after", type=Path, required=True)
    compare.add_argument("--output", type=Path, default=Path("runs/comparison.json"))
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            from clickseg.checkpoint import CHECKPOINT_SHA256, download_checkpoint

            path = download_checkpoint(args.checkpoint)
            print(f"Checkpoint verified: {path}\nSHA-256: {CHECKPOINT_SHA256}")
            return 0

        if args.command == "prepare":
            from clickseg.data import prepare

            report = prepare(args)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        if args.command == "compare":
            from clickseg.experiment import compare

            print(json.dumps(compare(args), ensure_ascii=False, indent=2))
            return 0

        from clickseg.runtime import choose_device, environment_report

        if args.command == "doctor":
            report = environment_report()
            write_json(args.output, report)
        elif args.command == "smoke":
            from clickseg.smoke import run_smoke

            report = run_smoke(args.checkpoint, choose_device(args.device), args.output)
            write_json(args.output / "report.json", report)
        elif args.command == "predict":
            from clickseg.prediction import predict_image

            report = predict_image(args, choose_device(args.device))
        elif args.command == "cache":
            from clickseg.features import cache_features

            report = cache_features(args, choose_device(args.device))
        elif args.command in ["train", "evaluate"]:
            from clickseg.experiment import evaluate, train

            report = {"train": train, "evaluate": evaluate}[args.command](
                args, choose_device(args.device)
            )
        else:
            from clickseg.app import launch

            launch(args.checkpoint, choose_device(args.device), args.port, args.decoder)
            return 0
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"clickseg: {error}", file=sys.stderr)
        return 1
