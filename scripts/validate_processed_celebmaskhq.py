"""CLI utilities for validating and inspecting CelebAMask-HQ processed outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess_celebmaskhq.validation import (
    inspect_processed_dataset,
    print_processed_stats,
    validate_processed_dataset,
    validate_source_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    source_parser = subparsers.add_parser("source", help="Validate source RGB and mask alignment.")
    source_parser.add_argument("--dataset-root", type=Path, default=Path("."))
    source_parser.add_argument("--split-seed", type=int, default=1337)
    source_parser.add_argument("--max-image-checks", type=int, default=64)
    source_parser.add_argument("--max-mask-checks", type=int, default=256)

    validate_parser = subparsers.add_parser("validate", help="Validate processed outputs and metadata.")
    validate_parser.add_argument("--processed-root", type=Path, default=Path("processed_celebmaskhq"))
    validate_parser.add_argument("--require-accessory", action="store_true")

    stats_parser = subparsers.add_parser("stats", help="Print stored processed dataset statistics.")
    stats_parser.add_argument("--processed-root", type=Path, default=Path("processed_celebmaskhq"))

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a deterministic random subset of samples.")
    inspect_parser.add_argument("--processed-root", type=Path, default=Path("processed_celebmaskhq"))
    inspect_parser.add_argument("--num-samples", type=int, default=5)
    inspect_parser.add_argument("--seed", type=int, default=1337)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "source":
        report = validate_source_dataset(
            args.dataset_root,
            split_seed=args.split_seed,
            max_image_checks=args.max_image_checks,
            max_mask_checks=args.max_mask_checks,
        )
    elif args.command == "validate":
        report = validate_processed_dataset(args.processed_root, require_accessory=args.require_accessory)
    elif args.command == "stats":
        report = print_processed_stats(args.processed_root)
    elif args.command == "inspect":
        report = inspect_processed_dataset(
            args.processed_root,
            num_samples=args.num_samples,
            seed=args.seed,
        )
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
