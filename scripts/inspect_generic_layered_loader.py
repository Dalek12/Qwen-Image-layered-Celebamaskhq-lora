"""Inspect batches from the generic layered CelebAMask-HQ training dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_celebmaskhq import CelebMaskHQGenericLayeredDataset, create_generic_layered_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=Path("processed_celebmaskhq"))
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=640, choices=[640, 1024])
    parser.add_argument("--max-layers", type=int, default=8)
    parser.add_argument("--drop-warning-samples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = CelebMaskHQGenericLayeredDataset(
        args.processed_root,
        split=args.split,
        resolution=args.resolution,
        max_layers=args.max_layers,
        drop_warning_samples=args.drop_warning_samples,
    )
    loader = create_generic_layered_dataloader(
        args.processed_root,
        split=args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        resolution=args.resolution,
        max_layers=args.max_layers,
        pad_to_max_layers=args.max_layers,
        drop_warning_samples=args.drop_warning_samples,
    )

    report = {
        "dataset_len": len(dataset),
        "split": args.split,
        "batch_size": args.batch_size,
        "resolution": args.resolution,
        "max_layers": args.max_layers,
        "drop_warning_samples": args.drop_warning_samples,
        "batches": [],
    }

    for batch_index, batch in enumerate(loader):
        batch_report = {
            "batch_index": batch_index,
            "sample_ids": batch["sample_id"].tolist() if isinstance(batch["sample_id"], torch.Tensor) else batch["sample_id"],
            "conditioning_rgba_shape": list(batch["conditioning_rgba"].shape),
            "target_rgba_layers_shape": list(batch["target_rgba_layers"].shape),
            "layer_valid_mask_shape": list(batch["layer_valid_mask"].shape),
            "layer_counts": batch["layer_count"].tolist(),
            "layer_valid_sums": batch["layer_valid_mask"].sum(dim=1).tolist(),
            "first_prompt": batch["prompt"][0],
            "first_layer_names": batch["layer_names"][0],
        }
        report["batches"].append(batch_report)
        if batch_index + 1 >= args.num_batches:
            break

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
