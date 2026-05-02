"""Inspect batches from the processed CelebAMask-HQ PyTorch dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_celebmaskhq.dataset import CelebMaskHQProcessedDataset, create_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=Path("processed_celebmaskhq"))
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--drop-warning-samples", action="store_true")
    parser.add_argument("--keep-native-image-size", action="store_true")
    parser.add_argument("--no-label-map", action="store_true")
    parser.add_argument("--no-masks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = CelebMaskHQProcessedDataset(
        args.processed_root,
        split=args.split,
        load_masks=not args.no_masks,
        load_label_map=not args.no_label_map,
        resize_image_to_mask=not args.keep_native_image_size,
        drop_warning_samples=args.drop_warning_samples,
    )
    loader = create_dataloader(
        args.processed_root,
        split=args.split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        load_masks=not args.no_masks,
        load_label_map=not args.no_label_map,
        resize_image_to_mask=not args.keep_native_image_size,
        drop_warning_samples=args.drop_warning_samples,
    )

    report = {
        "dataset_len": len(dataset),
        "split": args.split,
        "batch_size": args.batch_size,
        "drop_warning_samples": args.drop_warning_samples,
        "resize_image_to_mask": not args.keep_native_image_size,
        "batches": [],
    }

    for batch_index, batch in enumerate(loader):
        batch_report = {
            "batch_index": batch_index,
            "sample_ids": batch["sample_id"].tolist() if isinstance(batch["sample_id"], torch.Tensor) else batch["sample_id"],
            "image_shape": list(batch["image"].shape),
            "image_dtype": str(batch["image"].dtype),
            "presence_shape": list(batch["presence"].shape),
            "presence_sums": batch["presence"].sum(dim=0).tolist(),
        }
        if "label_map" in batch:
            batch_report["label_map_shape"] = list(batch["label_map"].shape)
            batch_report["label_map_dtype"] = str(batch["label_map"].dtype)
        if "slot_masks" in batch:
            batch_report["slot_masks_shape"] = list(batch["slot_masks"].shape)
            batch_report["slot_masks_dtype"] = str(batch["slot_masks"].dtype)
        report["batches"].append(batch_report)
        if batch_index + 1 >= args.num_batches:
            break

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
