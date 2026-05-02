"""PyTorch dataset utilities for processed CelebAMask-HQ."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from preprocess_celebmaskhq.constants import SLOT_NAMES


@dataclass(slots=True)
class SampleRecord:
    """Structured metadata for a processed sample."""

    sample_id: int
    split: str
    image_path: Path
    label_path: Path
    masks_path: Path
    rgba_path: Path | None
    slot_presence: list[int]
    slot_pixel_counts: list[int]
    warnings: list[str]


class CelebMaskHQProcessedDataset(Dataset[dict[str, Any]]):
    """Load processed CelebAMask-HQ samples as PyTorch tensors."""

    def __init__(
        self,
        processed_root: str | Path,
        split: str = "train",
        *,
        load_masks: bool = True,
        load_label_map: bool = True,
        load_rgba: bool = False,
        resize_image_to_mask: bool = True,
        image_size: tuple[int, int] | None = None,
        normalize_images: bool = True,
        drop_warning_samples: bool = False,
    ) -> None:
        self.processed_root = Path(processed_root).resolve()
        self.split = split
        self.load_masks = load_masks
        self.load_label_map = load_label_map
        self.load_rgba = load_rgba
        self.resize_image_to_mask = resize_image_to_mask
        self.image_size = image_size
        self.normalize_images = normalize_images
        self.drop_warning_samples = drop_warning_samples

        self.mapping = self._read_json(self.processed_root / "metadata" / "mapping.json")
        self.stats = self._read_json(self.processed_root / "metadata" / "stats.json")
        if self.mapping["slot_order"] != list(SLOT_NAMES):
            raise ValueError(f"Unexpected slot order in mapping.json: {self.mapping['slot_order']}")

        sample_rows = self._read_jsonl(self.processed_root / "metadata" / "samples.jsonl")
        self.samples = self._build_sample_records(sample_rows)
        if not self.samples:
            raise ValueError(
                f"No samples available for split '{split}' in {self.processed_root}. "
                "Check the processed metadata or the chosen filters."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.samples[index]
        image = self._load_image(record.image_path)

        item: dict[str, Any] = {
            "sample_id": record.sample_id,
            "split": record.split,
            "image": image,
            "presence": torch.tensor(record.slot_presence, dtype=torch.float32),
            "slot_pixel_counts": torch.tensor(record.slot_pixel_counts, dtype=torch.int64),
            "has_warnings": bool(record.warnings),
            "warning_count": len(record.warnings),
            "warning_text": " | ".join(record.warnings),
            "image_path": str(record.image_path),
            "label_path": str(record.label_path),
            "masks_path": str(record.masks_path),
        }

        if self.load_label_map:
            item["label_map"] = self._load_label_map(record.label_path)
        if self.load_masks:
            item["slot_masks"] = self._load_slot_masks(record.masks_path)
        if self.load_rgba:
            if record.rgba_path is None:
                raise FileNotFoundError(
                    "RGBA layers were requested, but this processed dataset does not contain rgba_paths."
                )
            item["rgba_layers"] = self._load_rgba_layers(record.rgba_path)

        return item

    def _build_sample_records(self, sample_rows: list[dict[str, Any]]) -> list[SampleRecord]:
        records: list[SampleRecord] = []
        for row in sample_rows:
            if row["split"] != self.split:
                continue
            if self.drop_warning_samples and row.get("warnings"):
                continue
            records.append(
                SampleRecord(
                    sample_id=int(row["sample_id"]),
                    split=row["split"],
                    image_path=self.processed_root / row["processed_image_path"],
                    label_path=self.processed_root / row["label_path"],
                    masks_path=self.processed_root / row["masks_path"],
                    rgba_path=(self.processed_root / row["rgba_path"]) if row.get("rgba_path") else None,
                    slot_presence=list(row["slot_presence"]),
                    slot_pixel_counts=list(row["slot_pixel_counts"]),
                    warnings=list(row.get("warnings", [])),
                )
            )
        return records

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.resize_image_to_mask:
                mask_size = tuple(self.mapping["mask_resolution"])
                image = image.resize(mask_size, resample=Image.Resampling.BILINEAR)
            elif self.image_size is not None:
                image = image.resize(self.image_size, resample=Image.Resampling.BILINEAR)
            image_array = np.asarray(image, dtype=np.float32)

        image_tensor = torch.from_numpy(np.transpose(image_array, (2, 0, 1)))
        if self.normalize_images:
            image_tensor = image_tensor / 255.0
        return image_tensor

    @staticmethod
    def _load_label_map(path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            label_array = np.asarray(image, dtype=np.int64)
        return torch.from_numpy(label_array)

    @staticmethod
    def _load_slot_masks(path: Path) -> torch.Tensor:
        masks = np.load(path)["masks"].astype(np.float32)
        return torch.from_numpy(masks)

    @staticmethod
    def _load_rgba_layers(path: Path) -> torch.Tensor:
        rgba_layers = np.load(path)["rgba_layers"].astype(np.float32) / 255.0
        return torch.from_numpy(rgba_layers)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing JSON file: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing JSONL file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]


def create_dataloader(
    processed_root: str | Path,
    split: str = "train",
    *,
    batch_size: int = 4,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    **dataset_kwargs: Any,
) -> DataLoader:
    """Create a DataLoader for processed CelebAMask-HQ."""

    dataset = CelebMaskHQProcessedDataset(processed_root, split=split, **dataset_kwargs)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

