"""Layered RGBA training dataset for Qwen-Image-Layered LoRA fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from preprocess_celebmaskhq.constants import SLOT_NAMES


class CelebMaskHQLayeredDataset(Dataset[dict[str, Any]]):
    """Build Qwen-Image-Layered style supervision from processed CelebAMask-HQ outputs."""

    def __init__(
        self,
        processed_root: str | Path,
        split: str = "train",
        *,
        resolution: int = 640,
        drop_warning_samples: bool = True,
        include_combined_frame: bool = True,
        max_samples: int | None = None,
    ) -> None:
        if resolution not in {640, 1024}:
            raise ValueError(f"resolution must be 640 or 1024 for Qwen-Image-Layered, got {resolution}")

        self.processed_root = Path(processed_root).resolve()
        self.split = split
        self.resolution = resolution
        self.drop_warning_samples = drop_warning_samples
        self.include_combined_frame = include_combined_frame
        self.max_samples = max_samples

        metadata_path = self.processed_root / "metadata" / "samples.jsonl"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing processed metadata: {metadata_path}")

        rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        filtered_rows = []
        for row in rows:
            if row["split"] != split:
                continue
            if drop_warning_samples and row.get("warnings"):
                continue
            filtered_rows.append(row)

        if max_samples is not None:
            filtered_rows = filtered_rows[:max_samples]

        if not filtered_rows:
            raise ValueError(f"No samples found for split='{split}' with current filters.")

        self.rows = filtered_rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = self.processed_root / row["processed_image_path"]
        masks_path = self.processed_root / row["masks_path"]

        rgb = self._load_resized_rgb(image_path)
        slot_masks = self._load_resized_masks(masks_path)
        conditioning_rgba = self._build_conditioning_rgba(rgb)
        target_rgba_layers = self._build_target_rgba_layers(rgb, slot_masks)

        return {
            "sample_id": int(row["sample_id"]),
            "split": row["split"],
            "conditioning_rgba": torch.from_numpy(np.transpose(conditioning_rgba, (2, 0, 1))),
            "target_rgba_layers": torch.from_numpy(np.transpose(target_rgba_layers, (0, 3, 1, 2))),
            "slot_presence": torch.tensor(row["slot_presence"], dtype=torch.float32),
            "slot_pixel_counts": torch.tensor(row["slot_pixel_counts"], dtype=torch.int64),
            "slot_names": SLOT_NAMES,
            "image_path": str(image_path),
            "masks_path": str(masks_path),
        }

    def _load_resized_rgb(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image.resize((self.resolution, self.resolution), resample=Image.Resampling.BILINEAR)
            rgb = np.asarray(image, dtype=np.float32) / 255.0
        return rgb

    def _load_resized_masks(self, masks_path: Path) -> np.ndarray:
        masks = np.load(masks_path)["masks"].astype(np.uint8)
        resized_masks = []
        for slot_mask in masks:
            pil_mask = Image.fromarray(slot_mask * 255, mode="L")
            pil_mask = pil_mask.resize((self.resolution, self.resolution), resample=Image.Resampling.NEAREST)
            resized_masks.append((np.asarray(pil_mask, dtype=np.uint8) > 0).astype(np.float32))
        return np.stack(resized_masks, axis=0)

    @staticmethod
    def _build_conditioning_rgba(rgb: np.ndarray) -> np.ndarray:
        alpha = np.ones((*rgb.shape[:2], 1), dtype=np.float32)
        return np.concatenate([rgb, alpha], axis=-1)

    def _build_target_rgba_layers(self, rgb: np.ndarray, slot_masks: np.ndarray) -> np.ndarray:
        frames = []
        if self.include_combined_frame:
            frames.append(self._build_conditioning_rgba(rgb))

        for slot_index in range(slot_masks.shape[0]):
            alpha = slot_masks[slot_index][..., None]
            layer_rgb = rgb * alpha
            frames.append(np.concatenate([layer_rgb, alpha], axis=-1))

        return np.stack(frames, axis=0).astype(np.float32)


def create_layered_dataloader(
    processed_root: str | Path,
    split: str = "train",
    *,
    batch_size: int = 1,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    **dataset_kwargs: Any,
) -> DataLoader:
    """Create a dataloader for layered RGBA supervision."""

    dataset = CelebMaskHQLayeredDataset(processed_root, split=split, **dataset_kwargs)
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
