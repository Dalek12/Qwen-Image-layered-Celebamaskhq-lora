"""Generic layered training dataset for processed CelebAMask-HQ exports."""

from __future__ import annotations

from functools import partial
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


class CelebMaskHQGenericLayeredDataset(Dataset[dict[str, Any]]):
    """Load composite-plus-layer examples from metadata/layered_samples.jsonl."""

    def __init__(
        self,
        processed_root: str | Path,
        split: str = "train",
        *,
        resolution: int = 640,
        drop_warning_samples: bool = True,
        max_samples: int | None = None,
        max_layers: int | None = None,
        prompt_override: str | None = None,
    ) -> None:
        if resolution not in {640, 1024}:
            raise ValueError(f"resolution must be 640 or 1024 for Qwen-Image-Layered, got {resolution}")

        self.processed_root = Path(processed_root).resolve()
        self.split = split
        self.resolution = resolution
        self.drop_warning_samples = drop_warning_samples
        self.max_samples = max_samples
        self.max_layers = max_layers
        self.prompt_override = prompt_override

        metadata_path = self.processed_root / "metadata" / "layered_samples.jsonl"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Missing layered metadata: {metadata_path}. Rebuild with layered export enabled."
            )

        rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        filtered_rows: list[dict[str, Any]] = []
        for row in rows:
            if row["split"] != split:
                continue
            if drop_warning_samples and row.get("warnings"):
                continue
            if not row.get("layer_paths"):
                continue
            filtered_rows.append(row)

        if max_samples is not None:
            filtered_rows = filtered_rows[:max_samples]
        if not filtered_rows:
            raise ValueError(f"No layered samples found for split='{split}' with current filters.")

        self.rows = filtered_rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        composite_path = self.processed_root / row["composite_path"]
        layer_paths = [self.processed_root / layer_path for layer_path in row["layer_paths"]]
        layer_names = list(row["layer_names"])

        if self.max_layers is not None:
            layer_paths = layer_paths[: self.max_layers]
            layer_names = layer_names[: self.max_layers]

        conditioning_rgba = self._load_rgb_as_rgba(composite_path)
        layer_tensors = [self._load_rgba(layer_path) for layer_path in layer_paths]
        target_rgba_layers = torch.stack(layer_tensors, dim=0)
        layer_valid_mask = torch.ones(target_rgba_layers.shape[0], dtype=torch.float32)

        return {
            "sample_id": int(row["sample_id"]),
            "split": row["split"],
            "prompt": self.prompt_override or row.get("prompt", "decompose this portrait into editable portrait layers"),
            "conditioning_rgba": conditioning_rgba,
            "target_rgba_layers": target_rgba_layers,
            "layer_valid_mask": layer_valid_mask,
            "layer_names": layer_names,
            "layer_count": int(target_rgba_layers.shape[0]),
            "composite_path": str(composite_path),
            "layer_paths": [str(path) for path in layer_paths],
        }

    def _load_rgb_as_rgba(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image = image.resize((self.resolution, self.resolution), resample=Image.Resampling.BILINEAR)
            rgb = np.asarray(image, dtype=np.float32) / 255.0
        alpha = np.ones((self.resolution, self.resolution, 1), dtype=np.float32)
        rgba = np.concatenate([rgb, alpha], axis=-1)
        return torch.from_numpy(np.transpose(rgba, (2, 0, 1)))

    def _load_rgba(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("RGBA")
            image = image.resize((self.resolution, self.resolution), resample=Image.Resampling.BILINEAR)
            rgba = np.asarray(image, dtype=np.float32) / 255.0
        # Remove hidden RGB under transparent pixels so sparse layers do not carry arbitrary
        # color noise in regions where alpha is effectively zero after resampling.
        rgba[..., :3] *= rgba[..., 3:4]
        return torch.from_numpy(np.transpose(rgba, (2, 0, 1)))


def collate_generic_layered_batch(examples: list[dict[str, Any]], pad_to_max_layers: int | None = None) -> dict[str, Any]:
    if not examples:
        raise ValueError("Cannot collate an empty batch.")

    max_layers_in_batch = max(int(example["layer_count"]) for example in examples)
    max_layers = max_layers_in_batch if pad_to_max_layers is None else max(pad_to_max_layers, max_layers_in_batch)
    batch_size = len(examples)
    _, channels, height, width = examples[0]["target_rgba_layers"].shape

    padded_targets = torch.zeros((batch_size, max_layers, channels, height, width), dtype=examples[0]["target_rgba_layers"].dtype)
    padded_valid_mask = torch.zeros((batch_size, max_layers), dtype=torch.float32)

    for batch_index, example in enumerate(examples):
        layer_count = int(example["layer_count"])
        padded_targets[batch_index, :layer_count] = example["target_rgba_layers"]
        padded_valid_mask[batch_index, :layer_count] = example["layer_valid_mask"]

    return {
        "sample_id": torch.tensor([int(example["sample_id"]) for example in examples], dtype=torch.int64),
        "conditioning_rgba": torch.stack([example["conditioning_rgba"] for example in examples], dim=0),
        "target_rgba_layers": padded_targets,
        "layer_valid_mask": padded_valid_mask,
        "layer_count": torch.tensor([int(example["layer_count"]) for example in examples], dtype=torch.int64),
        "prompt": [example["prompt"] for example in examples],
        "layer_names": [example["layer_names"] for example in examples],
        "composite_path": [example["composite_path"] for example in examples],
        "layer_paths": [example["layer_paths"] for example in examples],
    }


def create_generic_layered_dataloader(
    processed_root: str | Path,
    split: str = "train",
    *,
    batch_size: int = 1,
    shuffle: bool | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    pad_to_max_layers: int | None = None,
    **dataset_kwargs: Any,
) -> DataLoader:
    """Create a dataloader for the generic layered export."""

    dataset = CelebMaskHQGenericLayeredDataset(processed_root, split=split, **dataset_kwargs)
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=partial(collate_generic_layered_batch, pad_to_max_layers=pad_to_max_layers),
    )


