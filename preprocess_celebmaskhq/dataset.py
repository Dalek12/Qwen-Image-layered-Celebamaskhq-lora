"""Dataset discovery and split helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Iterable

from .constants import DEFAULT_SPLIT_RATIOS, IMAGE_SUFFIX, MASK_SUFFIX, SOURCE_CLASSES


@dataclass(slots=True)
class SourceSample:
    """Resolved source paths for a single CelebAMask-HQ sample."""

    sample_id: int
    image_path: Path
    mask_paths: dict[str, Path]
    split: str

    @property
    def sample_id_str(self) -> str:
        return f"{self.sample_id:05d}"


@dataclass(slots=True)
class DiscoverySummary:
    """High-level dataset inventory information."""

    image_count: int = 0
    mask_file_count: int = 0
    mask_id_count: int = 0
    non_png_files_skipped: int = 0
    malformed_mask_names: int = 0
    duplicate_masks: list[str] = field(default_factory=list)
    unknown_classes: list[str] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)


def discover_image_paths(dataset_root: Path) -> dict[int, Path]:
    """Return all JPG image paths keyed by integer sample id."""

    image_dir = dataset_root / "CelebA-HQ-img"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")

    image_paths: dict[int, Path] = {}
    for image_path in sorted(image_dir.glob(f"*{IMAGE_SUFFIX}")):
        try:
            sample_id = int(image_path.stem)
        except ValueError as exc:
            raise ValueError(f"Unexpected image filename: {image_path.name}") from exc
        if sample_id in image_paths:
            raise ValueError(f"Duplicate image id {sample_id}: {image_path}")
        image_paths[sample_id] = image_path
    return image_paths


def discover_mask_paths(dataset_root: Path) -> tuple[dict[int, dict[str, Path]], DiscoverySummary]:
    """Return all per-class mask paths keyed by sample id and class."""

    mask_root = dataset_root / "CelebAMask-HQ-mask-anno"
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Missing mask directory: {mask_root}")

    expected_classes = set(SOURCE_CLASSES)
    mask_paths: dict[int, dict[str, Path]] = {}
    class_counts = {class_name: 0 for class_name in SOURCE_CLASSES}
    summary = DiscoverySummary()

    for shard_dir in sorted(path for path in mask_root.iterdir() if path.is_dir()):
        for path in sorted(shard_dir.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() != MASK_SUFFIX:
                summary.non_png_files_skipped += 1
                continue

            name_parts = path.stem.split("_", 1)
            if len(name_parts) != 2 or not name_parts[0].isdigit():
                summary.malformed_mask_names += 1
                continue

            sample_id = int(name_parts[0])
            class_name = name_parts[1]
            if class_name not in expected_classes:
                summary.unknown_classes.append(class_name)
                continue

            sample_masks = mask_paths.setdefault(sample_id, {})
            if class_name in sample_masks:
                summary.duplicate_masks.append(str(path))
                continue

            sample_masks[class_name] = path
            class_counts[class_name] += 1
            summary.mask_file_count += 1

    summary.mask_id_count = len(mask_paths)
    summary.class_counts = class_counts
    summary.unknown_classes = sorted(set(summary.unknown_classes))
    return mask_paths, summary


def build_splits(
    sample_ids: Iterable[int],
    seed: int,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
) -> tuple[dict[str, list[int]], dict[int, str]]:
    """Build deterministic train/val/test splits from integer sample ids."""

    sample_ids = list(sorted(sample_ids))
    if not sample_ids:
        raise ValueError("Cannot build splits from an empty sample id set.")

    train_ratio, val_ratio, test_ratio = ratios
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, received {ratios}.")

    shuffled_ids = sample_ids.copy()
    random.Random(seed).shuffle(shuffled_ids)

    num_samples = len(shuffled_ids)
    train_count = int(num_samples * train_ratio)
    val_count = int(num_samples * val_ratio)
    test_count = num_samples - train_count - val_count

    train_ids = sorted(shuffled_ids[:train_count])
    val_ids = sorted(shuffled_ids[train_count : train_count + val_count])
    test_ids = sorted(shuffled_ids[train_count + val_count : train_count + val_count + test_count])

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    split_lookup: dict[int, str] = {}
    for split_name, split_ids in splits.items():
        for sample_id in split_ids:
            split_lookup[sample_id] = split_name
    return splits, split_lookup


def resolve_requested_ids(
    available_ids: Iterable[int],
    requested_ids: list[int] | None = None,
    ids_file: Path | None = None,
    limit: int | None = None,
) -> list[int]:
    """Resolve the subset of sample ids to process."""

    available_id_set = set(available_ids)
    if requested_ids:
        resolved_ids = sorted(dict.fromkeys(requested_ids))
    elif ids_file is not None:
        if not ids_file.is_file():
            raise FileNotFoundError(f"ID list file not found: {ids_file}")
        parsed_ids = [int(line.strip()) for line in ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        resolved_ids = sorted(dict.fromkeys(parsed_ids))
    else:
        resolved_ids = sorted(available_id_set)

    missing_ids = [sample_id for sample_id in resolved_ids if sample_id not in available_id_set]
    if missing_ids:
        raise ValueError(f"Requested sample ids are missing from the dataset: {missing_ids[:10]}")

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be a positive integer.")
        resolved_ids = resolved_ids[:limit]

    if not resolved_ids:
        raise ValueError("No sample ids were selected for processing.")

    return resolved_ids
