"""Validation utilities for source and processed CelebAMask-HQ datasets."""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image

from .constants import EXPECTED_IMAGE_SIZE, EXPECTED_MASK_SIZE, SLOT_NAMES
from .dataset import build_splits, discover_image_paths, discover_mask_paths


def validate_source_dataset(
    dataset_root: Path,
    split_seed: int = 1337,
    max_image_checks: int = 64,
    max_mask_checks: int = 256,
) -> dict[str, Any]:
    """Validate source-side alignment between RGB images and per-class masks."""

    dataset_root = dataset_root.resolve()
    image_paths = discover_image_paths(dataset_root)
    mask_paths, summary = discover_mask_paths(dataset_root)
    summary.image_count = len(image_paths)

    missing_images = sorted(set(mask_paths) - set(image_paths))
    missing_masks = sorted(set(image_paths) - set(mask_paths))
    all_ids = sorted(set(image_paths) & set(mask_paths))
    splits, _ = build_splits(all_ids, seed=split_seed)

    image_size_counts = _check_image_sizes(image_paths, max_checks=max_image_checks)
    mask_size_counts, mask_value_counts, checked_mask_files = _check_mask_content(mask_paths, max_checks=max_mask_checks)

    mask_count_values = [len(mask_paths[sample_id]) for sample_id in all_ids]
    source_report = {
        "image_count": len(image_paths),
        "mask_id_count": len(mask_paths),
        "mask_file_count": summary.mask_file_count,
        "missing_images_for_masks": missing_images,
        "images_without_masks": missing_masks,
        "non_png_files_skipped": summary.non_png_files_skipped,
        "malformed_mask_names": summary.malformed_mask_names,
        "duplicate_masks": summary.duplicate_masks,
        "unknown_classes": summary.unknown_classes,
        "class_counts": summary.class_counts,
        "min_masks_per_id": min(mask_count_values) if mask_count_values else 0,
        "max_masks_per_id": max(mask_count_values) if mask_count_values else 0,
        "split_counts": {split_name: len(split_ids) for split_name, split_ids in splits.items()},
        "expected_image_size": EXPECTED_IMAGE_SIZE,
        "expected_mask_size": EXPECTED_MASK_SIZE,
        "checked_image_count": min(max_image_checks, len(image_paths)),
        "checked_mask_file_count": checked_mask_files,
        "observed_image_sizes": image_size_counts,
        "observed_mask_sizes": mask_size_counts,
        "observed_mask_value_sets": mask_value_counts,
    }

    if missing_images:
        raise ValueError(f"Source validation failed: masks without RGB images: {missing_images[:10]}")
    if missing_masks:
        raise ValueError(f"Source validation failed: RGB images without masks: {missing_masks[:10]}")
    if summary.duplicate_masks:
        raise ValueError(f"Source validation failed: duplicate masks detected: {summary.duplicate_masks[:10]}")
    if summary.unknown_classes:
        raise ValueError(f"Source validation failed: unknown mask classes: {summary.unknown_classes}")
    if any(size_key != f"{EXPECTED_IMAGE_SIZE[0]}x{EXPECTED_IMAGE_SIZE[1]}" for size_key in image_size_counts):
        raise ValueError(f"Unexpected source image sizes found: {image_size_counts}")
    if any(size_key != f"{EXPECTED_MASK_SIZE[0]}x{EXPECTED_MASK_SIZE[1]}" for size_key in mask_size_counts):
        raise ValueError(f"Unexpected source mask sizes found: {mask_size_counts}")
    if any(value_key != "0,255" for value_key in mask_value_counts):
        raise ValueError(f"Unexpected source mask values found: {mask_value_counts}")

    return source_report


def validate_processed_dataset(processed_root: Path, require_accessory: bool = False) -> dict[str, Any]:
    """Validate merged labels, mask tensors, and per-sample metadata."""

    processed_root = processed_root.resolve()
    metadata_dir = processed_root / "metadata"
    mapping = _read_json(metadata_dir / "mapping.json")
    stats = _read_json(metadata_dir / "stats.json")
    samples = _read_jsonl(metadata_dir / "samples.jsonl")

    if mapping["slot_order"] != list(SLOT_NAMES):
        raise ValueError(f"Unexpected slot order in mapping.json: {mapping['slot_order']}")

    slot_presence_totals = [0 for _ in SLOT_NAMES]
    slot_pixel_totals = [0 for _ in SLOT_NAMES]
    total_overlap_pixels = 0
    samples_with_warnings = 0

    for record in samples:
        sample_id = record["sample_id"]
        label_path = processed_root / record["label_path"]
        masks_path = processed_root / record["masks_path"]

        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label map for sample {sample_id}: {label_path}")
        if not masks_path.is_file():
            raise FileNotFoundError(f"Missing slot masks for sample {sample_id}: {masks_path}")

        with Image.open(label_path) as image:
            label_map = np.asarray(image, dtype=np.uint8)
        if tuple(label_map.shape[::-1]) != EXPECTED_MASK_SIZE:
            raise ValueError(f"Unexpected label map size for sample {sample_id}: {label_map.shape}")

        unique_values = set(np.unique(label_map).tolist())
        expected_values = set(range(len(SLOT_NAMES)))
        if not unique_values.issubset(expected_values):
            raise ValueError(f"Label map for sample {sample_id} contains invalid slot ids: {sorted(unique_values)}")

        masks_npz = np.load(masks_path)
        slot_masks = masks_npz["masks"].astype(bool)
        expected_shape = (len(SLOT_NAMES), EXPECTED_MASK_SIZE[1], EXPECTED_MASK_SIZE[0])
        if slot_masks.shape != expected_shape:
            raise ValueError(f"Unexpected mask tensor shape for sample {sample_id}: {slot_masks.shape}")

        reconstructed_label_map = np.zeros_like(label_map)
        for slot_index in range(1, len(SLOT_NAMES)):
            reconstructed_label_map[slot_masks[slot_index]] = slot_index
        if not np.array_equal(reconstructed_label_map, label_map):
            raise ValueError(f"Label map mismatch for sample {sample_id}")

        expected_bg = ~np.any(slot_masks[1:], axis=0)
        if not np.array_equal(slot_masks[0], expected_bg):
            raise ValueError(f"Background channel mismatch for sample {sample_id}")

        presence_vector = [int(mask.any()) for mask in slot_masks]
        pixel_counts = [int(mask.sum()) for mask in slot_masks]
        if presence_vector != record["slot_presence"]:
            raise ValueError(f"Presence vector mismatch for sample {sample_id}")
        if pixel_counts != record["slot_pixel_counts"]:
            raise ValueError(f"Pixel count mismatch for sample {sample_id}")

        if record["warnings"]:
            samples_with_warnings += 1
        total_overlap_pixels += int(record["overlap_pixels"])
        for slot_index in range(len(SLOT_NAMES)):
            slot_presence_totals[slot_index] += presence_vector[slot_index]
            slot_pixel_totals[slot_index] += pixel_counts[slot_index]

    computed_stats = {
        "processed_sample_count": len(samples),
        "slot_presence_counts": dict(zip(SLOT_NAMES, slot_presence_totals, strict=True)),
        "slot_pixel_counts": dict(zip(SLOT_NAMES, slot_pixel_totals, strict=True)),
        "total_overlap_pixels": total_overlap_pixels,
        "samples_with_warnings": samples_with_warnings,
    }

    for key, value in computed_stats.items():
        if stats.get(key) != value:
            raise ValueError(f"stats.json mismatch for {key}: expected {value}, found {stats.get(key)}")

    layered_report = _validate_layered_export(processed_root, stats)

    if require_accessory and computed_stats["slot_presence_counts"]["ACCESSORY"] == 0:
        raise ValueError("Accessory slot is empty across the processed dataset subset.")

    return {
        "mapping": mapping,
        "stats": stats,
        "computed_stats": computed_stats,
        "layered_export": layered_report,
        "sample_count": len(samples),
        "require_accessory": require_accessory,
    }


def inspect_processed_dataset(
    processed_root: Path,
    num_samples: int = 5,
    seed: int = 1337,
) -> list[dict[str, Any]]:
    """Return a deterministic random subset of processed sample metadata."""

    processed_root = processed_root.resolve()
    samples = _read_jsonl(processed_root / "metadata" / "samples.jsonl")
    layered_lookup = {
        record["sample_id"]: record
        for record in _read_jsonl(processed_root / "metadata" / "layered_samples.jsonl")
    } if (processed_root / "metadata" / "layered_samples.jsonl").is_file() else {}
    if not samples:
        raise ValueError(f"No processed samples found in {processed_root}")

    rng = random.Random(seed)
    chosen = rng.sample(samples, k=min(num_samples, len(samples)))
    report: list[dict[str, Any]] = []
    for record in sorted(chosen, key=lambda row: row["sample_id"]):
        layered_record = layered_lookup.get(record["sample_id"])
        row = {
            "sample_id": record["sample_id"],
            "split": record["split"],
            "present_slots": [slot for slot, present in zip(SLOT_NAMES, record["slot_presence"], strict=True) if present],
            "slot_pixel_counts": record["slot_pixel_counts"],
            "warnings": record["warnings"],
            "label_path": record["label_path"],
            "masks_path": record["masks_path"],
        }
        if layered_record is not None:
            row["layer_count"] = layered_record["layer_count"]
            row["layer_names"] = layered_record["layer_names"]
            row["composite_path"] = layered_record["composite_path"]
        report.append(row)
    return report


def print_processed_stats(processed_root: Path) -> dict[str, Any]:
    """Return the stored processed dataset statistics."""

    processed_root = processed_root.resolve()
    return _read_json(processed_root / "metadata" / "stats.json")


def _validate_layered_export(processed_root: Path, stats: dict[str, Any]) -> dict[str, Any]:
    metadata_path = processed_root / "metadata" / "layered_samples.jsonl"
    if not metadata_path.is_file():
        return {"enabled": False, "sample_count": 0}

    layered_samples = _read_jsonl(metadata_path)
    layer_count_histogram: dict[str, int] = {}
    max_layers = 0
    for record in layered_samples:
        composite_path = processed_root / record["composite_path"]
        if not composite_path.is_file():
            raise FileNotFoundError(f"Missing layered composite for sample {record['sample_id']}: {composite_path}")
        with Image.open(composite_path) as image:
            if image.size != EXPECTED_MASK_SIZE:
                raise ValueError(f"Unexpected layered composite size for sample {record['sample_id']}: {image.size}")

        if len(record["layer_paths"]) != len(record["layer_names"]):
            raise ValueError(f"Layer path/name mismatch for sample {record['sample_id']}")
        if record["layer_count"] != len(record["layer_paths"]):
            raise ValueError(f"Layer count mismatch for sample {record['sample_id']}")
        if record["layer_count"] <= 0:
            raise ValueError(f"Layered export contains no target layers for sample {record['sample_id']}")

        for layer_path_str in record["layer_paths"]:
            layer_path = processed_root / layer_path_str
            if not layer_path.is_file():
                raise FileNotFoundError(f"Missing layered target for sample {record['sample_id']}: {layer_path}")
            with Image.open(layer_path) as image:
                if image.size != EXPECTED_MASK_SIZE:
                    raise ValueError(f"Unexpected layered target size for sample {record['sample_id']}: {image.size}")
                if image.mode != "RGBA":
                    raise ValueError(f"Layered target is not RGBA for sample {record['sample_id']}: {layer_path}")

        histogram_key = str(record["layer_count"])
        layer_count_histogram[histogram_key] = layer_count_histogram.get(histogram_key, 0) + 1
        max_layers = max(max_layers, int(record["layer_count"]))

    if stats.get("layered_sample_count") != len(layered_samples):
        raise ValueError(
            f"stats.json mismatch for layered_sample_count: expected {len(layered_samples)}, found {stats.get('layered_sample_count')}"
        )

    return {
        "enabled": True,
        "sample_count": len(layered_samples),
        "layer_count_histogram": layer_count_histogram,
        "max_layers": max_layers,
    }


def _check_image_sizes(image_paths: dict[int, Path], max_checks: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample_id in sorted(image_paths)[:max_checks]:
        with Image.open(image_paths[sample_id]) as image:
            size_key = f"{image.size[0]}x{image.size[1]}"
        counts[size_key] = counts.get(size_key, 0) + 1
    return counts


def _check_mask_content(mask_paths: dict[int, dict[str, Path]], max_checks: int) -> tuple[dict[str, int], dict[str, int], int]:
    size_counts: dict[str, int] = {}
    value_counts: dict[str, int] = {}
    checked = 0
    for sample_id in sorted(mask_paths):
        for class_name in sorted(mask_paths[sample_id]):
            with Image.open(mask_paths[sample_id][class_name]) as image:
                grayscale = image.convert("L")
                size_key = f"{grayscale.size[0]}x{grayscale.size[1]}"
                values = sorted(np.unique(np.asarray(grayscale, dtype=np.uint8)).tolist())
                value_key = ",".join(str(value) for value in values)
            size_counts[size_key] = size_counts.get(size_key, 0) + 1
            value_counts[value_key] = value_counts.get(value_key, 0) + 1
            checked += 1
            if checked >= max_checks:
                return size_counts, value_counts, checked
    return size_counts, value_counts, checked


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing JSONL file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
