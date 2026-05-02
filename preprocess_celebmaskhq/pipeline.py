"""Build pipeline for CelebAMask-HQ fixed-slot preprocessing and generic layered exports."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image

from .constants import (
    DEFAULT_SPLIT_RATIOS,
    EXPECTED_IMAGE_SIZE,
    EXPECTED_MASK_SIZE,
    SLOT_NAMES,
    SLOT_TO_SOURCE_CLASSES,
    SOURCE_CLASSES,
)
from .dataset import (
    DiscoverySummary,
    SourceSample,
    build_splits,
    discover_image_paths,
    discover_mask_paths,
    resolve_requested_ids,
)
from .preview import save_preview_sheet


DEFAULT_LAYERED_PROMPT = "decompose this portrait into editable portrait layers"
DEFAULT_LAYERED_SCHEME = "canonical_slots"

LAYERED_SCHEMES: dict[str, tuple[tuple[str, tuple[str, ...] | str], ...]] = {
    "canonical_slots": tuple((slot_name, (slot_name,)) for slot_name in SLOT_NAMES),
    "bg_hair_face": (
        ("BG", "__REMAINDER__"),
        ("HAIR", ("HAIR",)),
        ("FACE", ("FACE_SKIN", "EYES", "MOUTH")),
    ),
}


@dataclass(slots=True)
class BuildConfig:
    """Runtime configuration for preprocessing."""

    dataset_root: Path
    output_root: Path
    sample_ids: list[int] | None = None
    ids_file: Path | None = None
    limit: int | None = None
    preview_count: int = 8
    split_seed: int = 1337
    split_ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS
    write_rgba: bool = False
    copy_images: bool = False
    write_layered_export: bool = True
    include_background_layer: bool = True
    layered_scheme: str = DEFAULT_LAYERED_SCHEME
    layered_prompt: str = DEFAULT_LAYERED_PROMPT
    resume_existing: bool = False
    use_discovery_cache: bool = True
    refresh_discovery_cache: bool = False
    progress_every: int = 100


def run_build(config: BuildConfig) -> dict[str, Any]:
    """Execute preprocessing and write all outputs."""

    dataset_root = config.dataset_root.resolve()
    output_root = config.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    layered_scheme = _normalize_layered_scheme(config.layered_scheme)

    discovery_cache_path = output_root / "metadata" / "source_discovery_cache.json"
    discovery_cache_used = False
    cached_discovery = None
    if config.use_discovery_cache and not config.refresh_discovery_cache:
        cached_discovery = _try_load_source_discovery_cache(dataset_root, discovery_cache_path)

    if cached_discovery is not None:
        image_paths, mask_paths, discovery_summary = cached_discovery
        discovery_cache_used = True
    else:
        image_paths = discover_image_paths(dataset_root)
        mask_paths, discovery_summary = discover_mask_paths(dataset_root)
        discovery_summary.image_count = len(image_paths)
        if config.use_discovery_cache:
            _write_source_discovery_cache(
                dataset_root=dataset_root,
                cache_path=discovery_cache_path,
                image_paths=image_paths,
                mask_paths=mask_paths,
                discovery_summary=discovery_summary,
            )

    all_ids = sorted(set(image_paths) & set(mask_paths))
    missing_images = sorted(set(mask_paths) - set(image_paths))
    ids_without_masks = sorted(set(image_paths) - set(mask_paths))
    if not all_ids:
        raise ValueError("No overlapping sample ids were found between RGB images and masks.")

    requested_ids = resolve_requested_ids(
        available_ids=all_ids,
        requested_ids=config.sample_ids,
        ids_file=config.ids_file,
        limit=config.limit,
    )
    splits, split_lookup = build_splits(all_ids, seed=config.split_seed, ratios=config.split_ratios)

    output_paths = _prepare_output_directories(
        output_root,
        write_rgba=config.write_rgba,
        write_layered_export=config.write_layered_export,
    )
    actual_image_mode = "copy" if config.copy_images else "hardlink"
    image_write_counts = {"hardlink": 0, "copy": 0}
    sample_records: list[dict[str, Any]] = []
    layered_records: list[dict[str, Any]] = []

    slot_presence_totals = [0 for _ in SLOT_NAMES]
    slot_pixel_totals = [0 for _ in SLOT_NAMES]
    pair_overlap_totals = {pair_key: 0 for pair_key in _pair_overlap_keys()}
    layered_count_histogram: dict[str, int] = {}
    total_overlap_pixels = 0
    total_warning_count = 0
    resumed_existing_sample_count = 0
    newly_processed_sample_count = 0

    preview_ids = set(requested_ids[: max(config.preview_count, 0)])
    total_requested = len(requested_ids)

    for sample_index, sample_id in enumerate(requested_ids, start=1):
        sample = SourceSample(
            sample_id=sample_id,
            image_path=image_paths[sample_id],
            mask_paths=mask_paths[sample_id],
            split=split_lookup[sample_id],
        )
        reused_existing = False
        existing_records: tuple[dict[str, Any], dict[str, Any] | None, str] | None = None
        if config.resume_existing:
            existing_records = _try_load_existing_sample_records(
                dataset_root=dataset_root,
                output_root=output_root,
                output_paths=output_paths,
                sample=sample,
                write_rgba=config.write_rgba,
                write_layered_export=config.write_layered_export,
                include_background_layer=config.include_background_layer,
                layered_scheme=layered_scheme,
                layered_prompt=config.layered_prompt,
                require_preview=sample_id in preview_ids,
            )

        if existing_records is not None:
            sample_record, layered_record, image_mode_used = existing_records
            reused_existing = True
            resumed_existing_sample_count += 1
        else:
            processed = _process_sample(sample, write_rgba=config.write_rgba)

            image_mode_used = _link_or_copy_image(
                source_path=sample.image_path,
                target_path=output_paths["images"] / f"{sample_id:05d}.jpg",
                prefer_copy=config.copy_images,
            )

            label_output_path = output_paths["labels_7class"] / f"{sample_id:05d}.png"
            Image.fromarray(processed["label_map"], mode="L").save(label_output_path)

            masks_output_path = output_paths["masks_7slot"] / f"{sample_id:05d}.npz"
            np.savez_compressed(
                masks_output_path,
                masks=processed["slot_masks"].astype(bool),
                slot_names=np.asarray(SLOT_NAMES),
            )

            rgba_relative_path: str | None = None
            if config.write_rgba:
                rgba_output_path = output_paths["rgba_layers_7slot"] / f"{sample_id:05d}.npz"
                np.savez_compressed(
                    rgba_output_path,
                    rgba_layers=processed["rgba_layers"].astype(np.uint8),
                    slot_names=np.asarray(SLOT_NAMES),
                )
                rgba_relative_path = str(rgba_output_path.relative_to(output_root))

            if sample_id in preview_ids:
                preview_output_path = output_paths["previews"] / f"{sample_id:05d}.png"
                save_preview_sheet(
                    image_rgb=processed["preview_rgb"],
                    label_map=processed["label_map"],
                    slot_masks=processed["slot_masks"],
                    output_path=preview_output_path,
                    sample_id=f"{sample_id:05d}",
                )

            layered_record = None
            if config.write_layered_export:
                layered_record = _write_layered_sample(
                    output_root=output_root,
                    output_paths=output_paths,
                    sample=sample,
                    processed=processed,
                    include_background_layer=config.include_background_layer,
                    layered_scheme=layered_scheme,
                    layered_prompt=config.layered_prompt,
                )

            sample_record = _build_sample_record(
                dataset_root=dataset_root,
                output_root=output_root,
                output_paths=output_paths,
                sample=sample,
                image_size=processed["image_size"],
                slot_presence=processed["slot_presence"],
                slot_pixel_counts=processed["slot_pixel_counts"],
                overlap_pixels=processed["overlap_pixels"],
                pair_overlap_pixels=processed["pair_overlap_pixels"],
                warnings=processed["warnings"],
                rgba_relative_path=rgba_relative_path,
                layered_record=layered_record,
            )
            _write_sample_record_cache(
                output_paths=output_paths,
                sample=sample,
                sample_record=sample_record,
                layered_record=layered_record,
                image_mode_used=image_mode_used,
                write_rgba=config.write_rgba,
                write_layered_export=config.write_layered_export,
                include_background_layer=config.include_background_layer,
                layered_scheme=layered_scheme,
                layered_prompt=config.layered_prompt,
            )
            newly_processed_sample_count += 1

        image_write_counts[image_mode_used] += 1
        if image_mode_used != actual_image_mode and actual_image_mode == "hardlink":
            actual_image_mode = "hardlink_with_copy_fallback"

        sample_records.append(sample_record)
        if layered_record is not None:
            layered_records.append(layered_record)
            histogram_key = str(layered_record["layer_count"])
            layered_count_histogram[histogram_key] = layered_count_histogram.get(histogram_key, 0) + 1

        total_warning_count += len(sample_record["warnings"])
        total_overlap_pixels += sample_record["overlap_pixels"]
        for slot_index, present in enumerate(sample_record["slot_presence"]):
            slot_presence_totals[slot_index] += int(present)
        for slot_index, pixel_count in enumerate(sample_record["slot_pixel_counts"]):
            slot_pixel_totals[slot_index] += int(pixel_count)
        for pair_key, pair_count in sample_record["pair_overlap_pixels"].items():
            pair_overlap_totals[pair_key] += int(pair_count)
        if config.progress_every > 0 and (
            sample_index == 1
            or sample_index % config.progress_every == 0
            or sample_index == total_requested
        ):
            action = "reused" if reused_existing else "processed"
            print(
                f"[{sample_index}/{total_requested}] {action} sample {sample.sample_id_str} | "
                f"reused={resumed_existing_sample_count} new={newly_processed_sample_count}",
                flush=True,
            )

    mapping_metadata = {
        "slot_order": list(SLOT_NAMES),
        "source_classes": list(SOURCE_CLASSES),
        "slot_to_source_classes": {slot: list(classes) for slot, classes in SLOT_TO_SOURCE_CLASSES.items()},
        "background_definition": "Complement of merged foreground slots.",
        "label_priority": list(SLOT_NAMES),
        "image_resolution": list(EXPECTED_IMAGE_SIZE),
        "mask_resolution": list(EXPECTED_MASK_SIZE),
        "split_seed": config.split_seed,
        "split_ratios": {
            "train": config.split_ratios[0],
            "val": config.split_ratios[1],
            "test": config.split_ratios[2],
        },
        "image_storage_mode": actual_image_mode,
        "write_rgba": config.write_rgba,
        "layered_export": {
            "enabled": config.write_layered_export,
            "format": "generic_layered_pngs_v1",
            "composite_dir": "layered_composites",
            "layer_dir": "layered_layers",
            "metadata_path": "metadata/layered_samples.jsonl",
            "include_background_layer": config.include_background_layer,
            "scheme": layered_scheme,
            "target_layer_names": _layered_scheme_target_names(layered_scheme, config.include_background_layer),
            "layer_group_definitions": _layered_scheme_metadata(layered_scheme),
            "layer_order_policy": "scheme_order_present_layers",
            "default_prompt": config.layered_prompt,
        },
    }

    processed_splits = _build_processed_splits(sample_records)
    stats_metadata = {
        "processed_sample_count": len(sample_records),
        "requested_sample_count": len(requested_ids),
        "selection_limit": config.limit,
        "source_image_count": len(image_paths),
        "source_masked_id_count": len(mask_paths),
        "source_ids_with_both_image_and_masks": len(all_ids),
        "source_missing_image_id_count": len(missing_images),
        "source_missing_mask_id_count": len(ids_without_masks),
        "source_missing_image_ids_preview": missing_images[:10],
        "source_missing_mask_ids_preview": ids_without_masks[:10],
        "source_mask_file_count": discovery_summary.mask_file_count,
        "source_non_png_files_skipped": discovery_summary.non_png_files_skipped,
        "source_malformed_mask_names": discovery_summary.malformed_mask_names,
        "source_duplicate_masks": discovery_summary.duplicate_masks,
        "source_unknown_classes": discovery_summary.unknown_classes,
        "source_class_counts": discovery_summary.class_counts,
        "split_counts_full_dataset": {split_name: len(split_ids) for split_name, split_ids in splits.items()},
        "split_counts_processed_subset": _count_by_key(sample_records, "split"),
        "slot_presence_counts": dict(zip(SLOT_NAMES, slot_presence_totals, strict=True)),
        "slot_pixel_counts": dict(zip(SLOT_NAMES, slot_pixel_totals, strict=True)),
        "total_overlap_pixels": total_overlap_pixels,
        "pair_overlap_pixels": pair_overlap_totals,
        "samples_with_warnings": sum(1 for record in sample_records if record["warnings"]),
        "total_warning_count": total_warning_count,
        "image_write_counts": image_write_counts,
        "processed_sample_ids": [record["sample_id"] for record in sample_records],
        "processed_split_ids": processed_splits,
        "layered_sample_count": len(layered_records),
        "layered_layer_count_histogram": layered_count_histogram,
        "layered_max_layers": max((record["layer_count"] for record in layered_records), default=0),
        "resumed_existing_sample_count": resumed_existing_sample_count,
        "newly_processed_sample_count": newly_processed_sample_count,
        "resume_existing": config.resume_existing,
        "source_discovery_cache_used": discovery_cache_used,
        "source_discovery_cache_path": str(discovery_cache_path.relative_to(output_root)),
    }

    _write_json(output_paths["metadata"] / "mapping.json", mapping_metadata)
    _write_json(output_paths["metadata"] / "stats.json", stats_metadata)
    _write_json(output_paths["metadata"] / "splits.json", splits)
    _write_json(output_paths["metadata"] / "processed_splits.json", processed_splits)
    _write_jsonl(output_paths["metadata"] / "samples.jsonl", sample_records)
    if config.write_layered_export:
        _write_jsonl(output_paths["metadata"] / "layered_samples.jsonl", layered_records)

    return {
        "mapping": mapping_metadata,
        "stats": stats_metadata,
        "splits": splits,
        "samples": sample_records,
        "layered_samples": layered_records,
        "output_root": str(output_root),
    }


def _prepare_output_directories(output_root: Path, write_rgba: bool, write_layered_export: bool) -> dict[str, Path]:
    output_paths = {
        "images": output_root / "images",
        "labels_7class": output_root / "labels_7class",
        "masks_7slot": output_root / "masks_7slot",
        "metadata": output_root / "metadata",
        "sample_record_cache": output_root / "metadata" / "sample_records",
        "previews": output_root / "previews",
    }
    if write_rgba:
        output_paths["rgba_layers_7slot"] = output_root / "rgba_layers_7slot"
    if write_layered_export:
        output_paths["layered_composites"] = output_root / "layered_composites"
        output_paths["layered_layers"] = output_root / "layered_layers"

    for directory in output_paths.values():
        directory.mkdir(parents=True, exist_ok=True)
    return output_paths


def _try_load_source_discovery_cache(
    dataset_root: Path,
    cache_path: Path,
) -> tuple[dict[int, Path], dict[int, dict[str, Path]], DiscoverySummary] | None:
    if not cache_path.is_file():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("version") != 1:
        return None
    if payload.get("source_classes") != list(SOURCE_CLASSES):
        return None

    try:
        image_paths = {
            int(sample_id): dataset_root / relative_path
            for sample_id, relative_path in payload["image_paths"].items()
        }
        mask_paths = {
            int(sample_id): {
                class_name: dataset_root / relative_path
                for class_name, relative_path in sample_masks.items()
            }
            for sample_id, sample_masks in payload["mask_paths"].items()
        }
        summary_payload = payload["discovery_summary"]
    except (KeyError, TypeError, ValueError):
        return None

    if not image_paths or not mask_paths:
        return None
    discovery_summary = DiscoverySummary(
        image_count=int(summary_payload.get("image_count", len(image_paths))),
        mask_file_count=int(summary_payload.get("mask_file_count", 0)),
        mask_id_count=int(summary_payload.get("mask_id_count", len(mask_paths))),
        non_png_files_skipped=int(summary_payload.get("non_png_files_skipped", 0)),
        malformed_mask_names=int(summary_payload.get("malformed_mask_names", 0)),
        duplicate_masks=list(summary_payload.get("duplicate_masks", [])),
        unknown_classes=list(summary_payload.get("unknown_classes", [])),
        class_counts=dict(summary_payload.get("class_counts", {})),
    )
    return image_paths, mask_paths, discovery_summary


def _write_source_discovery_cache(
    *,
    dataset_root: Path,
    cache_path: Path,
    image_paths: dict[int, Path],
    mask_paths: dict[int, dict[str, Path]],
    discovery_summary: DiscoverySummary,
) -> None:
    payload = {
        "version": 1,
        "source_classes": list(SOURCE_CLASSES),
        "image_paths": {
            str(sample_id): path.relative_to(dataset_root).as_posix()
            for sample_id, path in sorted(image_paths.items())
        },
        "mask_paths": {
            str(sample_id): {
                class_name: path.relative_to(dataset_root).as_posix()
                for class_name, path in sorted(sample_masks.items())
            }
            for sample_id, sample_masks in sorted(mask_paths.items())
        },
        "discovery_summary": {
            "image_count": discovery_summary.image_count,
            "mask_file_count": discovery_summary.mask_file_count,
            "mask_id_count": discovery_summary.mask_id_count,
            "non_png_files_skipped": discovery_summary.non_png_files_skipped,
            "malformed_mask_names": discovery_summary.malformed_mask_names,
            "duplicate_masks": discovery_summary.duplicate_masks,
            "unknown_classes": discovery_summary.unknown_classes,
            "class_counts": discovery_summary.class_counts,
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temp_path.replace(cache_path)


def _try_load_existing_sample_records(
    *,
    dataset_root: Path,
    output_root: Path,
    output_paths: dict[str, Path],
    sample: SourceSample,
    write_rgba: bool,
    write_layered_export: bool,
    include_background_layer: bool,
    layered_scheme: str,
    layered_prompt: str,
    require_preview: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None, str] | None:
    sample_id_str = sample.sample_id_str
    image_output_path = output_paths["images"] / f"{sample_id_str}.jpg"
    label_output_path = output_paths["labels_7class"] / f"{sample_id_str}.png"
    masks_output_path = output_paths["masks_7slot"] / f"{sample_id_str}.npz"
    preview_output_path = output_paths["previews"] / f"{sample_id_str}.png"
    rgba_output_path = output_paths.get("rgba_layers_7slot", output_root / "__missing__") / f"{sample_id_str}.npz"

    required_paths = [image_output_path, label_output_path, masks_output_path]
    if require_preview:
        required_paths.append(preview_output_path)
    if write_rgba:
        required_paths.append(rgba_output_path)
    if any(not path.exists() for path in required_paths):
        return None

    cached_records = _try_load_sample_record_cache(
        output_root=output_root,
        output_paths=output_paths,
        sample=sample,
        write_rgba=write_rgba,
        write_layered_export=write_layered_export,
        include_background_layer=include_background_layer,
        layered_scheme=layered_scheme,
        layered_prompt=layered_prompt,
        require_preview=require_preview,
    )
    if cached_records is not None:
        return cached_records

    slot_masks = _load_saved_slot_masks(masks_output_path)
    slot_summary = _summarize_slot_masks(slot_masks)
    layered_record = None
    if write_layered_export:
        layered_record = _build_layered_record_from_slot_masks(
            output_root=output_root,
            output_paths=output_paths,
            sample=sample,
            slot_masks=slot_masks,
            include_background_layer=include_background_layer,
            layered_scheme=layered_scheme,
            layered_prompt=layered_prompt,
        )
        composite_output_path = output_root / layered_record["composite_path"]
        if not composite_output_path.is_file():
            return None
        expected_layer_paths = [output_root / relative_path for relative_path in layered_record["layer_paths"]]
        if any(not path.is_file() for path in expected_layer_paths):
            return None
        sample_layer_dir = output_paths["layered_layers"] / sample_id_str
        existing_layer_names = sorted(path.name for path in sample_layer_dir.glob("*.png"))
        expected_layer_names = sorted(path.name for path in expected_layer_paths)
        if existing_layer_names != expected_layer_names:
            return None

    rgba_relative_path = None
    if write_rgba:
        rgba_relative_path = str(rgba_output_path.relative_to(output_root))

    image_mode_used = _classify_existing_image_artifact(sample.image_path, image_output_path)
    sample_record = _build_sample_record(
        dataset_root=dataset_root,
        output_root=output_root,
        output_paths=output_paths,
        sample=sample,
        image_size=_load_image_size(sample.image_path),
        slot_presence=slot_summary["slot_presence"],
        slot_pixel_counts=slot_summary["slot_pixel_counts"],
        overlap_pixels=slot_summary["overlap_pixels"],
        pair_overlap_pixels=slot_summary["pair_overlap_pixels"],
        warnings=slot_summary["warnings"],
        rgba_relative_path=rgba_relative_path,
        layered_record=layered_record,
    )
    _write_sample_record_cache(
        output_paths=output_paths,
        sample=sample,
        sample_record=sample_record,
        layered_record=layered_record,
        image_mode_used=image_mode_used,
        write_rgba=write_rgba,
        write_layered_export=write_layered_export,
        include_background_layer=include_background_layer,
        layered_scheme=layered_scheme,
        layered_prompt=layered_prompt,
    )
    return sample_record, layered_record, image_mode_used


def _build_sample_record(
    *,
    dataset_root: Path,
    output_root: Path,
    output_paths: dict[str, Path],
    sample: SourceSample,
    image_size: tuple[int, int],
    slot_presence: list[int],
    slot_pixel_counts: list[int],
    overlap_pixels: int,
    pair_overlap_pixels: dict[str, int],
    warnings: list[str],
    rgba_relative_path: str | None,
    layered_record: dict[str, Any] | None,
) -> dict[str, Any]:
    sample_id_str = sample.sample_id_str
    processed_image_path = output_paths["images"] / f"{sample_id_str}.jpg"
    label_output_path = output_paths["labels_7class"] / f"{sample_id_str}.png"
    masks_output_path = output_paths["masks_7slot"] / f"{sample_id_str}.npz"
    return {
        "sample_id": sample.sample_id,
        "sample_id_str": sample_id_str,
        "split": sample.split,
        "image_path": str(sample.image_path.relative_to(dataset_root)),
        "processed_image_path": str(processed_image_path.relative_to(output_root)),
        "label_path": str(label_output_path.relative_to(output_root)),
        "masks_path": str(masks_output_path.relative_to(output_root)),
        "rgba_path": rgba_relative_path,
        "source_classes_found": sorted(sample.mask_paths),
        "slot_presence": slot_presence,
        "slot_pixel_counts": slot_pixel_counts,
        "image_size": list(image_size),
        "mask_size": list(EXPECTED_MASK_SIZE),
        "overlap_pixels": overlap_pixels,
        "pair_overlap_pixels": pair_overlap_pixels,
        "warnings": warnings,
        "layered_sample_available": layered_record is not None,
        "layered_layer_count": layered_record["layer_count"] if layered_record is not None else None,
    }


def _build_layered_record_from_slot_masks(
    *,
    output_root: Path,
    output_paths: dict[str, Path],
    sample: SourceSample,
    slot_masks: np.ndarray,
    include_background_layer: bool,
    layered_scheme: str,
    layered_prompt: str,
) -> dict[str, Any]:
    sample_id_str = sample.sample_id_str
    composite_output_path = output_paths["layered_composites"] / f"{sample_id_str}.png"
    layer_specs = _build_layered_layer_specs(slot_masks, include_background_layer, layered_scheme)
    if not layer_specs:
        raise ValueError(f"Layered export produced no layers for sample {sample.sample_id}")

    layer_paths = [
        str((output_paths["layered_layers"] / sample_id_str / layer_spec["filename"]).relative_to(output_root))
        for layer_spec in layer_specs
    ]
    return {
        "sample_id": sample.sample_id,
        "sample_id_str": sample_id_str,
        "split": sample.split,
        "prompt": layered_prompt,
        "composite_path": str(composite_output_path.relative_to(output_root)),
        "layer_paths": layer_paths,
        "layer_names": [layer_spec["layer_name"] for layer_spec in layer_specs],
        "layer_source_slots": [layer_spec["source_slots"] for layer_spec in layer_specs],
        "layered_scheme": layered_scheme,
        "layer_pixel_counts": [layer_spec["pixel_count"] for layer_spec in layer_specs],
        "layer_count": len(layer_paths),
        "canvas_size": list(EXPECTED_MASK_SIZE),
        "warnings": _summarize_slot_masks(slot_masks)["warnings"],
        "source_classes_found": sorted(sample.mask_paths),
    }


def _build_layered_layer_specs(
    slot_masks: np.ndarray,
    include_background_layer: bool,
    layered_scheme: str,
) -> list[dict[str, Any]]:
    layer_specs: list[dict[str, Any]] = []
    for mask_def in _build_layered_mask_defs(
        slot_masks=slot_masks,
        include_background_layer=include_background_layer,
        layered_scheme=layered_scheme,
    ):
        layer_specs.append({
            "layer_name": mask_def["layer_name"],
            "source_slots": mask_def["source_slots"],
            "pixel_count": mask_def["pixel_count"],
            "filename": f"{len(layer_specs):02d}_{_slugify(mask_def['layer_name'])}.png",
        })
    return layer_specs


def _load_saved_slot_masks(masks_output_path: Path) -> np.ndarray:
    with np.load(masks_output_path, allow_pickle=False) as payload:
        return payload["masks"].astype(bool)


def _load_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def _classify_existing_image_artifact(source_path: Path, target_path: Path) -> str:
    try:
        source_stat = source_path.stat()
        target_stat = target_path.stat()
    except OSError:
        return "copy"
    if (
        source_stat.st_ino == target_stat.st_ino
        and source_stat.st_dev == target_stat.st_dev
        and target_stat.st_nlink > 1
    ):
        return "hardlink"
    return "copy"


def _try_load_sample_record_cache(
    *,
    output_root: Path,
    output_paths: dict[str, Path],
    sample: SourceSample,
    write_rgba: bool,
    write_layered_export: bool,
    include_background_layer: bool,
    layered_scheme: str,
    layered_prompt: str,
    require_preview: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None, str] | None:
    cache_path = output_paths["sample_record_cache"] / f"{sample.sample_id_str}.json"
    if not cache_path.is_file():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    expected_config = {
        "write_rgba": write_rgba,
        "write_layered_export": write_layered_export,
        "include_background_layer": include_background_layer,
        "layered_scheme": layered_scheme,
        "layered_prompt": layered_prompt,
    }
    if payload.get("version") != 1 or payload.get("config") != expected_config:
        return None
    if int(payload.get("sample_id", -1)) != sample.sample_id:
        return None

    sample_record = payload.get("sample_record")
    layered_record = payload.get("layered_record")
    image_mode_used = payload.get("image_mode_used")
    if not isinstance(sample_record, dict) or image_mode_used not in {"hardlink", "copy"}:
        return None
    if write_layered_export and not isinstance(layered_record, dict):
        return None
    if not write_layered_export:
        layered_record = None
    if write_layered_export:
        expected_layer_names = _layered_scheme_target_names(layered_scheme, include_background_layer)
        if layered_record.get("layer_names") != expected_layer_names:
            return None

    required_paths = [
        output_root / sample_record["processed_image_path"],
        output_root / sample_record["label_path"],
        output_root / sample_record["masks_path"],
    ]
    if require_preview:
        required_paths.append(output_paths["previews"] / f"{sample.sample_id_str}.png")
    if write_rgba and sample_record.get("rgba_path"):
        required_paths.append(output_root / sample_record["rgba_path"])
    if write_layered_export:
        required_paths.append(output_root / layered_record["composite_path"])
        required_paths.extend(output_root / layer_path for layer_path in layered_record["layer_paths"])
    if any(not path.is_file() for path in required_paths):
        return None

    return sample_record, layered_record, image_mode_used


def _write_sample_record_cache(
    *,
    output_paths: dict[str, Path],
    sample: SourceSample,
    sample_record: dict[str, Any],
    layered_record: dict[str, Any] | None,
    image_mode_used: str,
    write_rgba: bool,
    write_layered_export: bool,
    include_background_layer: bool,
    layered_scheme: str,
    layered_prompt: str,
) -> None:
    cache_path = output_paths["sample_record_cache"] / f"{sample.sample_id_str}.json"
    payload = {
        "version": 1,
        "sample_id": sample.sample_id,
        "sample_id_str": sample.sample_id_str,
        "image_mode_used": image_mode_used,
        "config": {
            "write_rgba": write_rgba,
            "write_layered_export": write_layered_export,
            "include_background_layer": include_background_layer,
            "layered_scheme": layered_scheme,
            "layered_prompt": layered_prompt,
        },
        "sample_record": sample_record,
        "layered_record": layered_record,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(cache_path)


def _write_layered_sample(
    *,
    output_root: Path,
    output_paths: dict[str, Path],
    sample: SourceSample,
    processed: dict[str, Any],
    include_background_layer: bool,
    layered_scheme: str,
    layered_prompt: str,
) -> dict[str, Any]:
    sample_id_str = f"{sample.sample_id:05d}"
    composite_output_path = output_paths["layered_composites"] / f"{sample_id_str}.png"
    Image.fromarray(processed["preview_rgb"], mode="RGB").save(composite_output_path)

    sample_layer_dir = output_paths["layered_layers"] / sample_id_str
    sample_layer_dir.mkdir(parents=True, exist_ok=True)
    for existing_path in sample_layer_dir.glob("*.png"):
        existing_path.unlink()

    layer_defs = _build_layered_layers(
        image_rgb=processed["preview_rgb"],
        slot_masks=processed["slot_masks"],
        include_background_layer=include_background_layer,
        layered_scheme=layered_scheme,
    )
    if not layer_defs:
        raise ValueError(f"Layered export produced no layers for sample {sample.sample_id}")

    layer_paths: list[str] = []
    layer_names: list[str] = []
    layer_pixel_counts: list[int] = []
    for layer_index, layer_def in enumerate(layer_defs):
        layer_filename = f"{layer_index:02d}_{_slugify(layer_def['layer_name'])}.png"
        layer_output_path = sample_layer_dir / layer_filename
        Image.fromarray(layer_def["rgba"], mode="RGBA").save(layer_output_path)
        layer_paths.append(str(layer_output_path.relative_to(output_root)))
        layer_names.append(layer_def["layer_name"])
        layer_pixel_counts.append(layer_def["pixel_count"])

    return {
        "sample_id": sample.sample_id,
        "sample_id_str": sample_id_str,
        "split": sample.split,
        "prompt": layered_prompt,
        "composite_path": str(composite_output_path.relative_to(output_root)),
        "layer_paths": layer_paths,
        "layer_names": layer_names,
        "layer_source_slots": [layer_def["source_slots"] for layer_def in layer_defs],
        "layered_scheme": layered_scheme,
        "layer_pixel_counts": layer_pixel_counts,
        "layer_count": len(layer_paths),
        "canvas_size": list(EXPECTED_MASK_SIZE),
        "warnings": processed["warnings"],
        "source_classes_found": processed["source_classes_found"],
    }


def _process_sample(sample: SourceSample, write_rgba: bool) -> dict[str, Any]:
    image_size, preview_rgb = _load_preview_rgb(sample.image_path)
    mask_shape = (EXPECTED_MASK_SIZE[1], EXPECTED_MASK_SIZE[0])
    slot_masks = np.zeros((len(SLOT_NAMES), mask_shape[0], mask_shape[1]), dtype=bool)

    for slot_index, slot_name in enumerate(SLOT_NAMES[1:], start=1):
        merged_mask = np.zeros(mask_shape, dtype=bool)
        for class_name in SLOT_TO_SOURCE_CLASSES[slot_name]:
            mask_path = sample.mask_paths.get(class_name)
            if mask_path is None:
                continue
            merged_mask |= _load_binary_mask(mask_path)
        slot_masks[slot_index] = merged_mask

    foreground_stack = slot_masks[1:]
    slot_masks[0] = ~np.any(foreground_stack, axis=0)

    label_map = np.zeros(mask_shape, dtype=np.uint8)
    for slot_index in range(1, len(SLOT_NAMES)):
        label_map[slot_masks[slot_index]] = slot_index
    slot_summary = _summarize_slot_masks(slot_masks)

    return {
        "image_size": image_size,
        "mask_size": EXPECTED_MASK_SIZE,
        "preview_rgb": preview_rgb,
        "label_map": label_map,
        "slot_masks": slot_masks,
        "slot_presence": slot_summary["slot_presence"],
        "slot_pixel_counts": slot_summary["slot_pixel_counts"],
        "source_classes_found": sorted(sample.mask_paths),
        "overlap_pixels": slot_summary["overlap_pixels"],
        "pair_overlap_pixels": slot_summary["pair_overlap_pixels"],
        "warnings": slot_summary["warnings"],
        "rgba_layers": _build_rgba_layers(preview_rgb, slot_masks) if write_rgba else None,
    }


def _load_preview_rgb(image_path: Path) -> tuple[tuple[int, int], np.ndarray]:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_size = image.size
        preview_image = image.resize(EXPECTED_MASK_SIZE, resample=Image.Resampling.BILINEAR)
        preview_rgb = np.asarray(preview_image, dtype=np.uint8)
    return image_size, preview_rgb


def _load_binary_mask(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as image:
        image = image.convert("L")
        if image.size != EXPECTED_MASK_SIZE:
            raise ValueError(f"Unexpected mask size {image.size} for {mask_path}")
        mask = np.asarray(image, dtype=np.uint8)

    unique_values = set(np.unique(mask).tolist())
    if not unique_values.issubset({0, 255}):
        raise ValueError(f"Mask {mask_path} is not binary 0/255. Unique values: {sorted(unique_values)}")
    return mask > 0


def _summarize_slot_masks(slot_masks: np.ndarray) -> dict[str, Any]:
    foreground_stack = slot_masks[1:]
    overlap_pixels = int(np.count_nonzero(np.count_nonzero(foreground_stack, axis=0) > 1))
    slot_presence = [int(mask.any()) for mask in slot_masks]
    slot_pixel_counts = [int(mask.sum()) for mask in slot_masks]
    warnings: list[str] = []
    if slot_pixel_counts[0] == 0:
        warnings.append("empty_background")
    if not any(slot_presence[1:]):
        warnings.append("empty_foreground")
    return {
        "slot_presence": slot_presence,
        "slot_pixel_counts": slot_pixel_counts,
        "overlap_pixels": overlap_pixels,
        "pair_overlap_pixels": _compute_pair_overlaps(slot_masks),
        "warnings": warnings,
    }


def _build_rgba_layers(image_rgb: np.ndarray, slot_masks: np.ndarray) -> np.ndarray:
    rgba_layers = np.zeros((len(SLOT_NAMES), 4, image_rgb.shape[0], image_rgb.shape[1]), dtype=np.uint8)
    rgb_chw = np.transpose(image_rgb, (2, 0, 1))
    for slot_index in range(len(SLOT_NAMES)):
        rgba_layers[slot_index, :3] = rgb_chw
        rgba_layers[slot_index, 3] = slot_masks[slot_index].astype(np.uint8) * 255
    return rgba_layers


def _build_layered_layers(
    image_rgb: np.ndarray,
    slot_masks: np.ndarray,
    include_background_layer: bool,
    layered_scheme: str,
) -> list[dict[str, Any]]:
    layer_defs: list[dict[str, Any]] = []
    for mask_def in _build_layered_mask_defs(
        slot_masks=slot_masks,
        include_background_layer=include_background_layer,
        layered_scheme=layered_scheme,
    ):
        mask = mask_def["mask"]
        rgba = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 4), dtype=np.uint8)
        rgba[..., :3] = image_rgb * mask[..., None].astype(np.uint8)
        rgba[..., 3] = mask.astype(np.uint8) * 255
        layer_defs.append({
            "layer_name": mask_def["layer_name"],
            "source_slots": mask_def["source_slots"],
            "pixel_count": mask_def["pixel_count"],
            "rgba": rgba,
        })
    return layer_defs


def _build_layered_mask_defs(
    *,
    slot_masks: np.ndarray,
    include_background_layer: bool,
    layered_scheme: str,
) -> list[dict[str, Any]]:
    layered_scheme = _normalize_layered_scheme(layered_scheme)
    layer_defs: list[dict[str, Any]] = []
    slot_index_by_name = {slot_name: slot_index for slot_index, slot_name in enumerate(SLOT_NAMES)}
    grouped_non_background_mask = np.zeros(slot_masks.shape[1:], dtype=bool)

    for layer_name, source_slots in LAYERED_SCHEMES[layered_scheme]:
        if layer_name == "BG" and not include_background_layer:
            continue
        if source_slots == "__REMAINDER__":
            continue
        for slot_name in source_slots:
            grouped_non_background_mask |= slot_masks[slot_index_by_name[slot_name]]

    for layer_name, source_slots in LAYERED_SCHEMES[layered_scheme]:
        if layer_name == "BG" and not include_background_layer:
            continue
        if source_slots == "__REMAINDER__":
            mask = ~grouped_non_background_mask
            source_slot_names = ["__REMAINDER__"]
        else:
            mask = np.zeros(slot_masks.shape[1:], dtype=bool)
            for slot_name in source_slots:
                mask |= slot_masks[slot_index_by_name[slot_name]]
            source_slot_names = list(source_slots)
        pixel_count = int(mask.sum())
        if pixel_count == 0 and not _layered_scheme_keeps_empty_layers(layered_scheme):
            continue
        layer_defs.append({
            "layer_name": layer_name,
            "source_slots": source_slot_names,
            "pixel_count": pixel_count,
            "mask": mask,
        })
    return layer_defs


def _layered_scheme_keeps_empty_layers(layered_scheme: str) -> bool:
    return _normalize_layered_scheme(layered_scheme) == "bg_hair_face"


def _normalize_layered_scheme(layered_scheme: str) -> str:
    normalized = layered_scheme.strip().lower().replace("-", "_")
    aliases = {
        "canonical": "canonical_slots",
        "present_slots": "canonical_slots",
        "bg_hair_face_3layer": "bg_hair_face",
        "bghairface": "bg_hair_face",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in LAYERED_SCHEMES:
        valid_schemes = ", ".join(sorted(LAYERED_SCHEMES))
        raise ValueError(f"Unknown layered scheme '{layered_scheme}'. Valid schemes: {valid_schemes}")
    return normalized


def _layered_scheme_target_names(layered_scheme: str, include_background_layer: bool) -> list[str]:
    names: list[str] = []
    for layer_name, _source_slots in LAYERED_SCHEMES[_normalize_layered_scheme(layered_scheme)]:
        if layer_name == "BG" and not include_background_layer:
            continue
        names.append(layer_name)
    return names


def _layered_scheme_metadata(layered_scheme: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for layer_name, source_slots in LAYERED_SCHEMES[_normalize_layered_scheme(layered_scheme)]:
        if source_slots == "__REMAINDER__":
            rows.append({
                "layer_name": layer_name,
                "source_slots": ["__REMAINDER__"],
                "definition": "Complement of all non-background groups in this layered export scheme.",
            })
        else:
            rows.append({
                "layer_name": layer_name,
                "source_slots": list(source_slots),
            })
    return rows


def _pair_overlap_keys() -> list[str]:
    keys: list[str] = []
    for left_index in range(1, len(SLOT_NAMES)):
        for right_index in range(left_index + 1, len(SLOT_NAMES)):
            keys.append(f"{SLOT_NAMES[left_index]}__{SLOT_NAMES[right_index]}")
    return keys


def _compute_pair_overlaps(slot_masks: np.ndarray) -> dict[str, int]:
    overlaps: dict[str, int] = {}
    for left_index in range(1, len(SLOT_NAMES)):
        for right_index in range(left_index + 1, len(SLOT_NAMES)):
            pair_key = f"{SLOT_NAMES[left_index]}__{SLOT_NAMES[right_index]}"
            overlaps[pair_key] = int(np.logical_and(slot_masks[left_index], slot_masks[right_index]).sum())
    return overlaps


def _link_or_copy_image(source_path: Path, target_path: Path, prefer_copy: bool) -> str:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        target_path.unlink()

    if prefer_copy:
        shutil.copy2(source_path, target_path)
        return "copy"

    try:
        os.link(source_path, target_path)
        return "hardlink"
    except OSError:
        shutil.copy2(source_path, target_path)
        return "copy"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def _count_by_key(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record[key])
        counts[value] = counts.get(value, 0) + 1
    return counts


def _build_processed_splits(sample_records: list[dict[str, Any]]) -> dict[str, list[int]]:
    processed_splits = {"train": [], "val": [], "test": []}
    for record in sample_records:
        processed_splits.setdefault(record["split"], []).append(int(record["sample_id"]))
    for split_ids in processed_splits.values():
        split_ids.sort()
    return processed_splits


def _slugify(value: str) -> str:
    return value.lower().replace(" ", "_")
