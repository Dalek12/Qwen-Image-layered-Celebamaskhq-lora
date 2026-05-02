"""Recover metadata files from an existing processed CelebAMask-HQ export."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess_celebmaskhq.constants import (
    DEFAULT_SPLIT_RATIOS,
    EXPECTED_IMAGE_SIZE,
    EXPECTED_MASK_SIZE,
    SLOT_NAMES,
    SLOT_TO_SOURCE_CLASSES,
    SOURCE_CLASSES,
)
from preprocess_celebmaskhq.dataset import SourceSample, build_splits, discover_image_paths, discover_mask_paths
from preprocess_celebmaskhq.pipeline import (
    _build_layered_record_from_slot_masks,
    _build_sample_record,
    _classify_existing_image_artifact,
    _load_image_size,
    _load_saved_slot_masks,
    _pair_overlap_keys,
    _summarize_slot_masks,
    _write_json,
    _write_jsonl,
)


DEFAULT_LAYERED_PROMPT = "decompose this portrait into editable portrait layers"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--layered-prompt", type=str, default=DEFAULT_LAYERED_PROMPT)
    parser.add_argument("--exclude-background-layer", action="store_true")
    parser.add_argument("--require-preview-count", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_root = args.processed_root.resolve()
    dataset_root = args.dataset_root.resolve()
    include_background_layer = not args.exclude_background_layer

    print(f"Processed root: {processed_root}", flush=True)
    print(f"Dataset root: {dataset_root}", flush=True)
    print("Phase 1/5: preparing output paths", flush=True)

    output_paths = {
        "images": processed_root / "images",
        "labels_7class": processed_root / "labels_7class",
        "masks_7slot": processed_root / "masks_7slot",
        "metadata": processed_root / "metadata",
        "previews": processed_root / "previews",
        "layered_composites": processed_root / "layered_composites",
        "layered_layers": processed_root / "layered_layers",
    }
    for path in output_paths.values():
        path.mkdir(parents=True, exist_ok=True)

    print("Phase 2/5: discovering source images", flush=True)
    image_paths = discover_image_paths(dataset_root)
    print(f"Discovered {len(image_paths)} source images", flush=True)
    print("Phase 3/5: discovering source masks", flush=True)
    mask_paths, discovery_summary = discover_mask_paths(dataset_root)
    discovery_summary.image_count = len(image_paths)
    print(
        f"Discovered masks for {len(mask_paths)} sample ids across {discovery_summary.mask_file_count} mask files",
        flush=True,
    )
    all_ids = sorted(set(image_paths) & set(mask_paths))
    missing_images = sorted(set(mask_paths) - set(image_paths))
    ids_without_masks = sorted(set(image_paths) - set(mask_paths))
    if not all_ids:
        raise ValueError("No overlapping sample ids were found between RGB images and masks.")

    print("Phase 4/5: building deterministic splits", flush=True)
    splits, split_lookup = build_splits(all_ids, seed=args.split_seed, ratios=DEFAULT_SPLIT_RATIOS)

    preview_ids = set(all_ids[: max(args.require_preview_count, 0)])
    candidate_ids = sorted(
        {
            int(path.stem)
            for path in output_paths["layered_composites"].glob("*.png")
            if path.stem.isdigit()
        }
    )
    total_candidates = len(candidate_ids)
    print(f"Phase 5/5: recovering metadata from {total_candidates} candidate composites", flush=True)

    sample_records: list[dict[str, Any]] = []
    layered_records: list[dict[str, Any]] = []
    skipped_incomplete_ids: list[int] = []

    slot_presence_totals = [0 for _ in SLOT_NAMES]
    slot_pixel_totals = [0 for _ in SLOT_NAMES]
    pair_overlap_totals = {pair_key: 0 for pair_key in _pair_overlap_keys()}
    layered_count_histogram: dict[str, int] = {}
    total_overlap_pixels = 0
    total_warning_count = 0
    image_write_counts = {"hardlink": 0, "copy": 0}
    actual_image_mode = "copy"
    layered_manifest_path = output_paths["metadata"] / "layered_samples.jsonl"
    samples_manifest_path = output_paths["metadata"] / "samples.jsonl"

    layered_manifest_path.write_text("", encoding="utf-8")
    samples_manifest_path.write_text("", encoding="utf-8")

    print(f"Streaming layered manifest to: {layered_manifest_path}", flush=True)
    print(f"Streaming sample manifest to: {samples_manifest_path}", flush=True)

    with layered_manifest_path.open("a", encoding="utf-8") as layered_handle, samples_manifest_path.open("a", encoding="utf-8") as samples_handle:
        for index, sample_id in enumerate(candidate_ids, start=1):
            if sample_id not in image_paths or sample_id not in mask_paths:
                skipped_incomplete_ids.append(sample_id)
                maybe_print_progress(
                    index=index,
                    total=total_candidates,
                    progress_every=args.progress_every,
                    recovered_count=len(sample_records),
                    skipped_count=len(skipped_incomplete_ids),
                    sample_id=sample_id,
                    action="skipped-missing-source",
                )
                continue

            sample = SourceSample(
                sample_id=sample_id,
                image_path=image_paths[sample_id],
                mask_paths=mask_paths[sample_id],
                split=split_lookup[sample_id],
            )
            sample_id_str = sample.sample_id_str
            image_output_path = output_paths["images"] / f"{sample_id_str}.jpg"
            label_output_path = output_paths["labels_7class"] / f"{sample_id_str}.png"
            masks_output_path = output_paths["masks_7slot"] / f"{sample_id_str}.npz"
            preview_output_path = output_paths["previews"] / f"{sample_id_str}.png"

            required_paths = [image_output_path, label_output_path, masks_output_path]
            if sample_id in preview_ids:
                required_paths.append(preview_output_path)
            if any(not path.exists() for path in required_paths):
                skipped_incomplete_ids.append(sample_id)
                maybe_print_progress(
                    index=index,
                    total=total_candidates,
                    progress_every=args.progress_every,
                    recovered_count=len(sample_records),
                    skipped_count=len(skipped_incomplete_ids),
                    sample_id=sample_id,
                    action="skipped-incomplete-artifacts",
                )
                continue

            slot_masks = _load_saved_slot_masks(masks_output_path)
            slot_summary = _summarize_slot_masks(slot_masks)
            layered_record = _build_layered_record_from_slot_masks(
                output_root=processed_root,
                output_paths=output_paths,
                sample=sample,
                slot_masks=slot_masks,
                include_background_layer=include_background_layer,
                layered_prompt=args.layered_prompt,
            )
            composite_output_path = processed_root / layered_record["composite_path"]
            layer_output_paths = [processed_root / relative_path for relative_path in layered_record["layer_paths"]]
            if not composite_output_path.is_file() or any(not path.is_file() for path in layer_output_paths):
                skipped_incomplete_ids.append(sample_id)
                maybe_print_progress(
                    index=index,
                    total=total_candidates,
                    progress_every=args.progress_every,
                    recovered_count=len(sample_records),
                    skipped_count=len(skipped_incomplete_ids),
                    sample_id=sample_id,
                    action="skipped-incomplete-layered",
                )
                continue

            sample_record = _build_sample_record(
                dataset_root=dataset_root,
                output_root=processed_root,
                output_paths=output_paths,
                sample=sample,
                image_size=_load_image_size(sample.image_path),
                slot_presence=slot_summary["slot_presence"],
                slot_pixel_counts=slot_summary["slot_pixel_counts"],
                overlap_pixels=slot_summary["overlap_pixels"],
                pair_overlap_pixels=slot_summary["pair_overlap_pixels"],
                warnings=slot_summary["warnings"],
                rgba_relative_path=None,
                layered_record=layered_record,
            )

            sample_records.append(sample_record)
            layered_records.append(layered_record)
            append_jsonl_record(layered_handle, layered_record)
            append_jsonl_record(samples_handle, sample_record)

            histogram_key = str(layered_record["layer_count"])
            layered_count_histogram[histogram_key] = layered_count_histogram.get(histogram_key, 0) + 1
            total_overlap_pixels += sample_record["overlap_pixels"]
            total_warning_count += len(sample_record["warnings"])
            for slot_index, present in enumerate(sample_record["slot_presence"]):
                slot_presence_totals[slot_index] += int(present)
            for slot_index, pixel_count in enumerate(sample_record["slot_pixel_counts"]):
                slot_pixel_totals[slot_index] += int(pixel_count)
            for pair_key, pair_count in sample_record["pair_overlap_pixels"].items():
                pair_overlap_totals[pair_key] += int(pair_count)

            image_mode_used = _classify_existing_image_artifact(sample.image_path, image_output_path)
            image_write_counts[image_mode_used] += 1
            if image_mode_used == "hardlink":
                actual_image_mode = "hardlink_with_copy_fallback" if image_write_counts["copy"] > 0 else "hardlink"

            maybe_print_progress(
                index=index,
                total=total_candidates,
                progress_every=args.progress_every,
                recovered_count=len(sample_records),
                skipped_count=len(skipped_incomplete_ids),
                sample_id=sample_id,
                action="recovered",
            )

    mapping_metadata = {
        "slot_order": list(SLOT_NAMES),
        "source_classes": list(SOURCE_CLASSES),
        "slot_to_source_classes": {slot: list(classes) for slot, classes in SLOT_TO_SOURCE_CLASSES.items()},
        "background_definition": "Complement of merged foreground slots.",
        "label_priority": list(SLOT_NAMES),
        "image_resolution": list(EXPECTED_IMAGE_SIZE),
        "mask_resolution": list(EXPECTED_MASK_SIZE),
        "split_seed": args.split_seed,
        "split_ratios": {
            "train": DEFAULT_SPLIT_RATIOS[0],
            "val": DEFAULT_SPLIT_RATIOS[1],
            "test": DEFAULT_SPLIT_RATIOS[2],
        },
        "image_storage_mode": actual_image_mode,
        "write_rgba": False,
        "layered_export": {
            "enabled": True,
            "format": "generic_layered_pngs_v1",
            "composite_dir": "layered_composites",
            "layer_dir": "layered_layers",
            "metadata_path": "metadata/layered_samples.jsonl",
            "include_background_layer": include_background_layer,
            "layer_order_policy": "canonical_present_slots",
            "default_prompt": args.layered_prompt,
        },
        "metadata_recovered": True,
    }

    stats_metadata = {
        "processed_sample_count": len(sample_records),
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
        "split_counts_processed_subset": count_by_key(sample_records, "split"),
        "slot_presence_counts": dict(zip(SLOT_NAMES, slot_presence_totals, strict=True)),
        "slot_pixel_counts": dict(zip(SLOT_NAMES, slot_pixel_totals, strict=True)),
        "total_overlap_pixels": total_overlap_pixels,
        "pair_overlap_pixels": pair_overlap_totals,
        "samples_with_warnings": sum(1 for record in sample_records if record["warnings"]),
        "total_warning_count": total_warning_count,
        "image_write_counts": image_write_counts,
        "processed_sample_ids": [record["sample_id"] for record in sample_records],
        "layered_sample_count": len(layered_records),
        "layered_layer_count_histogram": layered_count_histogram,
        "layered_max_layers": max((record["layer_count"] for record in layered_records), default=0),
        "metadata_recovered": True,
        "skipped_incomplete_sample_count": len(skipped_incomplete_ids),
        "skipped_incomplete_sample_ids_preview": skipped_incomplete_ids[:20],
    }

    _write_json(output_paths["metadata"] / "mapping.json", mapping_metadata)
    _write_json(output_paths["metadata"] / "stats.json", stats_metadata)
    _write_json(output_paths["metadata"] / "splits.json", splits)
    _write_jsonl(output_paths["metadata"] / "samples.jsonl", sample_records)
    _write_jsonl(output_paths["metadata"] / "layered_samples.jsonl", layered_records)

    report = {
        "processed_root": str(processed_root),
        "dataset_root": str(dataset_root),
        "recovered_sample_count": len(sample_records),
        "skipped_incomplete_sample_count": len(skipped_incomplete_ids),
        "skipped_incomplete_sample_ids_preview": skipped_incomplete_ids[:20],
        "metadata_path": str(output_paths["metadata"] / "layered_samples.jsonl"),
        "split_counts_processed_subset": stats_metadata["split_counts_processed_subset"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def count_by_key(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record[key])
        counts[value] = counts.get(value, 0) + 1
    return counts


def append_jsonl_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, sort_keys=True))
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())


def maybe_print_progress(
    *,
    index: int,
    total: int,
    progress_every: int,
    recovered_count: int,
    skipped_count: int,
    sample_id: int,
    action: str,
) -> None:
    if progress_every <= 0:
        return
    if index != 1 and index % progress_every != 0 and index != total:
        return
    print(
        f"[{index}/{total}] {action} sample {sample_id:05d} | "
        f"recovered={recovered_count} skipped={skipped_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
