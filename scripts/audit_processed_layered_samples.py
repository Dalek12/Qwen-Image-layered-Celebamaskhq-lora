#!/usr/bin/env python
"""Sample processed layered examples into a lightweight audit folder."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import shutil
from typing import Any

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--sample-count", type=int, default=6)
    parser.add_argument("--sample-strategy", default="random", choices=["random", "spaced", "first"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--sample-ids", nargs="*", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def select_rows(
    rows: list[dict[str, Any]],
    *,
    sample_count: int,
    strategy: str,
    seed: int,
    sample_ids: list[int] | None,
) -> list[dict[str, Any]]:
    if sample_ids:
        row_by_id = {int(row["sample_id"]): row for row in rows}
        selected = []
        missing = []
        for sample_id in sample_ids:
            row = row_by_id.get(int(sample_id))
            if row is None:
                missing.append(sample_id)
            else:
                selected.append(row)
        if missing:
            raise ValueError(f"Requested sample ids were not found in the filtered split: {missing}")
        return selected

    if sample_count <= 0:
        raise ValueError("--sample-count must be positive.")
    if len(rows) <= sample_count:
        return rows

    if strategy == "first":
        return rows[:sample_count]
    if strategy == "random":
        generator = random.Random(seed)
        indices = sorted(generator.sample(range(len(rows)), sample_count))
        return [rows[index] for index in indices]

    # spaced
    if sample_count == 1:
        return [rows[len(rows) // 2]]
    last_index = len(rows) - 1
    indices = [round(i * last_index / (sample_count - 1)) for i in range(sample_count)]
    deduped = sorted(dict.fromkeys(indices))
    next_index = 0
    while len(deduped) < sample_count:
        if next_index not in deduped:
            deduped.append(next_index)
        next_index += 1
    deduped.sort()
    return [rows[index] for index in deduped[:sample_count]]


def copy_if_missing(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def build_contact_sheet(sample_dir: Path, composite_path: Path, layer_paths: list[Path]) -> Path:
    tile_size = 256
    tiles = [Image.open(composite_path).convert("RGBA")]
    tiles.extend(Image.open(layer_path).convert("RGBA") for layer_path in layer_paths)
    resized = [image.resize((tile_size, tile_size), resample=Image.Resampling.BILINEAR) for image in tiles]

    cols = 3
    rows = (len(resized) + cols - 1) // cols
    canvas = Image.new("RGBA", (cols * tile_size, rows * tile_size), (255, 255, 255, 255))
    for index, image in enumerate(resized):
        x = (index % cols) * tile_size
        y = (index // cols) * tile_size
        canvas.paste(image, (x, y), image)

    output_path = sample_dir / "contact_sheet.png"
    canvas.save(output_path)

    for image in tiles:
        image.close()
    for image in resized:
        image.close()
    canvas.close()
    return output_path


def main() -> None:
    args = parse_args()
    processed_root = args.processed_root.resolve()
    output_root = args.output_root.resolve()
    metadata_path = processed_root / "metadata" / "layered_samples.jsonl"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing layered metadata: {metadata_path}")

    if output_root.exists():
        if args.force:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(f"Output root already exists: {output_root}. Use --force to rebuild it.")
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows = read_jsonl(metadata_path)
    split_rows = [row for row in all_rows if row.get("split") == args.split]
    if not split_rows:
        raise ValueError(f"No rows found for split={args.split!r} in {metadata_path}")

    selected_rows = select_rows(
        split_rows,
        sample_count=args.sample_count,
        strategy=args.sample_strategy,
        seed=args.seed,
        sample_ids=args.sample_ids,
    )

    audit_records: list[dict[str, Any]] = []
    for row in selected_rows:
        sample_id_str = f"{int(row['sample_id']):05d}"
        sample_dir = output_root / sample_id_str
        sample_dir.mkdir(parents=True, exist_ok=True)

        composite_source = processed_root / row["composite_path"]
        composite_target = sample_dir / "input_composite.png"
        copy_if_missing(composite_source, composite_target)

        copied_layer_paths: list[str] = []
        layer_sources: list[Path] = []
        for index, layer_path_str in enumerate(row["layer_paths"]):
            layer_source = processed_root / layer_path_str
            layer_sources.append(layer_source)
            target_name = f"target_layer_{index:02d}_{Path(layer_path_str).name}"
            layer_target = sample_dir / target_name
            copy_if_missing(layer_source, layer_target)
            copied_layer_paths.append(target_name)

        contact_sheet_path = build_contact_sheet(sample_dir, composite_target, [sample_dir / path for path in copied_layer_paths])

        selected_payload = {
            **row,
            "audit_input_path": composite_target.name,
            "audit_layer_paths": copied_layer_paths,
            "audit_contact_sheet": contact_sheet_path.name,
        }
        write_json(sample_dir / "selected_sample.json", selected_payload)
        audit_records.append(selected_payload)

        print(
            f"Prepared audit sample {sample_id_str}: "
            f"layers={len(copied_layer_paths)} split={row['split']} "
            f"contact_sheet={contact_sheet_path.name}",
            flush=True,
        )

    audit_manifest = {
        "processed_root": str(processed_root),
        "output_root": str(output_root),
        "split": args.split,
        "sample_count": len(audit_records),
        "sample_strategy": args.sample_strategy,
        "seed": args.seed,
        "sample_ids": [int(row["sample_id"]) for row in audit_records],
    }
    write_json(output_root / "audit_manifest.json", audit_manifest)
    write_json(output_root / "audit_records.json", audit_records)

    print(json.dumps(audit_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
