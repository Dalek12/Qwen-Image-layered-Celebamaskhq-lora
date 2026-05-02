"""Create sharded train/val-only archives for Qwen-Image-Layered LoRA training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--package-name", default="processed_celebmaskhq_trainval")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_root = args.processed_root.resolve()
    output_root = args.output_root.resolve()
    package_name = args.package_name.strip()
    included_splits = tuple(dict.fromkeys(args.splits))

    if not package_name:
        raise ValueError("--package-name must be non-empty.")
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be a positive integer.")
    if not included_splits:
        raise ValueError("At least one split must be provided via --splits.")

    metadata_dir = processed_root / "metadata"
    layered_samples_path = metadata_dir / "layered_samples.jsonl"
    samples_path = metadata_dir / "samples.jsonl"
    if not layered_samples_path.is_file():
        raise FileNotFoundError(f"Missing layered metadata: {layered_samples_path}")

    if output_root.exists():
        if args.force:
            shutil.rmtree(output_root)
        elif not args.resume_existing:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Use --force to rebuild it or --resume-existing to reuse completed shards."
            )
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Processed root: {processed_root}", flush=True)
    print(f"Shard output root: {output_root}", flush=True)
    print(f"Included splits: {included_splits}", flush=True)
    print(f"Package name: {package_name}", flush=True)
    print(f"Shard size: {args.shard_size}", flush=True)
    print(f"Resume existing shards: {args.resume_existing}", flush=True)

    all_layered_rows = read_jsonl(layered_samples_path)
    filtered_layered_rows = sorted(
        [row for row in all_layered_rows if row.get("split") in included_splits],
        key=lambda row: int(row["sample_id"]),
    )
    if not filtered_layered_rows:
        raise ValueError(f"No layered rows found for splits {included_splits} in {layered_samples_path}")
    if any(row.get("split") not in included_splits for row in filtered_layered_rows):
        raise ValueError("Filtered layered rows still contain excluded splits.")

    filtered_sample_rows: list[dict[str, Any]] = []
    if samples_path.is_file():
        allowed_ids = {int(row["sample_id"]) for row in filtered_layered_rows}
        filtered_sample_rows = sorted(
            [
                row
                for row in read_jsonl(samples_path)
                if int(row["sample_id"]) in allowed_ids and row.get("split") in included_splits
            ],
            key=lambda row: int(row["sample_id"]),
        )

    excluded_split_counts = count_by_key(
        [row for row in all_layered_rows if row.get("split") not in included_splits],
        "split",
    )
    split_counts = count_by_key(filtered_layered_rows, "split")

    verify_packaged_files_exist(processed_root, filtered_layered_rows, progress_every=args.progress_every)

    filtered_splits = {
        split_name: [int(row["sample_id"]) for row in filtered_layered_rows if row["split"] == split_name]
        for split_name in included_splits
    }

    metadata_tar_path = output_root / "metadata.tar"
    shard_descriptors = build_data_shards(
        processed_root=processed_root,
        output_root=output_root,
        package_name=package_name,
        rows=filtered_layered_rows,
        shard_size=args.shard_size,
        progress_every=max(args.progress_every, 1),
        resume_existing=args.resume_existing,
    )

    package_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_name": package_name,
        "source_processed_root": str(processed_root),
        "included_splits": list(included_splits),
        "excluded_split_counts": excluded_split_counts,
        "sample_count": len(filtered_layered_rows),
        "split_counts": split_counts,
        "metadata_tar": metadata_tar_path.name,
        "data_shards": shard_descriptors,
        "shard_size": args.shard_size,
        "training_contract": {
            "required_paths": [
                "metadata/layered_samples.jsonl",
                "layered_composites/",
                "layered_layers/",
            ],
            "excluded_paths": [
                "images/",
                "labels_7class/",
                "masks_7slot/",
                "previews/",
            ],
        },
        "test_split_packaged": split_counts.get("test", 0),
    }
    if package_manifest["test_split_packaged"] != 0:
        raise ValueError(f"Test split leakage detected in package manifest: {package_manifest['test_split_packaged']}")

    write_json(output_root / "package_manifest.json", package_manifest)

    metadata_payloads = {
        f"{package_name}/metadata/layered_samples.jsonl": encode_jsonl(filtered_layered_rows),
        f"{package_name}/metadata/package_manifest.json": encode_json(package_manifest),
        f"{package_name}/metadata/splits.json": encode_json(filtered_splits),
    }
    if filtered_sample_rows:
        metadata_payloads[f"{package_name}/metadata/samples.jsonl"] = encode_jsonl(filtered_sample_rows)

    print(f"Writing metadata tar: {metadata_tar_path}", flush=True)
    with tarfile.open(metadata_tar_path, mode="w") as archive:
        for arcname, payload in metadata_payloads.items():
            add_bytes_to_tar(archive, arcname, payload)

    report = {
        "output_root": str(output_root),
        "metadata_tar": metadata_tar_path.name,
        "data_shard_count": len(shard_descriptors),
        "sample_count": len(filtered_layered_rows),
        "split_counts": split_counts,
        "excluded_split_counts": excluded_split_counts,
        "test_split_packaged": package_manifest["test_split_packaged"],
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def verify_packaged_files_exist(processed_root: Path, rows: list[dict[str, Any]], progress_every: int) -> None:
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        composite_path = processed_root / row["composite_path"]
        if not composite_path.is_file():
            raise FileNotFoundError(f"Missing composite for sample {row['sample_id']}: {composite_path}")
        for layer_path_str in row["layer_paths"]:
            layer_path = processed_root / layer_path_str
            if not layer_path.is_file():
                raise FileNotFoundError(f"Missing layer file for sample {row['sample_id']}: {layer_path}")
        if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == total):
            print(f"[verify {index}/{total}] sample {int(row['sample_id']):05d}", flush=True)


def build_data_shards(
    *,
    processed_root: Path,
    output_root: Path,
    package_name: str,
    rows: list[dict[str, Any]],
    shard_size: int,
    progress_every: int,
    resume_existing: bool,
) -> list[dict[str, Any]]:
    shard_descriptors: list[dict[str, Any]] = []
    total_rows = len(rows)
    for shard_index, start in enumerate(range(0, total_rows, shard_size)):
        chunk = rows[start : start + shard_size]
        shard_filename = f"data_shard_{shard_index:04d}.tar"
        shard_path = output_root / shard_filename
        split_counts = count_by_key(chunk, "split")
        sample_ids = [int(row["sample_id"]) for row in chunk]

        print(
            f"Writing shard {shard_index + 1}/{((total_rows - 1) // shard_size) + 1}: "
            f"{shard_filename} | samples={len(chunk)} | ids={sample_ids[0]:05d}-{sample_ids[-1]:05d}",
            flush=True,
        )
        if resume_existing and shard_path.is_file():
            print(f"  Reusing existing shard: {shard_path}", flush=True)
        else:
            if shard_path.exists():
                shard_path.unlink()
            with tarfile.open(shard_path, mode="w") as archive:
                for index, row in enumerate(chunk, start=1):
                    composite_path = processed_root / row["composite_path"]
                    archive.add(composite_path, arcname=f"{package_name}/{row['composite_path']}")
                    for layer_path_str in row["layer_paths"]:
                        layer_path = processed_root / layer_path_str
                        archive.add(layer_path, arcname=f"{package_name}/{layer_path_str}")
                    if progress_every > 0 and (index == 1 or index % progress_every == 0 or index == len(chunk)):
                        print(
                            f"  [shard {shard_index:04d} {index}/{len(chunk)}] sample {int(row['sample_id']):05d}",
                            flush=True,
                        )

        shard_descriptors.append(
            {
                "filename": shard_filename,
                "sample_count": len(chunk),
                "sample_id_start": sample_ids[0],
                "sample_id_end": sample_ids[-1],
                "split_counts": split_counts,
            }
        )
    return shard_descriptors


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def count_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return counts


def encode_json(payload: Any) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def encode_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")


def add_bytes_to_tar(archive: tarfile.TarFile, arcname: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = len(payload)
    info.mtime = int(datetime.now(timezone.utc).timestamp())
    archive.addfile(info, io.BytesIO(payload))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
