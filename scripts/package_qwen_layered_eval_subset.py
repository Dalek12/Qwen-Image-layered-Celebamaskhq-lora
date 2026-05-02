"""Create a small evaluation package from a held-out split for Qwen layered inference checks."""

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
    parser.add_argument("--package-name", default="processed_celebmaskhq_test_eval")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--sample-strategy", choices=["spaced", "first"], default="spaced")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_root = args.processed_root.resolve()
    output_root = args.output_root.resolve()
    package_name = args.package_name.strip()
    target_split = args.split.strip()

    if not package_name:
        raise ValueError("--package-name must be non-empty.")
    if not target_split:
        raise ValueError("--split must be non-empty.")
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive.")

    metadata_dir = processed_root / "metadata"
    layered_samples_path = metadata_dir / "layered_samples.jsonl"
    if not layered_samples_path.is_file():
        raise FileNotFoundError(f"Missing layered metadata: {layered_samples_path}")

    if output_root.exists():
        if args.force:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(f"Output root already exists: {output_root}. Use --force to rebuild it.")
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Processed root: {processed_root}", flush=True)
    print(f"Eval output root: {output_root}", flush=True)
    print(f"Package name: {package_name}", flush=True)
    print(f"Target split: {target_split}", flush=True)
    print(f"Max samples: {args.max_samples}", flush=True)
    print(f"Sample strategy: {args.sample_strategy}", flush=True)

    all_rows = read_jsonl(layered_samples_path)
    split_rows = sorted(
        [row for row in all_rows if row.get("split") == target_split],
        key=lambda row: int(row["sample_id"]),
    )
    if not split_rows:
        raise ValueError(f"No rows found for split={target_split} in {layered_samples_path}")

    selected_rows = select_rows(split_rows, max_samples=args.max_samples, strategy=args.sample_strategy)
    verify_packaged_files_exist(processed_root, selected_rows, progress_every=max(args.progress_every, 1))

    metadata_tar_path = output_root / "metadata.tar"
    data_tar_path = output_root / "data.tar"

    package_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_name": package_name,
        "source_processed_root": str(processed_root),
        "target_split": target_split,
        "sample_count": len(selected_rows),
        "all_split_sample_count": len(split_rows),
        "sample_strategy": args.sample_strategy,
        "selected_sample_ids": [int(row["sample_id"]) for row in selected_rows],
        "metadata_tar": metadata_tar_path.name,
        "data_tar": data_tar_path.name,
        "required_paths": [
            "metadata/layered_samples.jsonl",
            "layered_composites/",
            "layered_layers/",
        ],
    }
    write_json(output_root / "package_manifest.json", package_manifest)

    metadata_payloads = {
        f"{package_name}/metadata/layered_samples.jsonl": encode_jsonl(selected_rows),
        f"{package_name}/metadata/package_manifest.json": encode_json(package_manifest),
    }
    print(f"Writing metadata tar: {metadata_tar_path}", flush=True)
    with tarfile.open(metadata_tar_path, mode="w") as archive:
        for arcname, payload in metadata_payloads.items():
            add_bytes_to_tar(archive, arcname, payload)

    print(f"Writing data tar: {data_tar_path}", flush=True)
    with tarfile.open(data_tar_path, mode="w") as archive:
        for index, row in enumerate(selected_rows, start=1):
            composite_path = processed_root / row["composite_path"]
            archive.add(composite_path, arcname=f"{package_name}/{row['composite_path']}")
            for layer_path_str in row["layer_paths"]:
                layer_path = processed_root / layer_path_str
                archive.add(layer_path, arcname=f"{package_name}/{layer_path_str}")
            if args.progress_every > 0 and (index == 1 or index % args.progress_every == 0 or index == len(selected_rows)):
                print(f"[pack {index}/{len(selected_rows)}] sample {int(row['sample_id']):05d}", flush=True)

    report = {
        "output_root": str(output_root),
        "sample_count": len(selected_rows),
        "target_split": target_split,
        "selected_sample_ids": package_manifest["selected_sample_ids"],
        "metadata_tar": metadata_tar_path.name,
        "data_tar": data_tar_path.name,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def select_rows(rows: list[dict[str, Any]], *, max_samples: int, strategy: str) -> list[dict[str, Any]]:
    if len(rows) <= max_samples:
        return rows
    if strategy == "first":
        return rows[:max_samples]
    if max_samples == 1:
        return [rows[len(rows) // 2]]

    selected_indices: list[int] = []
    last_index = len(rows) - 1
    for i in range(max_samples):
        selected_indices.append(round(i * last_index / (max_samples - 1)))
    deduped_indices = sorted(dict.fromkeys(selected_indices))
    next_candidate = 0
    while len(deduped_indices) < max_samples:
        if next_candidate not in deduped_indices:
            deduped_indices.append(next_candidate)
        next_candidate += 1
    deduped_indices.sort()
    return [rows[index] for index in deduped_indices[:max_samples]]


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def encode_json(payload: Any) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def encode_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")


def add_bytes_to_tar(archive: tarfile.TarFile, arcname: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


if __name__ == "__main__":
    main()
