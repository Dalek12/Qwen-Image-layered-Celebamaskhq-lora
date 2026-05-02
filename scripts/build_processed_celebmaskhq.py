"""CLI entry point for CelebAMask-HQ preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocess_celebmaskhq.pipeline import BuildConfig, DEFAULT_LAYERED_SCHEME, LAYERED_SCHEMES, run_build


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("."),
        help="Root directory containing CelebA-HQ-img and CelebAMask-HQ-mask-anno.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("processed_celebmaskhq"),
        help="Directory where processed outputs will be written.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        type=int,
        default=None,
        help="Optional explicit list of integer sample ids to process.",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help="Optional text file containing one integer sample id per line.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of selected samples to process.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=8,
        help="Number of preview sheets to generate from the processed subset.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=1337,
        help="Deterministic random seed for the 80/10/10 split.",
    )
    parser.add_argument(
        "--write-rgba",
        action="store_true",
        help="Also write compressed RGBA layers with alpha from each fixed slot.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy RGB images into the processed tree instead of trying hardlinks first.",
    )
    parser.add_argument(
        "--no-layered-export",
        action="store_true",
        help="Skip the generic layered export and only write the original fixed-slot artifacts.",
    )
    parser.add_argument(
        "--exclude-background-layer",
        action="store_true",
        help="Do not include background as a target layer in the generic layered export.",
    )
    parser.add_argument(
        "--layered-scheme",
        choices=sorted(LAYERED_SCHEMES),
        default=DEFAULT_LAYERED_SCHEME,
        help=(
            "Layer grouping used for metadata/layered_samples.jsonl. "
            "Use bg_hair_face for the fast 3-layer portrait experiment."
        ),
    )
    parser.add_argument(
        "--layered-prompt",
        type=str,
        default="decompose this portrait into editable portrait layers",
        help="Prompt stored in metadata/layered_samples.jsonl for the generic layered export.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse already-complete sample outputs under --output-root and only process missing sample ids.",
    )
    parser.add_argument(
        "--no-discovery-cache",
        action="store_true",
        help="Always rescan raw image/mask directories instead of using metadata/source_discovery_cache.json.",
    )
    parser.add_argument(
        "--refresh-discovery-cache",
        action="store_true",
        help="Rescan raw image/mask directories and overwrite metadata/source_discovery_cache.json.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print a progress line every N samples while building or resuming.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BuildConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        sample_ids=args.sample_ids,
        ids_file=args.ids_file,
        limit=args.limit,
        preview_count=args.preview_count,
        split_seed=args.split_seed,
        write_rgba=args.write_rgba,
        copy_images=args.copy_images,
        write_layered_export=not args.no_layered_export,
        include_background_layer=not args.exclude_background_layer,
        layered_scheme=args.layered_scheme,
        layered_prompt=args.layered_prompt,
        resume_existing=args.resume_existing,
        use_discovery_cache=not args.no_discovery_cache,
        refresh_discovery_cache=args.refresh_discovery_cache,
        progress_every=args.progress_every,
    )
    result = run_build(config)
    print(json.dumps(result["stats"], indent=2, sort_keys=True))
    print(f"Processed outputs written to: {result['output_root']}")


if __name__ == "__main__":
    main()
