#!/usr/bin/env python
"""Evaluate generated Qwen layered outputs against packaged target layers."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_LAYER_NAMES = ["BG", "HAIR", "FACE"]
RESAMPLE_NEAREST = getattr(getattr(Image, "Resampling", Image), "NEAREST")
RESAMPLE_BILINEAR = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
RESAMPLE_BICUBIC = getattr(getattr(Image, "Resampling", Image), "BICUBIC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--alpha-threshold", type=int, default=128)
    parser.add_argument("--expected-layer-names", nargs="+", default=DEFAULT_LAYER_NAMES)
    parser.add_argument("--empty-coverage-threshold", type=float, default=0.001)
    parser.add_argument("--full-coverage-threshold", type=float, default=0.95)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison_root = args.comparison_root.resolve()
    eval_root = args.eval_root.resolve()
    output_root = args.output_root.resolve()
    layer_names = [name.strip() for name in args.expected_layer_names if name.strip()]

    if not comparison_root.is_dir():
        raise FileNotFoundError(f"Missing comparison root: {comparison_root}")
    if not eval_root.is_dir():
        raise FileNotFoundError(f"Missing eval root: {eval_root}")
    if not layer_names:
        raise ValueError("--expected-layer-names must include at least one layer name.")
    if not 0 <= args.alpha_threshold <= 255:
        raise ValueError("--alpha-threshold must be in [0, 255].")
    if output_root.exists():
        if args.force:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(f"Output root already exists: {output_root}. Use --force to rebuild it.")
    output_root.mkdir(parents=True, exist_ok=True)

    sample_dirs = discover_sample_dirs(comparison_root)
    if not sample_dirs:
        raise FileNotFoundError(f"No selected_sample.json files found under comparison root: {comparison_root}")

    per_layer_rows: list[dict[str, Any]] = []
    per_sample_rows: list[dict[str, Any]] = []
    for sample_dir in sample_dirs:
        layer_rows, sample_row = evaluate_sample_dir(
            sample_dir=sample_dir,
            comparison_root=comparison_root,
            eval_root=eval_root,
            layer_names=layer_names,
            alpha_threshold=args.alpha_threshold,
            empty_coverage_threshold=args.empty_coverage_threshold,
            full_coverage_threshold=args.full_coverage_threshold,
        )
        per_layer_rows.extend(layer_rows)
        per_sample_rows.append(sample_row)

    summary_rows = summarize_by_variant(per_layer_rows, per_sample_rows, layer_names)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_root": str(comparison_root),
        "eval_root": str(eval_root),
        "output_root": str(output_root),
        "alpha_threshold": args.alpha_threshold,
        "expected_layer_names": layer_names,
        "sample_dir_count": len(sample_dirs),
        "per_layer_row_count": len(per_layer_rows),
        "variant_count": len(summary_rows),
        "outputs": {
            "per_sample_layer_csv": "metrics_per_sample_layer.csv",
            "summary_by_variant_csv": "metrics_summary_by_variant.csv",
            "summary_json": "metrics_summary.json",
        },
        "summary_by_variant": summary_rows,
    }

    write_csv(output_root / "metrics_per_sample_layer.csv", per_layer_rows)
    write_csv(output_root / "metrics_per_sample.csv", per_sample_rows)
    write_csv(output_root / "metrics_summary_by_variant.csv", summary_rows)
    write_json(output_root / "metrics_summary.json", manifest)

    print(json.dumps({k: manifest[k] for k in ["sample_dir_count", "per_layer_row_count", "variant_count"]}, indent=2))
    print_markdown_summary(summary_rows)
    print(f"\nWrote metrics to: {output_root}")


def discover_sample_dirs(comparison_root: Path) -> list[Path]:
    sample_files = sorted(comparison_root.rglob("selected_sample.json"))
    return [path.parent for path in sample_files]


def evaluate_sample_dir(
    *,
    sample_dir: Path,
    comparison_root: Path,
    eval_root: Path,
    layer_names: list[str],
    alpha_threshold: int,
    empty_coverage_threshold: float,
    full_coverage_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_sample = read_json(sample_dir / "selected_sample.json")
    variant = read_json_optional(sample_dir / "variant.json")
    variant_info = infer_variant_info(sample_dir, comparison_root, variant)

    selected_layer_names = selected_sample.get("layer_names")
    if selected_layer_names and list(selected_layer_names) != layer_names:
        raise ValueError(
            f"Unexpected layer_names for sample {sample_dir}: {selected_layer_names}. "
            f"Expected {layer_names}."
        )
    target_layer_paths = selected_sample.get("layer_paths")
    if not isinstance(target_layer_paths, list) or len(target_layer_paths) != len(layer_names):
        raise ValueError(
            f"selected_sample.json in {sample_dir} must contain exactly {len(layer_names)} layer_paths."
        )

    pred_layers = [load_rgba(find_predicted_layer(sample_dir, index)) for index in range(len(layer_names))]
    target_relative_paths = [metadata_relative_path(path_str) for path_str in target_layer_paths]
    target_layers = [
        load_rgba(resolve_eval_file(eval_root, relative_path, sample_dir=sample_dir))
        for relative_path in target_relative_paths
    ]
    pred_size = pred_layers[0].size
    for layer_path, layer in zip([find_predicted_layer(sample_dir, i) for i in range(len(layer_names))], pred_layers):
        if layer.size != pred_size:
            raise ValueError(f"Predicted layer size mismatch in {sample_dir}: {layer_path} has {layer.size}, expected {pred_size}")

    pred_alphas = [image_alpha_array(layer) for layer in pred_layers]
    target_alphas_soft = [
        image_alpha_array(resize_image(layer, pred_size, RESAMPLE_BILINEAR))
        for layer in target_layers
    ]
    target_alphas_binary = [
        image_alpha_array(resize_image(layer, pred_size, RESAMPLE_NEAREST))
        for layer in target_layers
    ]

    threshold = alpha_threshold / 255.0
    pred_masks = [alpha >= threshold for alpha in pred_alphas]
    target_masks = [alpha >= threshold for alpha in target_alphas_binary]
    overlap_stats = compute_overlap_stats(pred_masks)
    composite_stats = compute_composite_stats(sample_dir, pred_layers, pred_size)

    base_row = {
        **variant_info,
        "sample_id": int(selected_sample["sample_id"]),
        "sample_id_str": str(selected_sample.get("sample_id_str", f"{int(selected_sample['sample_id']):05d}")),
        "split": selected_sample.get("split"),
        "sample_dir": str(sample_dir),
    }
    sample_row = {
        **base_row,
        **overlap_stats,
        **composite_stats,
    }

    layer_rows: list[dict[str, Any]] = []
    for index, layer_name in enumerate(layer_names):
        pred_alpha = pred_alphas[index]
        target_alpha = target_alphas_soft[index]
        pred_mask = pred_masks[index]
        target_mask = target_masks[index]
        target_path = resolve_eval_file(eval_root, target_relative_paths[index], sample_dir=sample_dir)
        pred_path = find_predicted_layer(sample_dir, index)
        row = {
            **base_row,
            "layer_index": index,
            "layer_name": layer_name,
            "pred_layer_path": str(pred_path),
            "target_layer_path": str(target_path),
            **compute_layer_metrics(
                pred_alpha=pred_alpha,
                target_alpha=target_alpha,
                pred_mask=pred_mask,
                target_mask=target_mask,
                empty_coverage_threshold=empty_coverage_threshold,
                full_coverage_threshold=full_coverage_threshold,
            ),
            **overlap_stats,
            **composite_stats,
        }
        layer_rows.append(row)
    return layer_rows, sample_row


def infer_variant_info(sample_dir: Path, comparison_root: Path, variant: dict[str, Any] | None) -> dict[str, Any]:
    rel_parts = sample_dir.relative_to(comparison_root).parts
    sample_folder = rel_parts[-1] if rel_parts else sample_dir.name
    path_model_variant = rel_parts[-2] if len(rel_parts) >= 2 else "unknown_model"
    path_prompt_variant = rel_parts[-3] if len(rel_parts) >= 3 else "default"

    model_variant_name = path_model_variant
    model_variant_type = None
    checkpoint_name = None
    prompt_variant_name = path_prompt_variant
    prompt_text = None
    use_en_prompt = None
    variant_name = "__".join(rel_parts[:-1]) if len(rel_parts) > 1 else path_model_variant

    if variant:
        if "model_variant" in variant and isinstance(variant["model_variant"], dict):
            model_variant = variant["model_variant"]
            model_variant_name = str(model_variant.get("name", model_variant_name))
            model_variant_type = model_variant.get("type")
            checkpoint_name = model_variant.get("checkpoint_name")
        else:
            model_variant_name = str(variant.get("name", model_variant_name))
            model_variant_type = variant.get("type")
            checkpoint_name = variant.get("checkpoint_name")

        if "prompt_variant" in variant and isinstance(variant["prompt_variant"], dict):
            prompt_variant = variant["prompt_variant"]
            prompt_variant_name = str(prompt_variant.get("name", prompt_variant_name))
            prompt_text = prompt_variant.get("prompt")
            use_en_prompt = prompt_variant.get("use_en_prompt")

        variant_name = str(variant.get("variant_name", variant_name))

    return {
        "variant_name": variant_name,
        "prompt_variant": prompt_variant_name,
        "model_variant": model_variant_name,
        "model_variant_type": model_variant_type,
        "checkpoint_name": checkpoint_name,
        "prompt": prompt_text,
        "use_en_prompt": use_en_prompt,
        "sample_folder": sample_folder,
    }


def compute_layer_metrics(
    *,
    pred_alpha: np.ndarray,
    target_alpha: np.ndarray,
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    empty_coverage_threshold: float,
    full_coverage_threshold: float,
) -> dict[str, Any]:
    intersection = int(np.logical_and(pred_mask, target_mask).sum())
    union = int(np.logical_or(pred_mask, target_mask).sum())
    pred_count = int(pred_mask.sum())
    target_count = int(target_mask.sum())
    total_pixels = int(pred_mask.size)

    iou = safe_ratio(intersection, union, empty_value=1.0)
    dice = safe_ratio(2 * intersection, pred_count + target_count, empty_value=1.0)
    precision = safe_ratio(intersection, pred_count, empty_value=1.0 if target_count == 0 else 0.0)
    recall = safe_ratio(intersection, target_count, empty_value=1.0 if pred_count == 0 else 0.0)

    soft_union = float(np.maximum(pred_alpha, target_alpha).sum())
    soft_intersection = float(np.minimum(pred_alpha, target_alpha).sum())
    soft_iou = safe_ratio(soft_intersection, soft_union, empty_value=1.0)
    alpha_mae = float(np.abs(pred_alpha - target_alpha).mean())

    pred_coverage = safe_ratio(pred_count, total_pixels, empty_value=0.0)
    target_coverage = safe_ratio(target_count, total_pixels, empty_value=0.0)
    coverage_ratio = safe_ratio(pred_coverage, target_coverage, empty_value=None)

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "soft_iou": soft_iou,
        "alpha_mae": alpha_mae,
        "pred_alpha_mean": float(pred_alpha.mean()),
        "target_alpha_mean": float(target_alpha.mean()),
        "pred_coverage": pred_coverage,
        "target_coverage": target_coverage,
        "coverage_ratio": coverage_ratio,
        "pred_pixel_count": pred_count,
        "target_pixel_count": target_count,
        "intersection_pixel_count": intersection,
        "union_pixel_count": union,
        "empty_layer": bool(pred_coverage <= empty_coverage_threshold),
        "full_layer": bool(pred_coverage >= full_coverage_threshold),
    }


def compute_overlap_stats(pred_masks: list[np.ndarray]) -> dict[str, Any]:
    if not pred_masks:
        return {
            "multi_layer_overlap_fraction": 0.0,
            "mean_pairwise_overlap_fraction": 0.0,
        }

    stack = np.stack(pred_masks, axis=0)
    active_count = stack.sum(axis=0)
    multi_layer_overlap_fraction = float((active_count > 1).mean())
    pairwise_values = []
    for left in range(len(pred_masks)):
        for right in range(left + 1, len(pred_masks)):
            pairwise_values.append(float(np.logical_and(pred_masks[left], pred_masks[right]).mean()))
    mean_pairwise_overlap_fraction = float(np.mean(pairwise_values)) if pairwise_values else 0.0
    return {
        "multi_layer_overlap_fraction": multi_layer_overlap_fraction,
        "mean_pairwise_overlap_fraction": mean_pairwise_overlap_fraction,
    }


def compute_composite_stats(sample_dir: Path, pred_layers: list[Image.Image], pred_size: tuple[int, int]) -> dict[str, Any]:
    input_path = sample_dir / "input.png"
    if not input_path.is_file():
        return {
            "composite_rgb_mae": None,
            "composite_rgb_psnr": None,
        }

    input_rgb = np.asarray(resize_image(Image.open(input_path).convert("RGB"), pred_size, RESAMPLE_BICUBIC), dtype=np.float32) / 255.0
    composite_rgb = alpha_composite_rgb(pred_layers, pred_size)
    mse = float(np.mean((composite_rgb - input_rgb) ** 2))
    mae = float(np.mean(np.abs(composite_rgb - input_rgb)))
    psnr = None if mse <= 0 else float(10.0 * math.log10(1.0 / mse))
    return {
        "composite_rgb_mae": mae,
        "composite_rgb_psnr": psnr,
    }


def alpha_composite_rgb(layers: list[Image.Image], size: tuple[int, int]) -> np.ndarray:
    out_rgb = np.zeros((size[1], size[0], 3), dtype=np.float32)
    out_alpha = np.zeros((size[1], size[0], 1), dtype=np.float32)
    for layer in layers:
        rgba = np.asarray(resize_image(layer, size, RESAMPLE_BICUBIC), dtype=np.float32) / 255.0
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3:4]
        out_rgb_premul = out_rgb * out_alpha
        merged_alpha = alpha + out_alpha * (1.0 - alpha)
        merged_rgb_premul = rgb * alpha + out_rgb_premul * (1.0 - alpha)
        out_rgb = np.divide(
            merged_rgb_premul,
            np.maximum(merged_alpha, 1e-8),
            out=np.zeros_like(merged_rgb_premul),
            where=merged_alpha > 1e-8,
        )
        out_alpha = merged_alpha
    return out_rgb


def summarize_by_variant(
    per_layer_rows: list[dict[str, Any]],
    per_sample_rows: list[dict[str, Any]],
    layer_names: list[str],
) -> list[dict[str, Any]]:
    layer_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    sample_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_layer_rows:
        key = (str(row["variant_name"]), str(row["prompt_variant"]), str(row["model_variant"]))
        layer_groups[key].append(row)
    for row in per_sample_rows:
        key = (str(row["variant_name"]), str(row["prompt_variant"]), str(row["model_variant"]))
        sample_groups[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(layer_groups):
        variant_name, prompt_variant, model_variant = key
        rows = layer_groups[key]
        sample_rows = sample_groups.get(key, [])
        summary: dict[str, Any] = {
            "variant_name": variant_name,
            "prompt_variant": prompt_variant,
            "model_variant": model_variant,
            "sample_count": len({row["sample_id"] for row in rows}),
            "layer_row_count": len(rows),
            "macro_iou": mean_field(rows, "iou"),
            "macro_dice": mean_field(rows, "dice"),
            "macro_precision": mean_field(rows, "precision"),
            "macro_recall": mean_field(rows, "recall"),
            "macro_soft_iou": mean_field(rows, "soft_iou"),
            "macro_alpha_mae": mean_field(rows, "alpha_mae"),
            "empty_layer_rate": mean_bool_field(rows, "empty_layer"),
            "full_layer_rate": mean_bool_field(rows, "full_layer"),
            "multi_layer_overlap_fraction": mean_field(sample_rows, "multi_layer_overlap_fraction"),
            "mean_pairwise_overlap_fraction": mean_field(sample_rows, "mean_pairwise_overlap_fraction"),
            "composite_rgb_mae": mean_field(sample_rows, "composite_rgb_mae"),
            "composite_rgb_psnr": mean_field(sample_rows, "composite_rgb_psnr"),
        }
        first = rows[0]
        summary["model_variant_type"] = first.get("model_variant_type")
        summary["checkpoint_name"] = first.get("checkpoint_name")

        for layer_name in layer_names:
            layer_rows = [row for row in rows if row["layer_name"] == layer_name]
            prefix = layer_name.lower()
            summary[f"{prefix}_iou"] = mean_field(layer_rows, "iou")
            summary[f"{prefix}_dice"] = mean_field(layer_rows, "dice")
            summary[f"{prefix}_soft_iou"] = mean_field(layer_rows, "soft_iou")
            summary[f"{prefix}_alpha_mae"] = mean_field(layer_rows, "alpha_mae")
            summary[f"{prefix}_pred_coverage"] = mean_field(layer_rows, "pred_coverage")
            summary[f"{prefix}_target_coverage"] = mean_field(layer_rows, "target_coverage")
            summary[f"{prefix}_coverage_ratio"] = mean_field(layer_rows, "coverage_ratio")
        summary_rows.append(summary)
    return summary_rows


def print_markdown_summary(summary_rows: list[dict[str, Any]]) -> None:
    columns = [
        ("variant", "variant_name"),
        ("samples", "sample_count"),
        ("macro IoU", "macro_iou"),
        ("soft IoU", "macro_soft_iou"),
        ("alpha MAE", "macro_alpha_mae"),
        ("empty", "empty_layer_rate"),
        ("full", "full_layer_rate"),
        ("overlap", "multi_layer_overlap_fraction"),
    ]
    print("\nMarkdown summary:")
    print("| " + " | ".join(title for title, _ in columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for row in summary_rows:
        cells = []
        for _, key in columns:
            value = row.get(key)
            if isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append("" if value is None else str(value))
        print("| " + " | ".join(cells) + " |")


def mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [row.get(field) for row in rows]
    numeric = [float(value) for value in values if value is not None and not isinstance(value, bool)]
    return float(np.mean(numeric)) if numeric else None


def mean_bool_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [row.get(field) for row in rows if row.get(field) is not None]
    return float(np.mean([1.0 if bool(value) else 0.0 for value in values])) if values else None


def find_predicted_layer(sample_dir: Path, index: int) -> Path:
    candidates = [
        sample_dir / f"layer_{index:02d}.png",
        sample_dir / f"layer_{index}.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    expected = ", ".join(str(candidate.name) for candidate in candidates)
    raise FileNotFoundError(f"Missing predicted layer {index} in {sample_dir}. Expected one of: {expected}")


def metadata_relative_path(path_str: str) -> Path:
    normalized = str(path_str).replace("\\", "/").strip("/")
    path = Path(normalized)
    if path.is_absolute():
        raise ValueError(f"Expected a relative metadata path, got absolute path: {path_str}")
    return path


def resolve_eval_file(eval_root: Path, relative_path: Path, *, sample_dir: Path) -> Path:
    candidates = [eval_root / relative_path]
    for child in eval_root.iterdir():
        if child.is_dir():
            candidates.append(child / relative_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    tar_hint = ""
    if (eval_root / "data.tar").is_file() or any(child.name == "data.tar" for child in eval_root.glob("*/data.tar")):
        tar_hint = " It looks like data.tar exists; extract the eval package before running metrics."
    raise FileNotFoundError(
        f"Target layer missing for {sample_dir}: {relative_path}. "
        f"Make sure --eval-root points to the extracted eval package folder containing layered_layers/ and metadata/.{tar_hint}"
    )


def load_rgba(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def image_alpha_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.getchannel("A"), dtype=np.float32) / 255.0


def resize_image(image: Image.Image, size: tuple[int, int], resample: Image.Resampling) -> Image.Image:
    if image.size == size:
        return image
    return image.resize(size, resample=resample)


def safe_ratio(numerator: float, denominator: float, *, empty_value: float | None) -> float | None:
    if denominator == 0:
        return empty_value
    return float(numerator / denominator)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return read_json(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def csv_value(value: Any) -> Any:
    value = to_jsonable(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
