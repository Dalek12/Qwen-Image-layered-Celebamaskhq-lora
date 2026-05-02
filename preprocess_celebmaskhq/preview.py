"""Preview rendering utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .constants import SLOT_COLORS, SLOT_NAMES


def colorize_label_map(label_map: np.ndarray) -> np.ndarray:
    """Convert a label map into an RGB visualization."""

    palette = np.asarray(SLOT_COLORS, dtype=np.uint8)
    return palette[label_map]


def save_preview_sheet(
    image_rgb: np.ndarray,
    label_map: np.ndarray,
    slot_masks: np.ndarray,
    output_path: Path,
    sample_id: str,
) -> None:
    """Save a compact preview sheet with RGB, label map, and slot masks."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    axes[0].imshow(image_rgb)
    axes[0].set_title(f"RGB {sample_id}")
    axes[0].axis("off")

    axes[1].imshow(colorize_label_map(label_map))
    axes[1].set_title("Merged 7-class")
    axes[1].axis("off")

    axes[2].imshow(slot_masks[0], cmap="gray")
    axes[2].set_title("BG")
    axes[2].axis("off")

    for slot_index, slot_name in enumerate(SLOT_NAMES[1:], start=1):
        axis = axes[slot_index + 2]
        axis.imshow(slot_masks[slot_index], cmap="gray")
        axis.set_title(slot_name)
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
