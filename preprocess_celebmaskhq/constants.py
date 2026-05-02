"""Constants for CelebAMask-HQ 7-slot preprocessing."""

from __future__ import annotations

from typing import Final

SOURCE_CLASSES: Final[tuple[str, ...]] = (
    "cloth",
    "ear_r",
    "eye_g",
    "hair",
    "hat",
    "l_brow",
    "l_ear",
    "l_eye",
    "l_lip",
    "mouth",
    "neck",
    "neck_l",
    "nose",
    "r_brow",
    "r_ear",
    "r_eye",
    "skin",
    "u_lip",
)

SLOT_NAMES: Final[tuple[str, ...]] = (
    "BG",
    "HAIR",
    "FACE_SKIN",
    "EYES",
    "MOUTH",
    "CLOTH",
    "ACCESSORY",
)

SLOT_TO_SOURCE_CLASSES: Final[dict[str, tuple[str, ...]]] = {
    "BG": (),
    "HAIR": ("hair",),
    "FACE_SKIN": ("skin", "nose", "neck", "l_ear", "r_ear"),
    "EYES": ("l_eye", "r_eye", "l_brow", "r_brow"),
    "MOUTH": ("mouth", "u_lip", "l_lip"),
    "CLOTH": ("cloth",),
    "ACCESSORY": ("eye_g", "hat", "ear_r", "neck_l"),
}

SOURCE_TO_SLOT: Final[dict[str, str]] = {
    source_class: slot_name
    for slot_name, source_classes in SLOT_TO_SOURCE_CLASSES.items()
    for source_class in source_classes
}

EXPECTED_MASK_SIZE: Final[tuple[int, int]] = (512, 512)
EXPECTED_IMAGE_SIZE: Final[tuple[int, int]] = (1024, 1024)
MASK_SUFFIX: Final[str] = ".png"
IMAGE_SUFFIX: Final[str] = ".jpg"
DEFAULT_SPLIT_SEED: Final[int] = 1337
DEFAULT_SPLIT_RATIOS: Final[tuple[float, float, float]] = (0.8, 0.1, 0.1)

SLOT_COLORS: Final[tuple[tuple[int, int, int], ...]] = (
    (20, 20, 20),
    (231, 111, 81),
    (242, 204, 143),
    (38, 70, 83),
    (214, 40, 40),
    (42, 157, 143),
    (123, 44, 191),
)
