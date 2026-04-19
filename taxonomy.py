"""
taxonomy.py
───────────
Custom class taxonomy for Indian unstructured road scenes.

16 semantic classes covering drivable surfaces, road hazards, traffic
participants, animals, and infrastructure — tuned for the diversity
encountered on Indian roads.
"""

from __future__ import annotations

# (class_id, name, BGR_colour)
CLASSES: list[tuple[int, str, tuple[int, int, int]]] = [
    (0,  "drivable_road",       (128,  64, 128)),
    (1,  "pothole",             (  0,   0, 192)),
    (2,  "speed_bump",          (  0,  60, 100)),
    (3,  "puddle",              ( 70, 130, 180)),
    (4,  "pedestrian",          ( 60,  20, 220)),
    (5,  "cyclist",             (  0,   0, 255)),
    (6,  "motorcyclist",        (230,   0,   0)),
    (7,  "car",                 (142,   0,   0)),
    (8,  "auto_rickshaw",       (100,  60,   0)),
    (9,  "truck",               (100,  80,   0)),
    (10, "bus",                 ( 32,  11, 119)),
    (11, "tractor",             (255,  64, 128)),
    (12, "cattle",              ( 35, 142, 107)),
    (13, "dog",                 (152, 251, 152)),
    (14, "construction_debris", ( 30, 170, 250)),
    (15, "barrier",             (  0, 220, 220)),
]

# Convenience lookups
CLASS_NAMES:  list[str]                  = [c[1] for c in CLASSES]
CLASS_COLORS: list[tuple[int, int, int]] = [c[2] for c in CLASSES]   # BGR
ID_TO_NAME:   dict[int, str]             = {c[0]: c[1] for c in CLASSES}
NAME_TO_ID:   dict[str, int]             = {c[1]: c[0] for c in CLASSES}

# Also index by space-separated form (Grounding DINO returns "auto rickshaw")
_SPACE_TO_ID: dict[str, int] = {
    name.replace("_", " "): cid for cid, name in ID_TO_NAME.items()
}

# Default Grounding DINO text prompt — each class as a phrase ending with "."
GROUNDED_SAM_PROMPT: str = (
    " . ".join(n.replace("_", " ") for n in CLASS_NAMES) + " ."
)

# Cityscapes-style class names for mapping semantic model output
CITYSCAPES_TO_TAXONOMY: dict[str, str] = {
    "road":        "drivable_road",
    "sidewalk":    "drivable_road",
    "person":      "pedestrian",
    "rider":       "motorcyclist",
    "car":         "car",
    "truck":       "truck",
    "bus":         "bus",
    "motorcycle":  "motorcyclist",
    "bicycle":     "cyclist",
}


def color_for(class_id: int) -> tuple[int, int, int]:
    """Return the BGR colour for *class_id* (cycles through palette)."""
    return CLASS_COLORS[class_id % len(CLASS_COLORS)]


def name_for(class_id: int) -> str:
    """Return the class name for *class_id*, or ``'unknown'``."""
    return ID_TO_NAME.get(class_id, "unknown")


def id_for(class_name: str) -> int:
    """
    Return the class ID for *class_name*.

    Tries exact match first, then underscore/space normalisation.
    Returns ``-1`` if not found.
    """
    if class_name in NAME_TO_ID:
        return NAME_TO_ID[class_name]
    normed = class_name.strip().lower()
    if normed in NAME_TO_ID:
        return NAME_TO_ID[normed]
    space_form = normed.replace("_", " ")
    if space_form in _SPACE_TO_ID:
        return _SPACE_TO_ID[space_form]
    under_form = normed.replace(" ", "_")
    if under_form in NAME_TO_ID:
        return NAME_TO_ID[under_form]
    return -1
