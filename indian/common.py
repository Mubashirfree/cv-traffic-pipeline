"""Shared constants for the Indian-data (UVH-26) work."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UVH_DIR = ROOT / "data" / "uvh26"
YOLO_DIR = ROOT / "data" / "uvh26_yolo"
FIG_DIR = ROOT / "report" / "figures"
EVAL_DIR = ROOT / "eval"

CLASSES = ["car", "bus", "truck", "auto_rickshaw", "motorcycle", "bicycle"]

CLASS_MAP = {
    "hatchback": "car", "sedan": "car", "suv": "car", "muv": "car", "van": "car",
    "bus": "bus", "mini-bus": "bus", "tempo-traveller": "bus",
    "truck": "truck", "lcv": "truck",
    "three-wheeler": "auto_rickshaw",
    "two-wheeler": "motorcycle",
    "bicycle": "bicycle",
    "other": None,
}

COCO_TO_OURS = {2: 0, 5: 1, 7: 2, 3: 4, 1: 5}
DETRAC_TO_OURS = {0: 0, 1: 1, 2: 2}


def norm_name(s: str) -> str:
    return s.strip().lower().replace(" ", "-").replace("_", "-")
