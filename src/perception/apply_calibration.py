"""
Phase 2: Apply homography + zone tagging to existing Phase 1 detection logs.
Fills in world_xy and zone for every detection, in place per-clip JSON file.
"""

import sys
import json
from pathlib import Path

# Make this importable regardless of current working directory
sys.path.insert(0, str(Path(__file__).parent))
from homography import load_homography, project_point
from zones import get_zone

# Project root = two levels up from this file (src/perception/ -> project root)
ROOT = Path(__file__).parent.parent.parent


def bbox_bottom_center(bbox_px):
    x1, y1, x2, y2 = bbox_px
    return ((x1 + x2) / 2, y2)  # bottom-center = point touching the ground


def process_clip(camera_id, detections_dir=None, calib_dir=None):
    detections_dir = detections_dir or ROOT / "data" / "detections"
    calib_dir = calib_dir or ROOT / "data" / "calibration"

    json_path = Path(detections_dir) / f"{camera_id}.json"
    with open(json_path) as f:
        logs = json.load(f)

    H = load_homography(camera_id, calib_dir)

    for frame in logs:
        for det in frame["detections"]:
            px_point = bbox_bottom_center(det["bbox_px"])
            world_xy = project_point(H, px_point)
            det["world_xy"] = [round(world_xy[0], 2), round(world_xy[1], 2)]
            det["zone"] = get_zone(world_xy)

    with open(json_path, "w") as f:
        json.dump(logs, f, indent=2)

    n_tagged = sum(1 for frame in logs for d in frame["detections"] if d["zone"])
    n_total = sum(len(frame["detections"]) for frame in logs)
    print(f"[{camera_id}] updated {n_total} detections, {n_tagged} landed inside a zone")


if __name__ == "__main__":
    for camera_id in ["MVI_20011", "MVI_20012", "MVI_20032", "MVI_20033", "MVI_20034", "MVI_40992"]:
        process_clip(camera_id)
