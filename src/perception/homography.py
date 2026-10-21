"""
Phase 2: Camera calibration via homography.
Maps pixel coordinates -> flat ground-plane world coordinates (meters),
using 4+ manually picked correspondence points per camera.
"""

import json
from pathlib import Path

import cv2
import numpy as np


def compute_homography(pixel_pts, world_pts):
    """pixel_pts, world_pts: list of (x, y) tuples, same length, >=4 points."""
    src = np.array(pixel_pts, dtype=np.float32)
    dst = np.array(world_pts, dtype=np.float32)
    H, status = cv2.findHomography(src, dst)
    return H


def save_calibration(camera_id, pixel_pts, world_pts, out_dir="data/calibration"):
    H = compute_homography(pixel_pts, world_pts)
    out_path = Path(out_dir) / f"{camera_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "camera_id": camera_id,
            "pixel_points": pixel_pts,
            "world_points": world_pts,
            "homography_matrix": H.tolist(),
        }, f, indent=2)
    print(f"[{camera_id}] calibration saved -> {out_path}")
    return H


def load_homography(camera_id, calib_dir="data/calibration"):
    path = Path(calib_dir) / f"{camera_id}.json"
    with open(path) as f:
        data = json.load(f)
    return np.array(data["homography_matrix"])


def project_point(H, px_point):
    """px_point: (x, y) pixel coordinate -> returns (world_x, world_y) in meters."""
    pt = np.array([[px_point]], dtype=np.float32)  # shape (1,1,2)
    world = cv2.perspectiveTransform(pt, H)
    return float(world[0][0][0]), float(world[0][0][1])


if __name__ == "__main__":
    # Calibration for MVI_20011 — near-lane correspondence points
    pixel_pts = [(505, 505), (700, 505), (555, 150), (645, 150)]
    world_pts = [(0, 0), (3.5, 0), (0, 20), (3.5, 20)]

    save_calibration("MVI_20011", pixel_pts, world_pts)

    # quick sanity test
    H = load_homography("MVI_20011")
    test_px = (600, 400)
    print("Test projection of", test_px, "->", project_point(H, test_px))
