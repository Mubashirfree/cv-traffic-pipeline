"""
Phase 3: Crop-level CLIP embeddings — for each 3s clip, crop every
tracked object's bbox from its source frame and embed it (ForeSea-style).
Produces one mean-pooled "crop" embedding per clip, representing all
tracked entities visible in that window.
"""

import sys
import json
from pathlib import Path

import numpy as np
from PIL import Image

# Make this importable and path-safe regardless of current working directory
sys.path.insert(0, str(Path(__file__).parent))
from clip_encoder import embed_image

ROOT = Path(__file__).parent.parent.parent
SOURCE_FPS = 25


def ts_to_frame_number(ts_seconds: float) -> int:
    return round(ts_seconds * SOURCE_FPS) + 1


def hhmmss_to_seconds(ts: str) -> float:
    ts = ts.rstrip('.')
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def load_detections(camera_id, detections_dir=None):
    detections_dir = detections_dir or ROOT / "data" / "detections"
    with open(Path(detections_dir) / f"{camera_id}.json") as f:
        return json.load(f)


def crop_embeddings_for_clip(camera_id, start_ts, end_ts, detections, raw_dir=None):
    raw_dir = raw_dir or ROOT / "data" / "raw"
    embeddings = []
    tracked_entities = set()

    for frame in detections:
        frame_sec = hhmmss_to_seconds(frame["frame_ts"])
        if not (start_ts <= frame_sec < end_ts):
            continue

        frame_num = ts_to_frame_number(frame_sec)
        img_path = Path(raw_dir) / camera_id / f"img{frame_num:05d}.jpg"
        if not img_path.exists():
            continue

        image = Image.open(img_path).convert("RGB")

        for det in frame["detections"]:
            x1, y1, x2, y2 = det["bbox_px"]
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image.crop((x1, y1, x2, y2))
            embeddings.append(embed_image(crop))
            tracked_entities.add(f"{det['class']}_{det['track_id']}")

    return embeddings, list(tracked_entities)


def mean_pool_crop_embedding(camera_id, start_ts, end_ts, detections):
    embeddings, entities = crop_embeddings_for_clip(camera_id, start_ts, end_ts, detections)
    if not embeddings:
        return None, entities
    mean_vec = np.mean(np.array(embeddings), axis=0)
    mean_vec = mean_vec / np.linalg.norm(mean_vec)
    return mean_vec.tolist(), entities


if __name__ == "__main__":
    dets = load_detections("MVI_20011")
    vec, entities = mean_pool_crop_embedding("MVI_20011", 0.0, 3.0, dets)
    print("Entities found:", entities)
    print("Crop embedding dim:", len(vec) if vec else None)
