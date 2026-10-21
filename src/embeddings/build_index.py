"""
Phase 3: Orchestration — populate Postgres + Qdrant for every sub-clip.
"""

import sys
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from clip_encoder import embed_image
from crop_encoder import hhmmss_to_seconds, ts_to_frame_number, crop_embeddings_for_clip
from perception.zones import get_zone_polygons

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.storage.postgres_client import insert_clip, insert_detection, insert_zone
from src.storage.qdrant_client import upsert_embedding

CAMERAS = ["MVI_20011", "MVI_20012", "MVI_20032", "MVI_20033", "MVI_20034", "MVI_40992"]


def seconds_to_hhmmss(s):
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def frame_embedding_for_window(camera_id, start_ts, end_ts, detections, raw_dir):
    """Mean-pooled whole-frame embedding over every sampled frame in the window."""
    embeddings = []
    seen_frames = set()
    for frame in detections:
        frame_sec = hhmmss_to_seconds(frame["frame_ts"])
        if not (start_ts <= frame_sec < end_ts):
            continue
        frame_num = ts_to_frame_number(frame_sec)
        if frame_num in seen_frames:
            continue
        seen_frames.add(frame_num)
        img_path = Path(raw_dir) / camera_id / f"img{frame_num:05d}.jpg"
        if not img_path.exists():
            continue
        image = Image.open(img_path).convert("RGB")
        embeddings.append(embed_image(image))
    if not embeddings:
        return None
    mean_vec = np.mean(np.array(embeddings), axis=0)
    mean_vec = mean_vec / np.linalg.norm(mean_vec)
    return mean_vec.tolist()


def dominant_zone(detections, start_ts, end_ts):
    zones = []
    for frame in detections:
        frame_sec = hhmmss_to_seconds(frame["frame_ts"])
        if not (start_ts <= frame_sec < end_ts):
            continue
        for det in frame["detections"]:
            if det["zone"]:
                zones.append(det["zone"])
    if not zones:
        return None
    return Counter(zones).most_common(1)[0][0]


def main():
    # 1. Insert zones (same polygon set applies per calibrated camera)
    zone_polys = get_zone_polygons()
    for camera_id in CAMERAS:
        for name, coords in zone_polys.items():
            insert_zone(camera_id, name, coords)
    print(f"Inserted zones for {len(CAMERAS)} cameras.")

    # 2. Load manifest + per-camera detections (cached)
    with open(ROOT / "data" / "clips" / "manifest.json") as f:
        manifest = json.load(f)

    det_cache = {}
    raw_dir = ROOT / "data" / "raw"

    for i, clip in enumerate(manifest):
        camera_id = clip["camera_id"]
        clip_id = clip["clip_id"]
        start_ts, end_ts = clip["start_ts"], clip["end_ts"]

        if camera_id not in det_cache:
            with open(ROOT / "data" / "detections" / f"{camera_id}.json") as f:
                det_cache[camera_id] = json.load(f)
        detections = det_cache[camera_id]

        # Postgres: clip row
        insert_clip(clip_id, camera_id, start_ts, end_ts, clip["path"])

        # Postgres: detection rows for this window
        for frame in detections:
            frame_sec = hhmmss_to_seconds(frame["frame_ts"])
            if not (start_ts <= frame_sec < end_ts):
                continue
            for det in frame["detections"]:
                insert_detection(
                    clip_id, det["track_id"], det["class"], det["bbox_px"],
                    det["world_xy"], det["zone"], det["confidence"],
                )

        # Embeddings
        frame_vec = frame_embedding_for_window(camera_id, start_ts, end_ts, detections, raw_dir)
        crop_embeds, tracked_entities = crop_embeddings_for_clip(camera_id, start_ts, end_ts, detections)
        crop_vec = None
        if crop_embeds:
            mean_vec = np.mean(np.array(crop_embeds), axis=0)
            crop_vec = (mean_vec / np.linalg.norm(mean_vec)).tolist()

        zone = dominant_zone(detections, start_ts, end_ts)
        base_payload = {
            "timestamp": seconds_to_hhmmss(start_ts),
            "camera_id": camera_id,
            "tracked_entities": tracked_entities,
            "zone": zone,
            "clip_path": clip["path"],
        }

        if frame_vec:
            upsert_embedding(frame_vec, {**base_payload, "embedding_type": "frame"})
        if crop_vec:
            upsert_embedding(crop_vec, {**base_payload, "embedding_type": "crop"})

        print(f"[{i+1}/{len(manifest)}] {clip_id}: frame={'ok' if frame_vec else 'skip'}, crop={'ok' if crop_vec else 'skip'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
