"""
Phase 1: YOLOv8n + ByteTrack detection & tracking.
Reads a video clip, runs tracking, writes per-frame JSON logs
matching the schema in 01_Project_Structure_and_Tech_Stack.md §4.
"""

import json
from pathlib import Path
from datetime import timedelta

from ultralytics import YOLO

# COCO class indices (Ultralytics default order) we care about
TARGET_CLASSES = {
    0: "person",
    2: "car",
    5: "bus",
    7: "truck",
    24: "backpack",
    26: "handbag",
}

SAMPLE_FPS = 2.5          # sample at 2-3 FPS per plan
SOURCE_FPS = 25           # UA-DETRAC native frame rate
VID_STRIDE = max(1, round(SOURCE_FPS / SAMPLE_FPS))


def frame_ts(frame_idx: int, fps: float = SOURCE_FPS) -> str:
    seconds = frame_idx / fps
    return str(timedelta(seconds=seconds))[:8]  # HH:MM:SS


def run_detect_track(clip_path: str, camera_id: str, out_dir: str):
    model = YOLO("yolov8n.pt")
    out_path = Path(out_dir) / f"{camera_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logs = []
    results = model.track(
        source=clip_path,
        tracker="bytetrack.yaml",
        classes=list(TARGET_CLASSES.keys()),
        vid_stride=VID_STRIDE,
        persist=True,
        stream=True,
        verbose=False,
    )

    for frame_idx, r in enumerate(results):
        detections = []
        if r.boxes is not None and r.boxes.id is not None:
            for box, track_id, cls, conf in zip(
                r.boxes.xyxy.cpu().tolist(),
                r.boxes.id.cpu().tolist(),
                r.boxes.cls.cpu().tolist(),
                r.boxes.conf.cpu().tolist(),
            ):
                detections.append({
                    "track_id": int(track_id),
                    "class": TARGET_CLASSES.get(int(cls), "unknown"),
                    "bbox_px": [round(x, 1) for x in box],
                    "world_xy": None,   # filled in Phase 2
                    "zone": None,       # filled in Phase 2
                    "confidence": round(float(conf), 3),
                })

        logs.append({
            "frame_ts": frame_ts(frame_idx * VID_STRIDE),
            "camera_id": camera_id,
            "detections": detections,
        })

    with open(out_path, "w") as f:
        json.dump(logs, f, indent=2)

    print(f"[{camera_id}] wrote {len(logs)} frame logs -> {out_path}")


if __name__ == "__main__":
    CLIPS_DIR = Path("data/videos")
    OUT_DIR = "data/detections"

    for clip in sorted(CLIPS_DIR.glob("*.mp4")):
        camera_id = clip.stem
        run_detect_track(str(clip), camera_id, OUT_DIR)
