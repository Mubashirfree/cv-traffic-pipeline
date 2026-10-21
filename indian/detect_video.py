"""Detector + ByteTrack on any video; writes JSON, annotated mp4, track statistics.
  python -m indian.detect_video --video data/indian_videos/clip1.mp4 --weights runs/indian/v1/weights/best.pt
"""
import argparse, json
from pathlib import Path

import cv2
import numpy as np

from indian.common import ROOT, FIG_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--fps", type=float, default=2.5)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    names = model.names
    cap = cv2.VideoCapture(args.video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(src_fps / args.fps))
    W, H = int(cap.get(3)), int(cap.get(4))
    name = Path(args.video).stem
    out_dir = ROOT / "data" / "detections_indian"; out_dir.mkdir(parents=True, exist_ok=True)
    vid_out = ROOT / "data" / "annotated"; vid_out.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(vid_out / f"{name}_annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             src_fps / stride, (W, H))
    log, idx, tracks = [], 0, {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            r = model.track(frame, tracker="bytetrack.yaml", conf=args.conf, imgsz=args.imgsz,
                            persist=True, verbose=False, device=0)[0]
            dets = []
            if r.boxes is not None and r.boxes.id is not None:
                for b, t, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.id.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    dets.append({"track_id": int(t), "class": names[int(c)], "bbox_px": [round(v, 1) for v in b],
                                 "world_xy": None, "zone": None, "confidence": round(float(s), 4)})
                    tracks.setdefault(int(t), {"cls": names[int(c)], "n": 0})["n"] += 1
            log.append({"frame_idx": idx, "frame_ts": round(idx / src_fps, 3), "camera_id": name, "detections": dets})
            writer.write(r.plot())
        idx += 1
    cap.release(); writer.release()
    (out_dir / f"{name}.json").write_text(json.dumps(log))

    lens = np.array([t["n"] for t in tracks.values()]) if tracks else np.array([0])
    by_cls = {}
    for t in tracks.values():
        by_cls[t["cls"]] = by_cls.get(t["cls"], 0) + 1
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    md = [f"### Tracking statistics: {name} (no ground truth; descriptive only)", "",
          f"- processed frames: {len(log)} at ~{src_fps/stride:.2f} fps", f"- unique track IDs: {len(tracks)}",
          f"- mean track length: {lens.mean():.1f} frames; median {np.median(lens):.0f}; tracks >= 5 frames: {(lens>=5).sum()}",
          f"- tracks by class: {by_cls}", ""]
    (FIG_DIR / f"indian_video_tracking_{name}.md").write_text("\n".join(md))
    print("\n".join(md)); print("annotated video:", vid_out / f"{name}_annotated.mp4")


if __name__ == "__main__":
    main()
