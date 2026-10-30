"""
Indian video -> searchable 3-second clips.  For every video in data/indian_videos/:
  cut into 3 s H.264 clips -> detect + ByteTrack (Indian weights; lower threshold for dark clips)
  -> CLIP frame + crop embeddings -> Qdrant collection 'indian_clips' (payload includes lighting day/night).
  python -m indian.ingest_video --weights models/indian_best.pt
"""
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from indian.common import ROOT, FIG_DIR, CLASSES

VID_DIR = ROOT / "data" / "indian_videos"
CLIP_DIR = ROOT / "data" / "indian_clips"
COLLECTION = "indian_clips"
CLIP_S = 3
NIGHT_BRIGHTNESS = 70
NIGHT_CONF = 0.12


def cut(video, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.mp4"):
        old.unlink()
    scale = r"scale=w=if(gt(iw\,ih)\,-2\,540):h=if(gt(iw\,ih)\,540\,-2)"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-an", "-vf", scale,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
           "-force_key_frames", f"expr:gte(t,n_forced*{CLIP_S})", "-f", "segment",
           "-segment_time", str(CLIP_S), "-reset_timestamps", "1", str(out_dir / f"{video.stem}_%03d.mp4")]
    subprocess.run(cmd, check=True)
    return sorted(out_dir.glob(f"{video.stem}_*.mp4"))


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def unit(v):
    v = np.asarray(v, dtype="float32")
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--no-qdrant", action="store_true")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found. Run: sudo apt install -y ffmpeg")
    videos = sorted(p for p in VID_DIR.glob("*") if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm"))
    if not videos:
        sys.exit(f"No videos in {VID_DIR}")

    from ultralytics import YOLO
    from src.embeddings.clip_encoder import embed_image
    client = None
    if not args.no_qdrant:
        from qdrant_client.models import VectorParams, Distance
        from src.storage.qdrant_client import get_client
        client = get_client()
        if client.collection_exists(COLLECTION):
            client.delete_collection(COLLECTION)
        client.create_collection(COLLECTION, vectors_config=VectorParams(size=512, distance=Distance.COSINE))
    from qdrant_client.models import PointStruct

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(CLIP_DIR, ignore_errors=True)
    points, stats_md = [], ["### Indian video ingest (descriptive statistics; no ground truth)", ""]
    for video in videos:
        model = YOLO(args.weights)
        names = model.names
        clips = cut(video, CLIP_DIR / video.stem)
        all_tracks, n_clips, n_night = {}, 0, 0
        for ci, clip in enumerate(clips):
            frames, fps = read_frames(clip)
            if len(frames) < fps * 1.5:
                continue
            idx = np.linspace(0, len(frames) - 1, 3).astype(int)
            bright = float(np.mean([cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).mean() for i in idx]))
            night = bright < NIGHT_BRIGHTNESS
            conf = NIGHT_CONF if night else args.conf
            n_night += night
            step = max(1, round(fps / 2.5))
            tracks = {}
            for k in range(0, len(frames), step):
                r = model.track(frames[k], tracker="bytetrack.yaml", conf=conf, imgsz=960,
                                persist=True, verbose=False, device=0)[0]
                if r.boxes is None or r.boxes.id is None:
                    continue
                for b, t, c in zip(r.boxes.xyxy.tolist(), r.boxes.id.tolist(), r.boxes.cls.tolist()):
                    x1, y1, x2, y2 = [int(v) for v in b]
                    area = (x2 - x1) * (y2 - y1)
                    tr = tracks.setdefault(int(t), {"cls": names[int(c)], "n": 0, "area": 0, "crop": None})
                    tr["n"] += 1
                    if area > tr["area"] and x2 - x1 >= 16 and y2 - y1 >= 16:
                        tr["area"] = area
                        tr["crop"] = cv2.cvtColor(frames[k][max(0, y1):y2, max(0, x1):x2], cv2.COLOR_BGR2RGB)
            kept = {t: v for t, v in tracks.items() if v["n"] >= 2}
            fe = unit(np.mean([embed_image(Image.fromarray(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))) for i in idx], axis=0))
            crops = [embed_image(Image.fromarray(v["crop"])) for v in kept.values() if v["crop"] is not None][:20]
            ce = unit(np.mean(crops, axis=0)) if crops else None
            ents = [f"{v['cls']}_{t}" for t, v in kept.items()]
            clip_id = clip.stem
            base = {"clip_path": str(clip), "camera_id": video.stem, "timestamp": ci * CLIP_S, "zone": None,
                    "tracked_entities": ents, "lighting": "night" if night else "day", "brightness": round(bright, 1)}
            pid = lambda typ: int(hashlib.md5(f"{clip_id}{typ}".encode()).hexdigest()[:12], 16)
            points.append(PointStruct(id=pid("frame"), vector=fe.tolist(), payload={**base, "embedding_type": "frame"}))
            if ce is not None:
                points.append(PointStruct(id=pid("crop"), vector=ce.tolist(), payload={**base, "embedding_type": "crop"}))
            for t, v in kept.items():
                all_tracks[(ci, t)] = v["cls"]
            n_clips += 1
            print(f"  {clip_id}: {'night' if night else 'day'} (brightness {bright:.0f}), {len(kept)} tracked objects")
        by_cls = {}
        for c in all_tracks.values():
            by_cls[c] = by_cls.get(c, 0) + 1
        stats_md.append(f"- **{video.name}**: {n_clips} clips ({n_night} night), {len(all_tracks)} tracked objects "
                        f"(per-clip tracks), by class: {by_cls}")

    if client is not None and points:
        client.upsert(COLLECTION, points)
    (FIG_DIR / "indian_video_ingest.md").write_text("\n".join(stats_md) + "\n")
    print("\n".join(stats_md))
    print(f"\nindexed {len(points)} vectors into '{COLLECTION}'" if client is not None else "\n(qdrant skipped)")


if __name__ == "__main__":
    main()
