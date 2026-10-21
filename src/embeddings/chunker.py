"""
Phase 3: Segment each source video into 3-second sub-clips.
Produces a manifest mapping each sub-clip to its camera/time range.
"""

import json
import subprocess
from pathlib import Path

CLIP_LEN = 3.0  # seconds


def get_duration(video_path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def chunk_video(video_path: str, camera_id: str, out_dir: str = "data/clips"):
    out_dir = Path(out_dir) / camera_id
    out_dir.mkdir(parents=True, exist_ok=True)

    duration = get_duration(video_path)
    n_chunks = int(duration // CLIP_LEN)
    manifest = []

    for i in range(n_chunks):
        start = i * CLIP_LEN
        end = start + CLIP_LEN
        clip_out = out_dir / f"{camera_id}_{i:03d}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-t", str(CLIP_LEN),
             "-i", video_path, "-c", "copy", str(clip_out)],
            capture_output=True,
        )
        manifest.append({
            "clip_id": f"{camera_id}_{i:03d}",
            "camera_id": camera_id,
            "start_ts": start,
            "end_ts": end,
            "path": str(clip_out),
        })

    return manifest


if __name__ == "__main__":
    videos_dir = Path("data/videos")
    full_manifest = []

    for video in sorted(videos_dir.glob("*.mp4")):
        camera_id = video.stem
        clips = chunk_video(str(video), camera_id)
        full_manifest.extend(clips)
        print(f"[{camera_id}] {len(clips)} sub-clips")

    with open("data/clips/manifest.json", "w") as f:
        json.dump(full_manifest, f, indent=2)
    print(f"\nTotal: {len(full_manifest)} sub-clips -> data/clips/manifest.json")
