"""Video library bookkeeping: what is in the library and its processing status.

Layout on disk (all under data/, which is git-ignored):
  data/indian_videos/<id>.<ext>   original video (uploads are saved here too)
  data/indian_clips/<id>/*.mp4    3-second clips (what search plays)
  data/library/<id>.json          status of videos added through the upload page
Videos that only have clips (the sample set) show up as ready automatically."""
import json
import re
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
VID_DIR = ROOT / "data" / "indian_videos"
CLIP_DIR = ROOT / "data" / "indian_clips"
META_DIR = ROOT / "data" / "library"
ALLOWED = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def valid_id(vid):
    return bool(ID_RE.match(vid or "")) and ".." not in vid


def make_id(filename):
    base = re.sub(r"[^A-Za-z0-9]+", "_", Path(filename).stem).strip("_").lower()[:40] or "video"
    vid, n = base, 2
    while exists(vid):
        vid, n = f"{base}_{n}", n + 1
    return vid


def original_path(vid):
    for p in VID_DIR.glob(f"{vid}.*"):
        if p.suffix.lower() in ALLOWED:
            return p
    return None


def exists(vid):
    return (CLIP_DIR / vid).exists() or (META_DIR / f"{vid}.json").exists() or original_path(vid) is not None


def read_meta(vid):
    try:
        return json.loads((META_DIR / f"{vid}.json").read_text())
    except (OSError, ValueError):
        return None


def write_meta(vid, **fields):
    META_DIR.mkdir(parents=True, exist_ok=True)
    meta = {**(read_meta(vid) or {}), **fields}
    tmp = META_DIR / f"{vid}.json.tmp"
    tmp.write_text(json.dumps(meta))
    tmp.replace(META_DIR / f"{vid}.json")
    return meta


def item(vid):
    meta = read_meta(vid) or {}
    clips = sorted((CLIP_DIR / vid).glob("*.mp4")) if (CLIP_DIR / vid).exists() else []
    src = original_path(vid)
    indexed = (meta.get("stats") or {}).get("clips") or len(clips)
    return {
        "id": vid,
        "name": meta.get("name") or vid,
        "source": "upload" if meta else "sample",
        "status": meta.get("status", "ready" if clips else "empty"),
        "stage": meta.get("stage"), "done": meta.get("done", 0), "total": meta.get("total", 0),
        "error": meta.get("error"),
        "clips": indexed, "seconds": indexed * 3,
        "size_mb": round(src.stat().st_size / 1e6, 1) if src else None,
        "added": meta.get("added"),
        "thumb": str(clips[0]) if clips else None,
    }


def list_items():
    ids = {p.name for p in CLIP_DIR.glob("*") if p.is_dir()} if CLIP_DIR.exists() else set()
    if META_DIR.exists():
        ids |= {p.stem for p in META_DIR.glob("*.json")}
    items = [item(v) for v in ids if valid_id(v)]
    items = [i for i in items if i["status"] != "empty"]
    return sorted(items, key=lambda i: (i["added"] or 0, i["id"]), reverse=True)


def new_upload(vid, display_name):
    return write_meta(vid, name=display_name, status="queued", stage="Waiting to start", done=0, total=0,
                      error=None, added=time.time())


def delete(vid, client=None):
    """Remove a video completely: search index entries, clips, original and status file."""
    if client is not None:
        from indian.ingest_video import remove_video_points
        remove_video_points(client, vid)
    shutil.rmtree(CLIP_DIR / vid, ignore_errors=True)
    for p in VID_DIR.glob(f"{vid}.*"):
        p.unlink(missing_ok=True)
    (META_DIR / f"{vid}.json").unlink(missing_ok=True)
