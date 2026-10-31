"""Background worker: processes uploaded videos one at a time (the GPU can only do one job comfortably)."""
import queue
import threading
import traceback

from src.library import store

ROOT = store.ROOT
WEIGHTS = ROOT / "models" / "indian_best.pt"
_jobs = queue.Queue()
_thread = None


def start():
    """Start the worker once, and pick up any job that was interrupted by a restart."""
    global _thread
    if _thread is not None:
        return
    _thread = threading.Thread(target=_loop, name="library-worker", daemon=True)
    _thread.start()
    for it in store.list_items():
        if it["status"] in ("queued", "processing") and store.original_path(it["id"]):
            enqueue(it["id"])
        elif it["status"] == "uploading":
            store.write_meta(it["id"], status="failed", error="The upload was interrupted. Upload the file again.")


def enqueue(vid):
    store.write_meta(vid, status="queued", stage="Waiting to start", error=None)
    _jobs.put(vid)


def _loop():
    while True:
        vid = _jobs.get()
        try:
            _process(vid)
        except Exception as e:                                   # keep the worker alive whatever happens
            traceback.print_exc()
            store.write_meta(vid, status="failed", stage=None, error=f"{type(e).__name__}: {str(e)[:300]}")


def _process(vid):
    from indian.ingest_video import process_video
    from src.storage.qdrant_client import get_client
    src = store.original_path(vid)
    if src is None:
        raise FileNotFoundError("The uploaded file is missing.")
    if not WEIGHTS.exists():
        raise FileNotFoundError("models/indian_best.pt is missing, so vehicles cannot be detected.")
    store.write_meta(vid, status="processing", stage="Starting", done=0, total=0, error=None)

    def progress(stage, done, total):
        store.write_meta(vid, stage=stage, done=done, total=total)

    stats = process_video(src, WEIGHTS, get_client(), progress=progress)
    if stats["clips"] == 0:
        raise ValueError("No usable clips. The video may be shorter than 2 seconds or unreadable.")
    store.write_meta(vid, status="ready", stage=None, done=stats["clips"], total=stats["clips"], stats=stats)
