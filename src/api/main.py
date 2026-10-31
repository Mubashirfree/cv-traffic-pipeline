"""Traffic video search API + web frontend.   uvicorn src.api.main:app --port 8000   then open http://localhost:8000"""
import json
import re
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.retrieval import service

app = FastAPI(title="Traffic Video Search")
DATA_DIR = (ROOT / "data").resolve()
FIG_DIR = ROOT / "report" / "figures"
EVAL_DIR = ROOT / "eval"


class SearchReq(BaseModel):
    query: str
    dataset: str = "indian_video"        # "indian_video", "indian" (UVH-26 images) or "detrac"
    embedding_type: str = "both"         # "frame", "crop" or "both"
    camera_id: Optional[str] = None
    zone: Optional[str] = None
    top_k: int = 5
    rerank: bool = True                  # False = pure CLIP similarity


class AnalyzeReq(SearchReq):
    max_clips: int = 3


DIST = ROOT / "frontend" / "dist"          # built React app (npm run build in frontend/)
if (DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/")
def home():
    built = DIST / "index.html"
    if built.exists():
        return FileResponse(built)
    return FileResponse(ROOT / "web" / "index.html")   # plain-HTML fallback if the React app is not built


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchReq):
    if not req.query.strip():
        raise HTTPException(400, "Type a description first.")
    if req.dataset not in service.COLLECTIONS or req.embedding_type not in ("frame", "crop", "both"):
        raise HTTPException(400, "Unknown dataset or search mode.")
    try:
        info = service.search_ex(req.query, req.dataset, req.embedding_type, req.camera_id, req.zone,
                                 max(1, min(req.top_k, 20)), rerank=req.rerank)
    except Exception as e:
        raise HTTPException(503, f"Search failed. Is Docker (Qdrant) running and the dataset indexed? Details: {e}")
    return {"query": req.query, "n": len(info["hits"]), "results": info["hits"],
            "understood": info["understood"], "relaxed": info["relaxed"]}


@app.post("/analyze")
def analyze(req: AnalyzeReq):
    from src.rag.gemini_summarizer import summarize
    from src.rag.check import check
    hits = search(req)["results"]
    if not hits:
        return {"summary": "No matching clips found for this query.", "flags": [], "results": []}
    try:
        out = summarize(req.query, hits, req.max_clips)
    except Exception as e:
        raise HTTPException(502, f"The Gemini request failed. Check GEMINI_API_KEY in .env and try again. Details: {e}")
    out["flags"] = check(out["summary"], hits[: req.max_clips])
    out["results"] = hits
    return out


@app.get("/cameras")
def cameras(dataset: str = "indian_video"):
    if dataset not in service.COLLECTIONS:
        raise HTTPException(400, "Unknown dataset.")
    try:
        return {"cameras": service.list_cameras(dataset)}
    except Exception as e:
        raise HTTPException(503, f"Could not list cameras: {e}")


@app.get("/keyframe")
def keyframe(path: str):
    p = Path(path).resolve()
    if ROOT.resolve() not in p.parents:
        raise HTTPException(403, "Path is outside the project.")
    try:
        return Response(service.keyframe_jpeg(p), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(404, str(e))


@app.get("/media")
def media(path: str):
    """Serve a clip from data/ (supports range requests, so the video can seek)."""
    p = Path(path).resolve()
    if DATA_DIR not in p.parents or p.suffix.lower() not in (".mp4", ".webm", ".jpg", ".jpeg", ".png"):
        raise HTTPException(403, "Not an allowed media file.")
    if not p.exists():
        raise HTTPException(404, "File not found.")
    return FileResponse(p)


def _md_table(text):
    rows = []
    for ln in text.splitlines():
        if ln.startswith("|") and not re.match(r"^\|[-| ]+\|$", ln.strip()):
            rows.append([c.strip() for c in ln.strip().strip("|").split("|")])
    return rows


@app.get("/metrics")
def metrics():
    """Measured results for the Evaluation tab (read from the result files the eval scripts wrote)."""
    out = {"detection": [], "retrieval": None, "faithfulness": None, "index": {}}
    for label in ("coco_baseline", "detrac_ft", "indian_ft"):
        f = EVAL_DIR / f"indian_det_{label}.json"
        if f.exists():
            out["detection"].append(json.loads(f.read_text()))
    f = FIG_DIR / "indian_retrieval_ablation.md"
    if f.exists():
        t = f.read_text()
        out["retrieval"] = {"rows": _md_table(t), "text": t}
    f = FIG_DIR / "rag_faithfulness.md"
    if f.exists():
        out["faithfulness"] = f.read_text()
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        client = service.get_client()
        for ds, col in service.COLLECTIONS.items():
            out["index"][ds] = client.count(col, count_filter=Filter(must=[FieldCondition(
                key="embedding_type", match=MatchValue(value="frame"))])).count
    except Exception:
        pass
    return out
