"""Search + keyframe helpers shared by the API, dashboard and eval scripts.

Hybrid search:
  1. query understanding: pulls structured conditions out of the text ('more than 3 trucks', 'no buses', 'at night',
     'empty road', 'heavy traffic');
  2. CLIP text->vector similarity (frame, crop, or a combination of both) for the visual part;
  3. hard filters on detected counts / lighting, then re-ranking by how well the counts fit the query.
If the hard filters match nothing, they are relaxed and the user is told."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.embeddings.clip_encoder import embed_text, embed_text_ensemble
from src.storage.qdrant_client import get_client, COLLECTION_NAME

COLLECTIONS = {"detrac": COLLECTION_NAME, "indian": "uvh26_items", "indian_video": "indian_clips"}
FRAME_W, CROP_W = 0.7, 0.3

EMPTY_WORDS = ["empty", "hardly any", "few vehicles", "no traffic", "clear road", "light traffic", "quiet",
               "deserted", "no vehicles", "free flowing", "free-flowing", "sparse"]
HEAVY_WORDS = ["heavy", "crowded", "congest", "many vehicles", "busy", "jam", "dense", "packed", "lots of vehicles",
               "rush hour", "bumper to bumper"]
NIGHT_WORDS = ["night", "dark", "evening", "after dark"]
DAY_WORDS = ["daytime", "sunny", "daylight", "bright", "during the day", "in the day"]
CLASS_WORDS = {
    "bus": ["buses", "bus"], "truck": ["trucks", "truck", "lorries", "lorry", "lcv", "tanker", "pickup"],
    "car": ["cars", "car", "sedan", "sedans", "suv", "suvs", "hatchback", "hatchbacks"],
    "auto_rickshaw": ["auto-rickshaws", "auto-rickshaw", "auto rickshaws", "auto rickshaw", "rickshaws", "rickshaw",
                      "autos", "auto", "three-wheelers", "three-wheeler", "three wheelers", "three wheeler", "tuk-tuk", "tuk tuk"],
    "motorcycle": ["motorcycles", "motorcycle", "scooters", "scooter", "motorbikes", "motorbike", "bikes", "bike",
                   "two-wheelers", "two-wheeler", "two wheelers", "two wheeler"],
    "bicycle": ["bicycles", "bicycle", "cycles", "cycle", "cyclists", "cyclist"],
    "person": ["people", "person", "pedestrians", "pedestrian"],
}
NUMWORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
QUANT = {"many": 4, "several": 3, "lots of": 5, "a lot of": 5, "a few": 2, "couple of": 2, "a couple of": 2, "multiple": 2}


def entity_counts(payload):
    out = {}
    for e in payload.get("tracked_entities") or []:
        cls = str(e).rsplit("_", 1)[0]
        out[cls] = out.get(cls, 0) + 1
    return out


def clip_id_of(path):
    return Path(path).stem if path else "unknown"


def _has(ql, words):
    return any(re.search(rf"\b{re.escape(w)}\b", ql) for w in words)


def parse_intent(query):
    """-> {empty, heavy, night, day, classes: {cls: {'min': n, 'exclude': bool}}}"""
    ql = " " + query.lower() + " "
    classes = {}
    nums = "|".join(NUMWORDS)
    quant = "|".join(re.escape(k) for k in sorted(QUANT, key=len, reverse=True))
    for cls, words in CLASS_WORDS.items():
        alt = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
        if re.search(rf"\b(?:no|without|zero|not any)\s+(?:\w+\s+)?(?:{alt})\b", ql):
            classes[cls] = {"min": 0, "exclude": True}
            continue
        m = re.search(rf"(?:(more than|over|at least|atleast|minimum of|min|>=|>)\s*)?\b(\d+|{nums})\s+(?:\w+\s+)?(?:{alt})\b", ql)
        if m:
            n = int(m.group(2)) if m.group(2).isdigit() else NUMWORDS[m.group(2)]
            if m.group(1) in ("more than", "over", ">"):
                n += 1
            classes[cls] = {"min": n, "exclude": False}
            continue
        m = re.search(rf"\b({quant})\s+(?:\w+\s+)?(?:{alt})\b", ql)
        if m:
            classes[cls] = {"min": QUANT[m.group(1)], "exclude": False}
            continue
        if re.search(rf"\b(?:{alt})\b", ql):
            classes[cls] = {"min": 1, "exclude": False}
    return {"empty": _has(ql, EMPTY_WORDS), "heavy": _has(ql, HEAVY_WORDS), "night": _has(ql, NIGHT_WORDS),
            "day": _has(ql, DAY_WORDS) and not _has(ql, NIGHT_WORDS), "classes": classes}


def describe_intent(intent):
    parts = []
    for c, r in intent["classes"].items():
        name = c.replace("_", " ")
        parts.append(f"no {name}" if r["exclude"] else (f"at least {r['min']} {name}" if r["min"] > 1 else f"{name} present"))
    if intent["night"]:
        parts.append("night")
    if intent["day"]:
        parts.append("daytime")
    if intent["empty"]:
        parts.append("few vehicles / empty road")
    if intent["heavy"]:
        parts.append("heavy traffic")
    return ", ".join(parts) if parts else "visual description only"


def _passes(h, intent):
    for c, r in intent["classes"].items():
        n = h["counts"].get(c, 0)
        if r["exclude"] and n > 0:
            return False
        if not r["exclude"] and n < r["min"]:
            return False
    lt = h.get("lighting")
    if lt:
        if intent["night"] and lt != "night":
            return False
        if intent["day"] and lt != "day":
            return False
    return True


def _meta_score(h, intent, max_total):
    parts = []
    total = sum(h["counts"].values())
    if intent["empty"]:
        parts.append(1.0 / (1.0 + total / 2.0))
    if intent["heavy"]:
        parts.append(total / max_total if max_total else 0.0)
    for c, r in intent["classes"].items():
        if not r["exclude"]:
            parts.append(min(h["counts"].get(c, 0) / max(3.0, r["min"]), 1.0))
    if intent["night"]:
        parts.append({"night": 1.0, "day": 0.0}.get(h.get("lighting"), 0.5))
    if intent["day"]:
        parts.append({"day": 1.0, "night": 0.0}.get(h.get("lighting"), 0.5))
    return sum(parts) / len(parts) if parts else None


def list_cameras(dataset="detrac"):
    client = get_client()
    cams, offset = set(), None
    while True:
        pts, offset = client.scroll(COLLECTIONS[dataset], limit=256, offset=offset, with_payload=["camera_id"], with_vectors=False)
        cams |= {p.payload.get("camera_id") for p in pts if p.payload.get("camera_id")}
        if offset is None:
            break
    return sorted(cams)


def _query(client, dataset, vec, etype, camera_id, zone, limit):
    must = [FieldCondition(key="embedding_type", match=MatchValue(value=etype))]
    if camera_id:
        must.append(FieldCondition(key="camera_id", match=MatchValue(value=camera_id)))
    if zone:
        must.append(FieldCondition(key="zone", match=MatchValue(value=zone)))
    return client.query_points(collection_name=COLLECTIONS[dataset], query=vec, query_filter=Filter(must=must),
                               limit=limit, with_payload=True).points


def _hit(p, score):
    pl = p.payload
    return {"clip_id": clip_id_of(pl.get("clip_path")), "clip_path": pl.get("clip_path"),
            "camera_id": pl.get("camera_id"), "timestamp": pl.get("timestamp"), "zone": pl.get("zone"),
            "lighting": pl.get("lighting"), "counts": entity_counts(pl), "clip_score": float(score)}


def search_ex(query, dataset="detrac", embedding_type="frame", camera_id=None, zone=None,
              top_k=5, use_ensemble=True, rerank=True):
    """embedding_type: 'frame', 'crop' or 'both'.  Returns {'hits', 'understood', 'relaxed'}."""
    client = get_client()
    vec = embed_text_ensemble(query) if use_ensemble else embed_text(query)
    limit = 300 if rerank else top_k
    if embedding_type == "both":
        f = {p.payload.get("clip_path"): p for p in _query(client, dataset, vec, "frame", camera_id, zone, limit)}
        c = {p.payload.get("clip_path"): p for p in _query(client, dataset, vec, "crop", camera_id, zone, limit)}
        hits = []
        for path, pf in f.items():
            sc = FRAME_W * pf.score + CROP_W * (c[path].score if path in c else pf.score)
            hits.append(_hit(pf, sc))
    else:
        hits = [_hit(p, p.score) for p in _query(client, dataset, vec, embedding_type, camera_id, zone, limit)]
    intent = parse_intent(query) if rerank else parse_intent("")
    relaxed = False
    if rerank:
        kept = [h for h in hits if _passes(h, intent)]
        if kept or not hits:
            hits = kept
        else:
            relaxed = True
    if hits:
        lo, hi = min(h["clip_score"] for h in hits), max(h["clip_score"] for h in hits)
        max_total = max(sum(h["counts"].values()) for h in hits)
        for h in hits:
            cs = (h["clip_score"] - lo) / (hi - lo) if hi > lo else 1.0
            m = _meta_score(h, intent, max_total) if rerank else None
            h["score"] = round(cs if m is None else 0.25 * cs + 0.75 * m, 4)
        hits.sort(key=lambda h: -h["score"])
    return {"hits": hits[:top_k], "understood": describe_intent(intent) if rerank else "pure visual similarity",
            "relaxed": relaxed}


def search(query, dataset="detrac", embedding_type="frame", camera_id=None, zone=None,
           top_k=5, use_ensemble=True, rerank=True):
    return search_ex(query, dataset, embedding_type, camera_id, zone, top_k, use_ensemble, rerank)["hits"]


def keyframe_jpeg(path, ts=None):
    """Return JPEG bytes: the image itself, or the middle frame of a video clip."""
    import cv2
    p = str(path)
    if p.lower().endswith((".jpg", ".jpeg", ".png")):
        img = cv2.imread(p)
    else:
        cap = cv2.VideoCapture(p)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
        ok, img = cap.read()
        cap.release()
        if not ok:
            img = None
    if img is None:
        raise FileNotFoundError(f"cannot read frame from {p}")
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()
