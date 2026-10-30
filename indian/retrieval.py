"""Indian-domain retrieval: index UVH-26 test images (frame + crop CLIP embeddings), frame-vs-crop ablation.
  python -m indian.retrieval build --weights runs/indian/v1/weights/best.pt --n 400 --qdrant
  python -m indian.retrieval eval
"""
import argparse, json

import numpy as np
from PIL import Image

from indian.common import YOLO_DIR, FIG_DIR, EVAL_DIR, CLASSES

NPZ = EVAL_DIR / "indian_index.npz"
META = EVAL_DIR / "indian_index_meta.json"
COLLECTION = "uvh26_items"


def gt_counts(split, stem):
    lab = YOLO_DIR / "labels" / split / f"{stem}.txt"
    c = [0] * len(CLASSES)
    for ln in lab.read_text().split("\n") if lab.exists() else []:
        if ln.strip():
            c[int(ln.split()[0])] += 1
    return c


def build(args):
    from ultralytics import YOLO
    from src.embeddings.clip_encoder import embed_image
    model = YOLO(args.weights)
    paths = sorted((YOLO_DIR / "images" / "test").glob("*.jpg"))[: args.n]
    frame_e, crop_e, meta = [], [], []
    for k, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        r = model.predict(str(p), imgsz=960, conf=0.3, verbose=False, device=0)[0]
        boxes = sorted(zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist()),
                       key=lambda t: -(t[0][2] - t[0][0]) * (t[0][3] - t[0][1]))[:25]
        crops = []
        for (x1, y1, x2, y2), c in boxes:
            if x2 - x1 >= 16 and y2 - y1 >= 16:
                crops.append(embed_image(img.crop((int(x1), int(y1), int(x2), int(y2)))))
        fe = np.array(embed_image(img), dtype="float32")
        if crops:
            ce = np.mean(np.array(crops, dtype="float32"), axis=0)
            ce /= np.linalg.norm(ce)
        else:
            ce = np.zeros_like(fe)
        det_counts = [0] * len(CLASSES)
        for _, c in boxes:
            det_counts[int(c)] += 1
        frame_e.append(fe); crop_e.append(ce)
        meta.append({"path": str(p), "stem": p.stem, "det_counts": det_counts,
                     "gt_counts": gt_counts("test", p.stem)})
        if (k + 1) % 50 == 0:
            print(f"  indexed {k+1}/{len(paths)}")
    EVAL_DIR.mkdir(exist_ok=True)
    np.savez(NPZ, frame=np.stack(frame_e), crop=np.stack(crop_e))
    META.write_text(json.dumps(meta))
    print("saved", NPZ, META)

    if args.qdrant:
        from qdrant_client.models import VectorParams, Distance, PointStruct
        from src.storage.qdrant_client import get_client
        cl = get_client()
        if not cl.collection_exists(COLLECTION):
            cl.create_collection(COLLECTION, vectors_config=VectorParams(size=512, distance=Distance.COSINE))
        pts = []
        for i, m in enumerate(meta):
            ents = [f"{CLASSES[c]}_{j}" for c, n in enumerate(m["det_counts"]) for j in range(n)]
            base = {"clip_path": m["path"], "camera_id": "uvh26", "timestamp": 0, "zone": None,
                    "tracked_entities": ents, "det_counts": dict(zip(CLASSES, m["det_counts"]))}
            pts.append(PointStruct(id=2 * i, vector=frame_e[i].tolist(), payload={**base, "embedding_type": "frame"}))
            if np.any(crop_e[i]):
                pts.append(PointStruct(id=2 * i + 1, vector=crop_e[i].tolist(), payload={**base, "embedding_type": "crop"}))
        cl.upsert(COLLECTION, pts)
        print(f"upserted {len(pts)} points into Qdrant collection '{COLLECTION}'")


def top_frac(v, frac=0.10, min_count=1):
    v = np.asarray(v)
    k = max(1, int(round(frac * len(v))))
    thr = max(min_count, np.sort(v)[::-1][k - 1])
    return set(np.where(v >= thr)[0].tolist())


def make_queries(meta):
    g = np.array([m["gt_counts"] for m in meta])
    total = g.sum(1)
    ci = {c: i for i, c in enumerate(CLASSES)}
    sparse_thr = np.sort(total)[max(0, int(0.10 * len(total)) - 1)]
    return [
        ("many motorcycles and scooters on the road", top_frac(g[:, ci["motorcycle"]], min_count=3)),
        ("auto rickshaws on the road", top_frac(g[:, ci["auto_rickshaw"]], min_count=2)),
        ("a bus on the road", top_frac(g[:, ci["bus"]], min_count=1)),
        ("trucks on the road", top_frac(g[:, ci["truck"]], min_count=1)),
        ("bicycles on the road", top_frac(g[:, ci["bicycle"]], min_count=1)),
        ("many cars on the road", top_frac(g[:, ci["car"]], min_count=4)),
        ("heavy traffic with many vehicles", top_frac(total, min_count=8)),
        ("an almost empty road with hardly any vehicles", set(np.where(total <= min(sparse_thr, 1))[0].tolist())),
    ]


def evaluate(args):
    from src.embeddings.clip_encoder import embed_text, embed_text_ensemble
    meta = json.load(open(META))
    z = np.load(NPZ)
    N = len(meta)
    queries = make_queries(meta)
    Ks = [1, 3, 5, 10]
    rows = []
    for etype in ("frame", "crop", "both"):
        E = z["frame"] if etype == "both" else z[etype]
        valid = np.linalg.norm(E, axis=1) > 0
        Ec = z["crop"]
        valid_c = np.linalg.norm(Ec, axis=1) > 0
        for ens in (False, True):
            P = {k: [] for k in Ks}; RR = []
            for q, rel in queries:
                rel_v = {i for i in rel if valid[i]}
                if not rel_v:
                    continue
                t = np.array(embed_text_ensemble(q) if ens else embed_text(q), dtype="float32")
                if etype == "both":
                    sims = np.where(valid, 0.7 * (E @ t) + 0.3 * np.where(valid_c, Ec @ t, E @ t), -9)
                else:
                    sims = np.where(valid, E @ t, -9)
                order = np.argsort(-sims)
                hits = [int(i in rel_v) for i in order]
                for k in Ks:
                    P[k].append(sum(hits[:k]) / k)
                RR.append(1.0 / (hits.index(1) + 1) if 1 in hits else 0.0)
            rows.append((etype, ens, {k: float(np.mean(P[k])) for k in Ks}, float(np.mean(RR))))
    prev = float(np.mean([len(r) / N for _, r in queries if r]))
    md = [f"### Indian-domain retrieval ablation (UVH-26 test images, N={N}, {len(queries)} queries)", "",
          "| embedding | prompt ensemble | " + " | ".join(f"P@{k}" for k in Ks) + " | MRR |",
          "|---|---|" + "---|" * (len(Ks) + 1)]
    for e, ens, P, mrr in rows:
        md.append(f"| {'frame+crop (combined)' if e == 'both' else e} | {'yes' if ens else 'no'} | " + " | ".join(f"{P[k]:.3f}" for k in Ks) + f" | {mrr:.3f} |")
    md += ["", f"Random-ranking baseline (mean fraction of relevant images): {prev:.3f}", "",
           "Queries and number of relevant images (from annotation counts):", ""]
    md += [f"- \"{q}\": {len(r)} relevant" for q, r in queries]
    out = "\n".join(md)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "indian_retrieval_ablation.md").write_text(out)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--weights", required=True)
    b.add_argument("--n", type=int, default=400)
    b.add_argument("--qdrant", action="store_true")
    sub.add_parser("eval")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else evaluate(a)
