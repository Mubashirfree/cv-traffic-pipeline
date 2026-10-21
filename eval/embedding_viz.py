"""
eval/embedding_viz.py -- Phase 5: does CLIP separate categories meaningfully?

Part A  Per-object crop embeddings (from our detections) -> t-SNE coloured by detected class,
        plus silhouette score and 5-fold k-NN class accuracy in the ORIGINAL 512-d space
        (the plot is only a picture; these numbers are the evidence).
Part B  Clip-level embeddings from Qdrant (frame vs crop) -> t-SNE coloured by camera,
        plus leave-one-out k-NN camera accuracy. Shows how much scene information each
        embedding type carries (relevant to the frame-vs-crop retrieval ablation).

Run from the project root:
    python -m eval.embedding_viz
    python -m eval.embedding_viz --skip-qdrant      # Part A only

Outputs (report/figures/):
    embedding_tsne_classes.png, embedding_tsne_clips.png, embedding_viz_metrics.txt
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DET_DIR = ROOT / "data" / "detections"
RAW_DIR = ROOT / "data" / "raw"
FIG_DIR = ROOT / "report" / "figures"

VID_STRIDE = 10          # detections were sampled every 10th source frame (25 fps -> 2.5 fps)
MIN_CONF = 0.5           # cleaner class labels
MIN_SIDE = 24            # px; skip tiny crops that CLIP cannot say much about
MAX_PER_CLASS = 300
MIN_PER_CLASS = 20


# --------------------------------------------------------------------------- #
# Part A: per-object crops
# --------------------------------------------------------------------------- #
def collect_crop_specs(seed=0):
    """Sample up to MAX_PER_CLASS confident detections per class -> [(cls, camera, frame_num, bbox)]."""
    by_class = defaultdict(list)
    for p in sorted(DET_DIR.glob("*.json")):
        log = json.load(open(p))
        if not isinstance(log, list):
            continue
        for idx, entry in enumerate(log):
            # array index -> source frame (frame_ts is truncated to whole seconds; do not parse it)
            frame_num = idx * VID_STRIDE + 1
            for d in entry.get("detections", []):
                x1, y1, x2, y2 = d["bbox_px"]
                if d["confidence"] < MIN_CONF or (x2 - x1) < MIN_SIDE or (y2 - y1) < MIN_SIDE:
                    continue
                by_class[d["class"]].append((d["class"], entry["camera_id"], frame_num, d["bbox_px"]))
    rng, specs = random.Random(seed), []
    for cls, items in sorted(by_class.items()):
        if len(items) < MIN_PER_CLASS:
            print(f"[skip] class '{cls}' has only {len(items)} confident crops (< {MIN_PER_CLASS})")
            continue
        rng.shuffle(items)
        specs += items[:MAX_PER_CLASS]
    return specs


def embed_specs(specs, embed_fn):
    from PIL import Image
    specs = sorted(specs, key=lambda s: (s[1], s[2]))          # group by image, open each once
    X, y, cur_key, img = [], [], None, None
    for cls, cam, fn, bbox in specs:
        if (cam, fn) != cur_key:
            img = Image.open(RAW_DIR / cam / f"img{fn:05d}.jpg").convert("RGB")
            cur_key = (cam, fn)
        w, h = img.size
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        crop = img.crop((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
        X.append(embed_fn(crop))
        y.append(cls)
    return np.array(X, dtype=np.float32), np.array(y)


def class_metrics(X, y):
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    counts = Counter(str(c) for c in y)
    n_splits = min(5, min(counts.values()))
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=0)
    knn = KNeighborsClassifier(5, metric="cosine")
    return {
        "n_crops": len(y),
        "class_counts": dict(counts),
        "silhouette_cosine": float(silhouette_score(X, y, metric="cosine")),
        "knn5_accuracy": float(cross_val_score(knn, X, y, cv=cv).mean()),
        "knn5_balanced_accuracy": float(cross_val_score(knn, X, y, cv=cv, scoring="balanced_accuracy").mean()),
        "chance_balanced_accuracy": 1.0 / len(counts),
        "majority_class_accuracy": max(counts.values()) / len(y),
    }


def tsne_2d(X, seed=0):
    from sklearn.manifold import TSNE
    perplexity = min(30, max(5, (len(X) - 1) // 4))
    return TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
                metric="cosine", random_state=seed).fit_transform(X)


def plot_classes(Z, y, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for cls in sorted(set(y)):
        m = y == cls
        ax.scatter(Z[m, 0], Z[m, 1], s=10, alpha=0.7, label=f"{cls} (n={m.sum()})")
    ax.set_title("t-SNE of CLIP embeddings of detected-object crops")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=2)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# --------------------------------------------------------------------------- #
# Part B: clip-level embeddings from Qdrant
# --------------------------------------------------------------------------- #
def load_clip_vectors():
    from qdrant_client.models import FieldCondition, Filter, MatchValue
    from src.storage.qdrant_client import COLLECTION_NAME, get_client
    client = get_client()
    out = {}
    for etype in ("frame", "crop"):
        pts, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[FieldCondition(key="embedding_type", match=MatchValue(value=etype))]),
            limit=200, with_payload=True, with_vectors=True)
        pts = sorted(pts, key=lambda p: Path(p.payload["clip_path"]).stem)
        out[etype] = (np.array([p.vector for p in pts], dtype=np.float32),
                      [p.payload["camera_id"] for p in pts])
    return out


def camera_loo_accuracy(X, cams, k=3):
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.neighbors import KNeighborsClassifier
    pred = cross_val_predict(KNeighborsClassifier(k, metric="cosine"), X, cams, cv=LeaveOneOut())
    return float(np.mean(np.array(pred) == np.array(cams)))


def plot_clips(vecs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, etype in zip(axes, ("frame", "crop")):
        X, cams = vecs[etype]
        Z = tsne_2d(X)
        cams = np.array(cams)
        for cam in sorted(set(cams)):
            m = cams == cam
            ax.scatter(Z[m, 0], Z[m, 1], s=40, marker="*" if cam == "MVI_40992" else "o",
                       label=cam + (" (night)" if cam == "MVI_40992" else ""))
        ax.set_title(f"{etype} embeddings ({len(cams)} clips)")
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(fontsize=8)
    fig.suptitle("t-SNE of clip-level CLIP embeddings, coloured by camera")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-qdrant", action="store_true")
    args = ap.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = []

    from src.embeddings.clip_encoder import embed_image
    specs = collect_crop_specs()
    print(f"Embedding {len(specs)} crops ...")
    X, y = embed_specs(specs, embed_image)
    m = class_metrics(X, y)
    plot_classes(tsne_2d(X), y, FIG_DIR / "embedding_tsne_classes.png")
    lines += ["PART A: per-object crop embeddings (labels = detector class, conf >= %.1f)" % MIN_CONF,
              f"crops: {m['n_crops']}  counts: {m['class_counts']}",
              f"silhouette (cosine, 512-d): {m['silhouette_cosine']:.3f}",
              f"5-NN accuracy: {m['knn5_accuracy']:.3f}   (majority-class baseline {m['majority_class_accuracy']:.3f})",
              f"5-NN balanced accuracy: {m['knn5_balanced_accuracy']:.3f}   (chance {m['chance_balanced_accuracy']:.3f})",
              "NOTE: labels come from YOLO, which misclassifies bus/truck/van, so this is a lower bound.", ""]

    if not args.skip_qdrant:
        vecs = load_clip_vectors()
        plot_clips(vecs, FIG_DIR / "embedding_tsne_clips.png")
        lines.append("PART B: clip-level embeddings, leave-one-out 3-NN camera-ID accuracy")
        for etype in ("frame", "crop"):
            Xc, cams = vecs[etype]
            lines.append(f"  {etype}: {camera_loo_accuracy(Xc, cams):.3f}   "
                         f"(majority-camera baseline {Counter(cams).most_common(1)[0][1] / len(cams):.3f})")

    text = "\n".join(lines)
    (FIG_DIR / "embedding_viz_metrics.txt").write_text(text)
    print("\n" + text + f"\n\nSaved figures + metrics to {FIG_DIR}")


if __name__ == "__main__":
    main()
