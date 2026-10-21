"""
app/demo.py -- Demo frontend for the guide presentation.

Three tabs, all backed by the real pipeline (nothing here is mocked or precomputed for show):
  1. Search      -- live call to src.retrieval.hybrid_search.search()
  2. Results     -- reads the JSON/MD files already produced by eval_fair.py,
                    compare_before_after.py, occlusion_check.py, benchmark.py
  3. Detection   -- runs the fine-tuned YOLO weights on a chosen frame, live, and draws boxes

Run from the project root (Docker containers must be up for the Search tab):
    pip install streamlit --break-system-packages   # one-time
    streamlit run app/demo.py
"""
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVAL_SEQS = ["MVI_20011", "MVI_20012", "MVI_20032", "MVI_20033", "MVI_20034", "MVI_40992"]
FINETUNED_WEIGHTS = ROOT / "runs" / "detect" / "runs" / "finetune" / "v1" / "weights" / "best.pt"

st.set_page_config(page_title="Traffic CV Pipeline Demo", layout="wide")
st.title("Traffic Video CV Pipeline — Demo")

tab_search, tab_results, tab_detect = st.tabs(["🔍 Search", "📊 Fine-tune Results", "🎯 Live Detection"])


# --------------------------------------------------------------------------- #
def load_json(path):
    return json.loads(path.read_text()) if path.exists() else None


# --------------------------------------------------------------------------- #
with tab_search:
    st.subheader("Semantic clip search")
    st.caption("Live call to hybrid_search() -- CLIP text embedding -> Qdrant nearest-neighbour search.")
    col1, col2, col3 = st.columns([3, 1, 1])
    query = col1.text_input("Query", placeholder="e.g. a bus driving on the road")
    embedding_type = col2.selectbox("Embedding", ["frame", "crop"], index=0)
    top_k = col3.slider("Top K", 1, 10, 5)

    if query:
        try:
            from src.retrieval.hybrid_search import search
            with st.spinner("Searching..."):
                results = search(query, embedding_type=embedding_type, top_k=top_k)
        except Exception as e:
            st.error(f"Search failed -- is Docker running? (`docker-compose up -d`)\n\n{e}")
            results = []

        if not results:
            st.info("No results.")
        for i, r in enumerate(results, 1):
            clip_path = r.get("clip_path", "")
            score = r.get("score", 0.0)
            full_path = ROOT / clip_path if not str(clip_path).startswith("/") else Path(clip_path)
            with st.container(border=True):
                c1, c2 = st.columns([1, 3])
                c1.metric(f"#{i} score", f"{score:.3f}")
                if full_path.exists():
                    c1.video(str(full_path))
                else:
                    c1.warning(f"clip not found: {full_path}")
                c2.write(f"**{Path(clip_path).stem}**")
                extra = {k: v for k, v in r.items() if k not in ("clip_path", "score")}
                if extra:
                    c2.json(extra)


# --------------------------------------------------------------------------- #
with tab_results:
    st.subheader("Detector fine-tune: before vs after")
    st.caption("Same 6 held-out eval sequences, same protocol-fair scoring (ignored regions + "
              "3-class car/bus/truck). Only the detector weights differ between rows.")

    ft = load_json(ROOT / "eval" / "finetune_before_after.json")
    if ft:
        b, f = ft["baseline"]["detection"], ft["finetuned"]["detection"]
        bt, ftt = ft["baseline"]["tracking"], ft["finetuned"]["tracking"]

        st.markdown("**Detection (COCO AP)**")
        cols = st.columns(5)
        for col, label, bv, fv in zip(cols,
                ["mAP@0.5", "mAP@0.5:0.95", "car AP", "bus AP", "truck AP"],
                [b["mAP50"], b["mAP50_95"], b["per_class_AP50"]["car"],
                 b["per_class_AP50"]["bus"], b["per_class_AP50"]["truck"]],
                [f["mAP50"], f["mAP50_95"], f["per_class_AP50"]["car"],
                 f["per_class_AP50"]["bus"], f["per_class_AP50"]["truck"]]):
            col.metric(label, f"{fv:.3f}", delta=f"{fv - bv:+.3f}")

        st.markdown("**Tracking (motmetrics)**")
        cols = st.columns(5)
        for col, label, bv, fv in zip(cols,
                ["MOTA", "IDF1", "ID switches", "false positives", "misses"],
                [bt["MOTA"], bt["IDF1"], bt["ID_switches"], bt["false_positives"], bt["misses"]],
                [ftt["MOTA"], ftt["IDF1"], ftt["ID_switches"], ftt["false_positives"], ftt["misses"]]):
            higher_is_better = label in ("MOTA", "IDF1")
            delta = fv - bv
            col.metric(label, f"{fv:.3f}" if isinstance(fv, float) else fv,
                      delta=f"{delta:+.3f}" if isinstance(delta, float) else f"{delta:+d}",
                      delta_color="normal" if higher_is_better else "inverse")
    else:
        st.warning("eval/finetune_before_after.json not found -- run `python -m eval.compare_before_after` first.")

    st.divider()
    st.subheader("Occlusion recovery: baseline vs fine-tuned")
    col1, col2 = st.columns(2)
    for col, label, suffix in [(col1, "Baseline", ""), (col2, "Fine-tuned", "_finetuned")]:
        ev = load_json(ROOT / "eval" / f"occlusion_events{suffix}.json")
        with col:
            st.markdown(f"**{label}**")
            if ev is None:
                st.info("not run yet")
                continue
            counts = {"survived": 0, "switched": 0, "lost": 0}
            for e in ev:
                counts[e["status"]] += 1
            n = sum(counts.values())
            st.metric("Survival rate", f"{counts['survived'] / n:.0%}" if n else "n/a",
                      help=f"{counts['survived']}/{n} occlusion events")
            st.bar_chart(counts)

    st.divider()
    st.subheader("Speed (RTX 3050)")
    bench = load_json(ROOT / "eval" / "benchmark_results.json")
    if bench:
        rows = [{"setting": k, "FPS": round(v["fps"], 1), "x real-time": round(v["realtime_x"], 2)}
                for k, v in bench.get("detection", {}).items()]
        st.table(rows)
    else:
        st.info("eval/benchmark_results.json not found -- run `python -m eval.benchmark` first.")


# --------------------------------------------------------------------------- #
with tab_detect:
    st.subheader("Live detection with the fine-tuned model")
    st.caption("Runs runs/finetune/v1/weights/best.pt on a chosen frame, right now, in this browser tab.")

    if not FINETUNED_WEIGHTS.exists():
        st.error(f"Weights not found at {FINETUNED_WEIGHTS}")
    else:
        col1, col2 = st.columns(2)
        camera = col1.selectbox("Camera", EVAL_SEQS)
        frame_dir = ROOT / "data" / "raw" / camera
        frames = sorted(frame_dir.glob("img*.jpg")) if frame_dir.exists() else []
        if not frames:
            st.warning(f"No frames found in {frame_dir}")
        else:
            idx = col2.slider("Frame", 0, len(frames) - 1, 0)
            conf = st.slider("Confidence threshold", 0.05, 0.9, 0.25, 0.05)

            @st.cache_resource
            def load_model():
                from ultralytics import YOLO
                return YOLO(str(FINETUNED_WEIGHTS))

            model = load_model()
            with st.spinner("Running detection..."):
                result = model.predict(str(frames[idx]), conf=conf, device=0, verbose=False)[0]
                annotated = result.plot()[:, :, ::-1]   # BGR -> RGB
            st.image(annotated, caption=f"{camera} / {frames[idx].name}", width='stretch')
            n_by_class = {}
            for c in result.boxes.cls.tolist():
                name = model.names[int(c)]
                n_by_class[name] = n_by_class.get(name, 0) + 1
            st.write(f"**Detections:** {n_by_class or 'none'}")
