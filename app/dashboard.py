"""Phase 7 dashboard.   Terminal 1: uvicorn src.api.main:app --port 8000
                       Terminal 2: streamlit run app/dashboard.py"""
import os
import re
from pathlib import Path

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Traffic Video Search", layout="wide")
st.title("Traffic video search")
st.caption("Describe what you want to find - you can include counts, vehicle types and lighting. "
           "The system finds matching clips, and Gemini explains them in plain language.")

LABELS = {"indian_video": "Indian traffic video clips", "indian": "Indian CCTV images (UVH-26)", "detrac": "UA-DETRAC video clips"}
MODES = {"both": "Combined (recommended)", "frame": "Whole scene (frame)", "crop": "Individual vehicles (crop)"}


def err_text(e):
    try:
        return e.response.json().get("detail", str(e))
    except Exception:
        return str(e)


@st.cache_data(ttl=120)
def get_cameras(ds):
    try:
        r = requests.get(f"{API}/cameras", params={"dataset": ds}, timeout=30)
        r.raise_for_status()
        return r.json()["cameras"]
    except Exception:
        return []


def cam_label(c):
    return c if c == "(any)" else (f"Video {c.split('_')[0]}" if re.match(r"^\d{7,}_", c) else c)


with st.sidebar:
    dataset = st.selectbox("Dataset", list(LABELS), format_func=LABELS.get)
    mode = st.radio("Search mode", list(MODES), format_func=MODES.get,
                    help="Whole scene: compares the full picture (best for 'busy junction at night'). "
                         "Individual vehicles: compares the vehicles that were detected (meant for 'a white truck'). "
                         "Combined: uses both (70% scene, 30% vehicles).")
    camera = st.selectbox("Only search within", ["(any)"] + get_cameras(dataset), format_func=cam_label,
                          help="Restricts the search to one camera / video file.")
    smart = st.checkbox("Understand counts, vehicle types and lighting", value=True,
                        help="On: conditions in your text ('more than 3 trucks', 'no buses', 'at night', 'empty road') are "
                             "applied to the detected objects. Off: pure picture similarity only.")
    top_k = st.slider("Results", 1, 10, 5)
    st.caption("Try: 'more than 3 trucks at night', 'at least 5 motorcycles', 'buses but no trucks', "
               "'empty road with few vehicles', 'heavy traffic with many vehicles', '3 auto rickshaws in the day'")

query = st.text_input("Describe what you are looking for", "more than 3 motorcycles and auto rickshaws")
if st.button("Search", type="primary") and query.strip():
    req = {"query": query, "dataset": dataset, "embedding_type": mode,
           "camera_id": None if camera == "(any)" else camera, "top_k": top_k, "rerank": smart}
    try:
        r = requests.post(f"{API}/search", json=req, timeout=60)
        r.raise_for_status()
        j = r.json()
        st.session_state["res"] = j["results"]
        st.session_state["understood"] = j.get("understood")
        st.session_state["relaxed"] = j.get("relaxed")
        st.session_state["req"] = req
        st.session_state.pop("summary", None)
    except Exception as e:
        st.error(f"Search failed: {err_text(e)}")

res = st.session_state.get("res")
if res is not None:
    if st.session_state.get("understood"):
        st.info(f"Understood as: **{st.session_state['understood']}**")
    if st.session_state.get("relaxed"):
        st.warning("No clip satisfied all the conditions, so these are the closest matches instead.")
    if not res:
        st.warning("No results. Try a different query or filter.")
    for i, h in enumerate(res, 1):
        c1, c2 = st.columns([2, 3])
        with c1:
            path = h["clip_path"] or ""
            if path.lower().endswith((".mp4", ".avi", ".mov")) and Path(path).exists():
                st.video(path)
            else:
                try:
                    k = requests.get(f"{API}/keyframe", params={"path": path}, timeout=30)
                    k.raise_for_status()
                    st.image(k.content, width="stretch")
                except Exception:
                    st.caption("(preview unavailable)")
        with c2:
            light = f" · {h['lighting']}" if h.get("lighting") else ""
            st.markdown(f"### Result {i}")
            st.markdown(f"`{cam_label(h['camera_id'])}` · starts at {h['timestamp']} s{light} · match {h['score']:.2f}")
            st.write("Detected: " + (", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(h["counts"].items()))
                                      or "no objects"))
        st.divider()

    if st.button("Explain these results (Gemini)"):
        with st.spinner("Asking Gemini ..."):
            try:
                a = requests.post(f"{API}/analyze", json={**st.session_state["req"], "max_clips": 3}, timeout=180)
                a.raise_for_status()
                st.session_state["summary"] = a.json()
            except Exception as e:
                st.error(f"Summary failed: {err_text(e)}")
    s = st.session_state.get("summary")
    if s:
        st.subheader("Summary of the top 3 results")
        st.write(s["summary"])
        if s.get("flags"):
            st.warning("Automatic check flagged possible unsupported details: " + "; ".join(s["flags"]))
        else:
            st.caption("Automatic check: no unsupported vehicle types or numbers found (not a full verification). "
                       "Counts come from an automatic detector and may be approximate.")
