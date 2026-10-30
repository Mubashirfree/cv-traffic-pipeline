"""Phase 8: end-to-end integration test through the API (in-process), with latency numbers.
  python -m eval.integration_test            # search + edge cases (no Gemini calls)
  python -m eval.integration_test --gemini   # also 3 /analyze calls
"""
import argparse, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient
from src.api.main import app

Q_DETRAC = ["heavy traffic with many vehicles", "multiple buses on the road", "an almost empty road with hardly any traffic",
            "a night road with several cars", "cars close to the camera", "a truck on the road", "a crowded junction"]
Q_INDIAN = ["many motorcycles and scooters on the road", "auto rickshaws on the road", "a bus on the road",
            "heavy traffic with many vehicles", "trucks on the road", "bicycles on the road", "an almost empty road"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--gemini", action="store_true"); a = ap.parse_args()
    c = TestClient(app)
    rows, lat = [], []

    def record(name, ok, detail="", ms=None):
        rows.append((name, "PASS" if ok else "FAIL", detail, ms))
        print(("PASS " if ok else "FAIL "), name, detail)

    record("GET /health", c.get("/health").status_code == 200)
    c.post("/search", json={"query": "warm up"})
    for ds, qs in (("detrac", Q_DETRAC), ("indian", Q_INDIAN), ("indian_video", Q_INDIAN)):
        for q in qs:
            for et in ("frame", "crop"):
                t = time.perf_counter()
                r = c.post("/search", json={"query": q, "dataset": ds, "embedding_type": et, "top_k": 5})
                ms = (time.perf_counter() - t) * 1000
                ok = r.status_code == 200 and r.json()["n"] > 0
                if ok:
                    lat.append(ms)
                record(f"search[{ds}/{et}] {q}", ok, f"{r.status_code} n={r.json().get('n') if r.status_code==200 else r.text[:80]}", ms)
    record("empty query -> 400", c.post("/search", json={"query": "  "}).status_code == 400)
    record("bad dataset -> 400", c.post("/search", json={"query": "x", "dataset": "nope"}).status_code == 400)
    record("camera filter", c.post("/search", json={"query": "traffic", "camera_id": "MVI_20011"}).status_code == 200)
    record("keyframe outside project -> 403", c.get("/keyframe", params={"path": "/etc/passwd"}).status_code == 403)
    if a.gemini:
        for ds, q in (("detrac", Q_DETRAC[0]), ("indian", Q_INDIAN[0]), ("indian", Q_INDIAN[1])):
            t = time.perf_counter()
            r = c.post("/analyze", json={"query": q, "dataset": ds})
            record(f"analyze[{ds}] {q}", r.status_code == 200 and len(r.json().get("summary", "")) > 20,
                   f"{r.status_code} flags={r.json().get('flags') if r.status_code==200 else r.text[:80]}",
                   (time.perf_counter() - t) * 1000)
    n_pass = sum(1 for r in rows if r[1] == "PASS")
    md = ["### Integration test", "", f"- checks passed: {n_pass}/{len(rows)}",
          f"- /search latency (warm, incl. CLIP text encode + Qdrant): median {np.median(lat):.0f} ms, p95 {np.percentile(lat,95):.0f} ms, n={len(lat)}" if lat else "- no successful searches",
          "", "| check | result | detail | ms |", "|---|---|---|---|"]
    md += [f"| {n} | {s} | {d} | {'' if m is None else f'{m:.0f}'} |" for n, s, d, m in rows]
    (ROOT / "report" / "figures").mkdir(parents=True, exist_ok=True)
    (ROOT / "report" / "figures" / "integration_results.md").write_text("\n".join(md))
    print(f"\n{n_pass}/{len(rows)} passed -> report/figures/integration_results.md")


if __name__ == "__main__":
    main()
