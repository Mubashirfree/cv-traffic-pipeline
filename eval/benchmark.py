"""
eval/benchmark.py -- Phase 5: detection FPS and end-to-end query latency.

1. Detection throughput: YOLOv8n predict-only and predict+ByteTrack on consecutive frames of one
   video, on GPU (if available) and CPU. Reports FPS and ms/frame (mean, median, p95) plus
   Ultralytics' own preprocess / inference / postprocess split. Warm-up frames are excluded.
   NOTE: consecutive native-rate (25 fps) frames are used because tracking needs continuity;
   the actual pipeline samples every 10th frame (2.5 fps), so it runs far below this cost.
2. Query latency: hybrid_search.search() for every query in eval/retrieval_gt.json, both
   embedding types, repeated. Reports text-encode time, Qdrant time (total - encode) and total,
   plus the one-off cold-start (import + CLIP load) time.

Run from the project root:
    python -m eval.benchmark
    python -m eval.benchmark --frames 300 --repeats 20 --skip-cpu

Outputs: eval/benchmark_results.json, report/figures/benchmark.md
"""
import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_VIDEO = ROOT / "data" / "videos" / "MVI_20011.mp4"
GT_PATH = ROOT / "eval" / "retrieval_gt.json"
OUT_JSON = ROOT / "eval" / "benchmark_results.json"
OUT_MD = ROOT / "report" / "figures" / "benchmark.md"
CLASSES = [0, 2, 5, 7, 24, 26]   # same COCO classes as detect_track.py
SOURCE_FPS = 25


def pct(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def summarize_ms(values):
    return {"n": len(values), "mean_ms": statistics.mean(values),
            "median_ms": statistics.median(values), "p95_ms": pct(values, 95)}


# --------------------------------------------------------------------------- #
def bench_detection(video, device, mode, n_frames, warmup):
    import cv2
    from ultralytics import YOLO
    weights = ROOT / "yolov8n.pt"
    model = YOLO(str(weights) if weights.exists() else "yolov8n.pt")
    cap = cv2.VideoCapture(str(video))
    wall, pre, inf, post = [], [], [], []
    i = 0
    while i < warmup + n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        if mode == "track":
            r = model.track(frame, tracker="bytetrack.yaml", classes=CLASSES, persist=True,
                            device=device, verbose=False)
        else:
            r = model.predict(frame, classes=CLASSES, device=device, verbose=False)
        dt = (time.perf_counter() - t0) * 1000
        if i >= warmup:
            wall.append(dt)
            sp = r[0].speed
            pre.append(sp["preprocess"]); inf.append(sp["inference"]); post.append(sp["postprocess"])
        i += 1
    cap.release()
    if not wall:
        raise RuntimeError(f"no frames read from {video}")
    s = summarize_ms(wall)
    s.update(fps=1000.0 / s["mean_ms"], realtime_x=(1000.0 / s["mean_ms"]) / SOURCE_FPS,
             preprocess_ms=statistics.mean(pre), inference_ms=statistics.mean(inf),
             postprocess_ms=statistics.mean(post))
    return s


def bench_queries(repeats, warmup_calls=3):
    t0 = time.perf_counter()
    from src.embeddings.clip_encoder import embed_text
    from src.retrieval.hybrid_search import search
    cold_start_s = time.perf_counter() - t0

    if GT_PATH.exists():
        queries = [q["query"] for q in json.load(open(GT_PATH))["queries"]]
    else:
        queries = ["a bus on the road", "heavy traffic", "an empty road at night"]

    out = {"cold_start_s": cold_start_s, "n_queries": len(queries), "repeats": repeats}
    for etype in ("frame", "crop"):
        for q in queries[:1] * warmup_calls:
            search(q, embedding_type=etype, top_k=5)
        totals, encodes = [], []
        for q in queries:
            for _ in range(repeats):
                t = time.perf_counter(); search(q, embedding_type=etype, top_k=5)
                totals.append((time.perf_counter() - t) * 1000)
                t = time.perf_counter(); embed_text(q)
                encodes.append((time.perf_counter() - t) * 1000)
        res = summarize_ms(totals)
        res["text_encode_mean_ms"] = statistics.median(encodes)
        res["qdrant_and_overhead_mean_ms"] = res["median_ms"] - res["text_encode_mean_ms"]
        out[etype] = res
    return out


# --------------------------------------------------------------------------- #
def hardware_info():
    info = {"python": platform.python_version(), "platform": platform.platform(), "cpu": platform.processor()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def write_markdown(results):
    hw = results["hardware"]
    L = ["# Benchmark", "",
         f"Hardware: {hw.get('gpu', 'no GPU')} / torch {hw.get('torch', '?')} / Python {hw['python']}", "",
         "## Detection throughput (YOLOv8n, consecutive native-rate frames)", "",
         "| Setting | FPS | ms/frame (mean) | median | p95 | preprocess | inference | postprocess | x real-time (25 fps) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, d in results["detection"].items():
        L.append(f"| {name} | {d['fps']:.1f} | {d['mean_ms']:.1f} | {d['median_ms']:.1f} | {d['p95_ms']:.1f} | "
                 f"{d['preprocess_ms']:.1f} | {d['inference_ms']:.1f} | {d['postprocess_ms']:.1f} | {d['realtime_x']:.2f} |")
    if "query" in results:
        q = results["query"]
        L += ["", f"## Query latency (top_k=5, {q['n_queries']} queries x {q['repeats']} repeats)", "",
              f"One-off cold start (imports + CLIP load): {q['cold_start_s']:.1f} s", "",
              "| Embedding | total mean (ms) | median | p95 | text encode, median (ms) | Qdrant + overhead, median (ms) |",
              "|---|---|---|---|---|---|"]
        for etype in ("frame", "crop"):
            d = q[etype]
            L.append(f"| {etype} | {d['mean_ms']:.1f} | {d['median_ms']:.1f} | {d['p95_ms']:.1f} | "
                     f"{d['text_encode_mean_ms']:.1f} | {d['qdrant_and_overhead_mean_ms']:.1f} |")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--cpu-frames", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--skip-cpu", action="store_true")
    ap.add_argument("--skip-query", action="store_true")
    args = ap.parse_args()

    results = {"hardware": hardware_info(), "detection": {}}
    devices = []
    if results["hardware"].get("cuda_available"):
        devices.append(("GPU", 0, args.frames))
    if not args.skip_cpu:
        devices.append(("CPU", "cpu", args.cpu_frames))
    for label, dev, n in devices:
        for mode in ("predict", "track"):
            name = f"{label} {'detect only' if mode == 'predict' else 'detect + ByteTrack'}"
            print(f"[bench] {name} ({n} frames) ...")
            results["detection"][name] = bench_detection(args.video, dev, mode, n, args.warmup)
    if not args.skip_query:
        print("[bench] query latency ...")
        results["query"] = bench_queries(args.repeats)

    OUT_JSON.write_text(json.dumps(results, indent=2))
    write_markdown(results)
    print(f"\nSaved -> {OUT_JSON}\nSaved -> {OUT_MD}")


if __name__ == "__main__":
    main()
