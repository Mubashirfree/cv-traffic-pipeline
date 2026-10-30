"""Phase 6 evaluation: 10 queries -> Gemini summaries -> automated screen + CSV for MANUAL checking.
  python -m eval.rag_eval          then fill manual_ok in report/figures/rag_manual_check.csv, then
  python -m eval.rag_eval --score
"""
import argparse, csv, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FIG = ROOT / "report" / "figures"
CSV = FIG / "rag_manual_check.csv"

QUERIES = [("detrac", "heavy traffic with many vehicles"), ("detrac", "multiple buses on the road"),
           ("detrac", "an almost empty road with hardly any traffic"), ("detrac", "a night road with several cars"),
           ("detrac", "cars close to the camera"),
           ("indian", "many motorcycles and scooters on the road"), ("indian", "auto rickshaws on the road"),
           ("indian", "a bus on the road"), ("indian", "heavy traffic with many vehicles"),
           ("indian", "trucks on the road"),
           ("indian_video", "heavy traffic with many vehicles"), ("indian_video", "motorcycles on the road")]


def run():
    from src.retrieval.service import search
    from src.rag.gemini_summarizer import summarize
    from src.rag.check import check
    FIG.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds, q in QUERIES:
        hits = search(q, ds, "frame", top_k=3)
        out = {"summary": "ERROR", "model": None}
        for attempt in range(3):
            try:
                out = summarize(q, hits); break
            except Exception as e:
                print("  retry:", str(e)[:100]); time.sleep(20); out = {"summary": f"ERROR {e}", "model": None}
        flags = check(out["summary"], hits)
        rows.append({"dataset": ds, "query": q, "summary": out["summary"], "auto_flags": "; ".join(flags),
                     "metadata": json.dumps([{k: h[k] for k in ("clip_id", "camera_id", "timestamp", "counts")} for h in hits]),
                     "manual_ok": ""})
        print(f"[{ds}] {q} -> flags: {flags or 'none'}")
        time.sleep(5)
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    n_auto_ok = sum(1 for r in rows if not r["auto_flags"])
    print(f"\nautomated screen: {n_auto_ok}/{len(rows)} summaries without flags. Now fill manual_ok in {CSV}")


def score():
    rows = list(csv.DictReader(open(CSV)))
    done = [r for r in rows if r["manual_ok"].strip() in ("0", "1")]
    ok = sum(int(r["manual_ok"]) for r in done)
    auto_ok = sum(1 for r in rows if not r["auto_flags"])
    md = ["### RAG summary faithfulness", "",
          f"- summaries evaluated: {len(rows)} (manually checked: {len(done)})",
          f"- manual check, fully supported: {ok}/{len(done)} = {ok/max(1,len(done)):.0%}  "
          f"(hallucination rate {1-ok/max(1,len(done)):.0%})",
          f"- automated screen, no flags: {auto_ok}/{len(rows)}",
          f"- summary counts consistent with a human count of the image: "
          f"{sum(1 for r in rows if r.get('image_consistent')=='1')}/{sum(1 for r in rows if r.get('image_consistent') in ('0','1'))} "
          "(errors here come from the detector counts, not from Gemini)", "",
          "Failures (kept, not hidden):", ""]
    md += [f"- [{r['dataset']}] {r['query']}: {r['summary'][:200]}" for r in done if r["manual_ok"] == "0"]
    (FIG / "rag_faithfulness.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--score", action="store_true")
    score() if ap.parse_args().score else run()
