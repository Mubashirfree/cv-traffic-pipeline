"""
eval/retrieval_eval.py  --  Phase 5 retrieval ablation: frame vs crop embeddings.

Runs every labelled query in eval/retrieval_gt.json through
src.retrieval.hybrid_search.search() twice (embedding_type='frame' and 'crop'),
with NO camera/zone filters so the comparison isolates the embedding type,
then computes Precision@K, Recall@K and MRR for each.

Run from the project root:
    python -m eval.retrieval_eval
    python -m eval.retrieval_eval --ks 1 3 5 10

Outputs:
    eval/retrieval_results.csv            per-query, per-embedding-type metrics
    eval/retrieval_details.json           top-k retrieved clips per query (error analysis)
    report/figures/retrieval_ablation.md  summary table, ready to paste into the report
    report/figures/retrieval_ablation.png bar chart (if matplotlib is available)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `from src...` works however the script is launched

GT_PATH = ROOT / "eval" / "retrieval_gt.json"
MANIFEST_PATH = ROOT / "data" / "clips" / "manifest.json"
RESULTS_CSV = ROOT / "eval" / "retrieval_results.csv"
DETAILS_JSON = ROOT / "eval" / "retrieval_details.json"
FIG_DIR = ROOT / "report" / "figures"
EMBEDDING_TYPES = ["frame", "crop"]


# --------------------------------------------------------------------------- #
# Pure metric functions (no DB / model needed, easy to unit-test)
# --------------------------------------------------------------------------- #
def clip_id_from_path(path_or_id: str) -> str:
    """'data/clips/MVI_20012_005.mp4' -> 'MVI_20012_005'; bare ids pass through."""
    return Path(str(path_or_id)).stem


def precision_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & relevant) / k


def recall_at_k(retrieved, relevant, k):
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved, relevant):
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_query(retrieved, relevant, ks):
    row = {f"P@{k}": precision_at_k(retrieved, relevant, k) for k in ks}
    row.update({f"R@{k}": recall_at_k(retrieved, relevant, k) for k in ks})
    row["MRR"] = reciprocal_rank(retrieved, relevant)
    return row


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_ground_truth():
    with open(GT_PATH) as f:
        raw = json.load(f)
    queries, skipped = [], []
    for q in raw["queries"]:
        rel = {clip_id_from_path(c) for c in q.get("relevant_clips", [])}
        if not rel:
            skipped.append(q["id"])
            continue
        queries.append({
            "id": q["id"],
            "query": q["query"],
            "category": q.get("category", "uncategorised"),
            "relevant": rel,
        })
    return queries, skipped


def count_total_clips():
    try:
        with open(MANIFEST_PATH) as f:
            m = json.load(f)
        return len(m)
    except Exception:
        return 73  # known corpus size from Phase 3


# --------------------------------------------------------------------------- #
# Main experiment
# --------------------------------------------------------------------------- #
def run(ks, search_fn=None):
    if search_fn is None:
        from src.retrieval.hybrid_search import search as search_fn  # loads CLIP once

    queries, skipped = load_ground_truth()
    if skipped:
        print(f"[warn] skipping {len(skipped)} query(ies) with no relevant_clips yet: {skipped}")
    if not queries:
        sys.exit("No labelled queries found. Fill in relevant_clips in eval/retrieval_gt.json first.")

    top_k = max(ks)
    n_total = count_total_clips()
    rows, details = [], []

    for q in queries:
        detail = {"id": q["id"], "query": q["query"], "category": q["category"],
                  "relevant": sorted(q["relevant"])}
        for etype in EMBEDDING_TYPES:
            hits = search_fn(q["query"], embedding_type=etype, top_k=top_k)
            retrieved = [clip_id_from_path(h["clip_path"]) for h in hits]
            metrics = evaluate_query(retrieved, q["relevant"], ks)
            rows.append({"query_id": q["id"], "category": q["category"],
                         "n_relevant": len(q["relevant"]), "embedding_type": etype,
                         **metrics})
            detail[etype] = [{"clip": c, "relevant": c in q["relevant"],
                              "score": round(h["score"], 4)}
                             for c, h in zip(retrieved, hits)]
        details.append(detail)

    # Random baseline: expected P@K for random ranking = |relevant| / N_total
    baseline_p = mean(len(q["relevant"]) / n_total for q in queries)
    return queries, rows, details, baseline_p, n_total


def summarise(rows, ks):
    """Mean of each metric per (embedding_type) and per (category, embedding_type)."""
    metric_names = [f"P@{k}" for k in ks] + [f"R@{k}" for k in ks] + ["MRR"]
    overall = {e: {m: mean(r[m] for r in rows if r["embedding_type"] == e)
                   for m in metric_names} for e in EMBEDDING_TYPES}
    categories = sorted({r["category"] for r in rows})
    by_cat = {}
    for c in categories:
        by_cat[c] = {e: {m: mean(r[m] for r in rows
                             if r["embedding_type"] == e and r["category"] == c)
                         for m in metric_names} for e in EMBEDDING_TYPES}
    return metric_names, overall, by_cat


def write_outputs(queries, rows, details, baseline_p, n_total, ks):
    metric_names, overall, by_cat = summarise(rows, ks)

    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(DETAILS_JSON, "w") as f:
        json.dump(details, f, indent=2)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    def table(title, data):
        lines = [f"**{title}**", "",
                 "| Embedding | " + " | ".join(metric_names) + " |",
                 "|---|" + "---|" * len(metric_names)]
        for e in EMBEDDING_TYPES:
            lines.append(f"| {e} | " + " | ".join(f"{data[e][m]:.3f}" for m in metric_names) + " |")
        return "\n".join(lines)

    md = [f"# Retrieval ablation: frame vs crop embeddings", "",
          f"{len(queries)} labelled queries, corpus of {n_total} sub-clips, "
          f"no camera/zone filters. Random-ranking baseline P@K ~ {baseline_p:.3f}.", "",
          table("Overall (mean over queries)", overall), ""]
    if len(by_cat) > 1:
        for c, data in by_cat.items():
            n_c = len({r["query_id"] for r in rows if r["category"] == c})
            md += [table(f"Category: {c} ({n_c} queries)", data), ""]
    (FIG_DIR / "retrieval_ablation.md").write_text("\n".join(md))

    # Console summary
    print("\n" + "\n".join(md))

    # Chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        width = 0.38
        xs = range(len(metric_names))
        for i, e in enumerate(EMBEDDING_TYPES):
            ax.bar([x + (i - 0.5) * width for x in xs],
                   [overall[e][m] for m in metric_names], width, label=e)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0, 1)
        ax.set_ylabel("score")
        ax.set_title(f"Frame vs crop embeddings ({len(queries)} queries)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "retrieval_ablation.png", dpi=200)
        print(f"Saved chart -> {FIG_DIR / 'retrieval_ablation.png'}")
    except ImportError:
        print("[warn] matplotlib not installed; skipped chart")

    print(f"Saved -> {RESULTS_CSV}\nSaved -> {DETAILS_JSON}\nSaved -> {FIG_DIR / 'retrieval_ablation.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    args = ap.parse_args()
    q, r, d, b, n = run(args.ks)
    write_outputs(q, r, d, b, n, args.ks)
