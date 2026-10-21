"""
eval/dump_clip_summary.py -- write a compact, sorted per-clip summary to eval/clip_summary.txt

Use it as a SHORTLIST to decide which clips to watch while labelling
eval/retrieval_gt.json. Entity counts are known to overstate bus/truck presence
(detector misclassification + ID-switch double counting), so do not label from this alone.

Run from the project root:
    python -m eval.dump_clip_summary
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qdrant_client.models import FieldCondition, Filter, MatchValue  # noqa: E402

from src.storage.qdrant_client import COLLECTION_NAME, get_client  # noqa: E402

OUT = ROOT / "eval" / "clip_summary.txt"


def main():
    client = get_client()
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=[FieldCondition(key="embedding_type", match=MatchValue(value="frame"))]),
        limit=200,
        with_payload=True,
    )
    rows = []
    for p in points:
        pl = p.payload
        clip_id = Path(pl["clip_path"]).stem
        classes = Counter(e.rsplit("_", 1)[0] for e in pl.get("tracked_entities", []))
        rows.append((clip_id, pl["camera_id"], pl.get("timestamp"), pl.get("zone"), dict(classes)))
    rows.sort()

    with open(OUT, "w") as f:
        f.write(f"{'clip_id':<18}{'camera':<12}{'timestamp':<12}{'zone':<12}entity counts (by class)\n")
        for clip_id, cam, ts, zone, classes in rows:
            f.write(f"{clip_id:<18}{cam:<12}{str(ts):<12}{str(zone):<12}{classes}\n")
    print(f"{len(rows)} clips written to {OUT}")


if __name__ == "__main__":
    main()
