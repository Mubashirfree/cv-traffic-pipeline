"""
Phase 4: Hybrid retrieval — payload filters + CLIP vector similarity,
in a single Qdrant search call.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "embeddings"))
from clip_encoder import embed_text, embed_text_ensemble

from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.storage.qdrant_client import get_client, COLLECTION_NAME


def build_filter(camera_id=None, zone=None, embedding_type="frame"):
    conditions = [FieldCondition(key="embedding_type", match=MatchValue(value=embedding_type))]
    if camera_id:
        conditions.append(FieldCondition(key="camera_id", match=MatchValue(value=camera_id)))
    if zone:
        conditions.append(FieldCondition(key="zone", match=MatchValue(value=zone)))
    return Filter(must=conditions)


def search(query_text, embedding_type="frame", camera_id=None, zone=None, top_k=5, use_ensemble=True):
    """
    query_text: free-text descriptive part, e.g. "a red car near the bus stop"
    embedding_type: "frame" or "crop" -- parameterized for Phase 5 ablation
    camera_id / zone: optional hard filters
    use_ensemble: if True (default), average the CLIP text embedding over several
        phrasings of the query before searching, to reduce sensitivity to exact
        wording. Set False to reproduce the original single-embedding behaviour
        (e.g. for exact comparison against earlier eval runs).
    """
    client = get_client()
    text_vec = embed_text_ensemble(query_text) if use_ensemble else embed_text(query_text)
    query_filter = build_filter(camera_id=camera_id, zone=zone, embedding_type=embedding_type)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=text_vec,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "clip_path": r.payload.get("clip_path"),
            "camera_id": r.payload.get("camera_id"),
            "timestamp": r.payload.get("timestamp"),
            "zone": r.payload.get("zone"),
            "tracked_entities": r.payload.get("tracked_entities"),
            "score": round(r.score, 4),
        }
        for r in results.points
    ]


if __name__ == "__main__":
    results = search("a white car driving on the road", embedding_type="frame", top_k=5)
    for r in results:
        print(r)
