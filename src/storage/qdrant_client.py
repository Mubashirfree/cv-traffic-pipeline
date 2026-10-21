"""
Phase 3: Qdrant collection setup + upsert helpers for frame/crop vectors.
"""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION_NAME = "clip_embeddings"
VECTOR_DIM = 512

_client = None


def get_client():
    global _client
    if _client is None:
        _client = QdrantClient(host="localhost", port=6333)
    return _client


def init_collection():
    client = get_client()
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        print(f"Collection '{COLLECTION_NAME}' created.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")


def upsert_embedding(vector, payload):
    """payload should match §4 schema: timestamp, camera_id,
    tracked_entities, zone, clip_path, embedding_type."""
    client = get_client()
    point_id = str(uuid.uuid4())
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(id=point_id, vector=vector, payload=payload)],
    )
    return point_id


if __name__ == "__main__":
    init_collection()
