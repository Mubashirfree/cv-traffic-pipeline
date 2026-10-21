"""
Phase 3: Postgres connection + insert helpers for clips/detections/zones.
"""

import json
from pathlib import Path

import psycopg2

ROOT = Path(__file__).parent.parent.parent

CONN_PARAMS = dict(
    host="localhost",
    port=5432,
    dbname="cvproject",
    user="cvproject",
    password="cvproject",
)


def get_connection():
    return psycopg2.connect(**CONN_PARAMS)


def init_schema():
    schema_path = ROOT / "src" / "storage" / "postgres_schema.sql"
    with open(schema_path) as f:
        sql = f.read()
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    conn.close()
    print("Schema initialized.")


def insert_clip(clip_id, camera_id, start_ts, end_ts, path):
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO clips (clip_id, camera_id, start_ts, end_ts, path)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (clip_id) DO NOTHING""",
            (clip_id, camera_id, start_ts, end_ts, path),
        )
    conn.commit()
    conn.close()


def insert_detection(clip_id, track_id, cls, bbox_px, world_xy, zone, confidence):
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO detections (clip_id, track_id, class, bbox_px, world_xy, zone, confidence)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (clip_id, track_id, cls, bbox_px, world_xy, zone, confidence),
        )
    conn.commit()
    conn.close()


def insert_zone(camera_id, name, polygon_world):
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO zones (camera_id, name, polygon_world)
               VALUES (%s, %s, %s)
               ON CONFLICT (camera_id, name) DO NOTHING""",
            (camera_id, name, json.dumps(polygon_world)),
        )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_schema()
