-- Phase 3: Postgres schema per 01_Project_Structure_and_Tech_Stack.md §4

CREATE TABLE IF NOT EXISTS clips (
    id SERIAL PRIMARY KEY,
    clip_id TEXT UNIQUE NOT NULL,
    camera_id TEXT NOT NULL,
    start_ts FLOAT NOT NULL,
    end_ts FLOAT NOT NULL,
    path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS detections (
    id SERIAL PRIMARY KEY,
    clip_id TEXT REFERENCES clips(clip_id),
    track_id INTEGER,
    class TEXT,
    bbox_px FLOAT[],
    world_xy FLOAT[],
    zone TEXT,
    confidence FLOAT
);

CREATE TABLE IF NOT EXISTS zones (
    camera_id TEXT NOT NULL,
    name TEXT NOT NULL,
    polygon_world JSONB NOT NULL,
    PRIMARY KEY (camera_id, name)
);
