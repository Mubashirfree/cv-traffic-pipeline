"""
Phase 5: Extract ground-truth tracks (frame, track_id, bbox) from
UA-DETRAC XML, restricted to the same sampled frames used elsewhere.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).parent.parent
CAMERAS = ["MVI_20011", "MVI_20012", "MVI_20032", "MVI_20033", "MVI_20034", "MVI_40992"]
VID_STRIDE = 10


def extract_gt_tracks(camera_id):
    xml_path = ROOT / "data" / "annotations" / f"{camera_id}.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()

    all_frames = [int(f.get("num")) for f in root.findall("frame")]
    max_frame = max(all_frames) if all_frames else 0
    valid_frames = set(range(1, max_frame + 1, VID_STRIDE))

    records = []
    for frame in root.findall("frame"):
        fnum = int(frame.get("num"))
        if fnum not in valid_frames:
            continue
        target_list = frame.find("target_list")
        if target_list is None:
            continue
        for target in target_list.findall("target"):
            box = target.find("box")
            if box is None:
                continue
            records.append({
                "frame_num": fnum,
                "gt_track_id": int(target.get("id")),
                "bbox": [float(box.get("left")), float(box.get("top")),
                         float(box.get("width")), float(box.get("height"))],
            })
    return records


if __name__ == "__main__":
    all_records = {}
    for camera_id in CAMERAS:
        recs = extract_gt_tracks(camera_id)
        all_records[camera_id] = recs
        print(f"[{camera_id}] {len(recs)} GT track records")

    with open(ROOT / "eval" / "gt_tracks.json", "w") as f:
        json.dump(all_records, f)
