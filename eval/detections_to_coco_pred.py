"""
Phase 5: Convert our Phase 1 detection JSON logs into COCO-format
predictions, matched to the same image_ids as gt_coco.json.

Uses the detection log's array index (not the lossy frame_ts string,
which truncates to whole seconds and collapses distinct sampled frames)
to recover the correct original UA-DETRAC frame number.
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
CAT_NAME_TO_ID = {"car": 1, "bus": 2, "truck": 3}
VID_STRIDE = 10  # must match detect_track.py


def build_predictions():
    with open(ROOT / "eval" / "gt_coco.json") as f:
        gt = json.load(f)

    lookup = {(im["camera_id"], im["frame_num"]): im["id"] for im in gt["images"]}

    predictions = []
    cameras = sorted(set(im["camera_id"] for im in gt["images"]))

    for camera_id in cameras:
        with open(ROOT / "data" / "detections" / f"{camera_id}.json") as f:
            logs = json.load(f)

        for idx, frame in enumerate(logs):
            frame_num = idx * VID_STRIDE + 1  # exact, index-based -- no string parsing
            image_id = lookup.get((camera_id, frame_num))
            if image_id is None:
                continue

            for det in frame["detections"]:
                cls = det["class"]
                if cls not in CAT_NAME_TO_ID:
                    continue
                x1, y1, x2, y2 = det["bbox_px"]
                predictions.append({
                    "image_id": image_id,
                    "category_id": CAT_NAME_TO_ID[cls],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": det["confidence"],
                })

    out_path = ROOT / "eval" / "pred_coco.json"
    with open(out_path, "w") as f:
        json.dump(predictions, f)
    print(f"Wrote {len(predictions)} predictions -> {out_path}")


if __name__ == "__main__":
    build_predictions()
