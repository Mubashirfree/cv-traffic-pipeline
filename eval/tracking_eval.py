"""
Phase 5: Compute MOTA, MOTP, IDF1, and ID-switch count using motmetrics,
comparing our ByteTrack output against UA-DETRAC ground-truth tracks.

Predictions are filtered to vehicle classes only (car/bus/truck) since
UA-DETRAC ground truth contains no person/backpack/handbag annotations --
without this filter, every non-vehicle detection counts as a false
positive against a vehicle-only ground truth, badly distorting MOTA.
"""

import json
from pathlib import Path

import motmetrics as mm
import numpy as np

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

ROOT = Path(__file__).parent.parent
CAMERAS = ["MVI_20011", "MVI_20012", "MVI_20032", "MVI_20033", "MVI_20034", "MVI_40992"]
VID_STRIDE = 10
VEHICLE_CLASSES = {"car", "bus", "truck"}


def iou_distance_matrix(gt_boxes, pred_boxes):
    return mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)


def evaluate_camera(camera_id, gt_records):
    with open(ROOT / "data" / "detections" / f"{camera_id}.json") as f:
        pred_logs = json.load(f)

    gt_by_frame = {}
    for r in gt_records:
        gt_by_frame.setdefault(r["frame_num"], []).append(r)

    acc = mm.MOTAccumulator(auto_id=True)

    for idx, frame in enumerate(pred_logs):
        frame_num = idx * VID_STRIDE + 1
        gt_here = gt_by_frame.get(frame_num, [])
        gt_ids = [g["gt_track_id"] for g in gt_here]
        gt_boxes = [g["bbox"] for g in gt_here]

        # filter to vehicle-class predictions only -- matches GT scope
        preds_here = [d for d in frame["detections"] if d["class"] in VEHICLE_CLASSES]
        pred_ids = [d["track_id"] for d in preds_here]
        pred_boxes = []
        for d in preds_here:
            x1, y1, x2, y2 = d["bbox_px"]
            pred_boxes.append([x1, y1, x2 - x1, y2 - y1])

        dist = iou_distance_matrix(gt_boxes, pred_boxes) if gt_boxes and pred_boxes else np.empty((len(gt_boxes), len(pred_boxes)))
        acc.update(gt_ids, pred_ids, dist)

    return acc


def main():
    with open(ROOT / "eval" / "gt_tracks.json") as f:
        gt_tracks = json.load(f)

    accs, names = [], []
    for camera_id in CAMERAS:
        acc = evaluate_camera(camera_id, gt_tracks[camera_id])
        accs.append(acc)
        names.append(camera_id)

    mh = mm.metrics.create()
    summary = mh.compute_many(
        accs, names=names,
        metrics=["mota", "motp", "idf1", "num_switches", "num_misses", "num_false_positives"],
        generate_overall=True,
    )
    print(summary.to_string())

    summary.to_csv(ROOT / "eval" / "tracking_results.csv")
    print(f"\nSaved -> eval/tracking_results.csv")


if __name__ == "__main__":
    main()
