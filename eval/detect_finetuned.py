"""
eval/detect_finetuned.py -- run the fine-tuned weights on the 6 HELD-OUT eval sequences only,
producing detections in the same JSON schema as the original baseline detections, so eval_fair.py
and occlusion_check.py can be pointed at either one for a fair before/after comparison.

Writes to data/detections_finetuned/<camera>.json (does NOT touch data/detections/, which stays
the original-model baseline).

Class ids from data/yolo_dataset/data.yaml: 0=car, 1=bus, 2=truck.
Tracker state is reset per camera (fresh model instance) so track ids don't carry over
between sequences.

Run from the project root:
    python -m eval.detect_finetuned
    python -m eval.detect_finetuned --weights runs/detect/runs/finetune/v1/weights/best.pt
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RAW_DIR = ROOT / "data" / "raw"
ANN_DIR = ROOT / "data" / "annotations"          # only the 6 eval sequences live here
OUT_DIR = ROOT / "data" / "detections_finetuned"
EVAL_SEQS = ["MVI_20011", "MVI_20012", "MVI_20032", "MVI_20033", "MVI_20034", "MVI_40992"]
VID_STRIDE = 10
ID_TO_CLASS = {0: "car", 1: "bus", 2: "truck"}
CONF_THRESH = 0.1        # matches the lowest confidence logged by the original pipeline


def max_frame_for(camera):
    import xml.etree.ElementTree as ET
    root = ET.parse(ANN_DIR / f"{camera}.xml").getroot()
    nums = [int(f.get("num")) for f in root.findall("frame")]
    return max(nums) if nums else 0


def run_camera(weights, camera):
    import cv2
    from ultralytics import YOLO
    model = YOLO(str(weights))                   # fresh instance -> fresh tracker state
    max_frame = max_frame_for(camera)
    log = []
    for fnum in range(1, max_frame + 1, VID_STRIDE):
        img_path = RAW_DIR / camera / f"img{fnum:05d}.jpg"
        if not img_path.exists():
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        r = model.track(frame, tracker="bytetrack.yaml", conf=CONF_THRESH, device=0,
                        persist=True, verbose=False)[0]
        detections = []
        boxes = r.boxes
        if boxes is not None and boxes.id is not None:
            for box, tid, cls, conf in zip(boxes.xyxy.tolist(), boxes.id.tolist(),
                                            boxes.cls.tolist(), boxes.conf.tolist()):
                detections.append({"track_id": int(tid), "class": ID_TO_CLASS[int(cls)],
                                   "bbox_px": [round(v, 1) for v in box], "confidence": round(float(conf), 4)})
        log.append({"frame_ts": None, "camera_id": camera, "detections": detections})
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/detect/runs/finetune/v1/weights/best.pt"))
    args = ap.parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        sys.exit(f"weights not found: {weights}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for camera in EVAL_SEQS:
        print(f"[detect] {camera} ...")
        log = run_camera(weights, camera)
        n_det = sum(len(f["detections"]) for f in log)
        (OUT_DIR / f"{camera}.json").write_text(json.dumps(log))
        print(f"  {len(log)} sampled frames, {n_det} detections -> {OUT_DIR / f'{camera}.json'}")

    print(f"\nDone. Original baseline detections in data/detections/ are untouched.")
    print(f"Fine-tuned detections written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
