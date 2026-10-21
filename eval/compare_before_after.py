"""
eval/compare_before_after.py -- fine-tune before/after, on the SAME held-out eval sequences,
under the SAME protocol-fair rules as eval_fair.py (ignored regions honoured, 3-class
car/bus/truck, van->truck, others dropped). Only the detections directory differs between rows.

Run from the project root, AFTER eval/detect_finetuned.py has produced
data/detections_finetuned/*.json:
    python -m eval.compare_before_after
Outputs: report/figures/finetune_before_after.md, eval/finetune_before_after.json
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ANN_DIR = ROOT / "data" / "annotations"
RAW_DIR = ROOT / "data" / "raw"
OUT_MD = ROOT / "report" / "figures" / "finetune_before_after.md"
OUT_JSON = ROOT / "eval" / "finetune_before_after.json"

CAMERAS = ["MVI_20011", "MVI_20012", "MVI_20032", "MVI_20033", "MVI_20034", "MVI_40992"]
VID_STRIDE = 10
VEHICLE_CLASSES = {"car", "bus", "truck"}
IGNORE_FRACTION = 0.5
ORIGINAL_GT_MAP = {"car": "car", "bus": "bus", "van": "truck"}


def frame_size(camera):
    try:
        import cv2
        img = cv2.imread(str(RAW_DIR / camera / "img00001.jpg"))
        if img is not None:
            return img.shape[0], img.shape[1]
    except ImportError:
        pass
    return 540, 960


def load_gt_and_mask(camera):
    root = ET.parse(ANN_DIR / f"{camera}.xml").getroot()
    ignored = []
    ig = root.find("ignored_region")
    if ig is not None:
        for b in ig.findall("box"):
            ignored.append([float(b.get(k)) for k in ("left", "top", "width", "height")])
    gt, frame_nums = {}, []
    for fr in root.findall("frame"):
        fnum = int(fr.get("num")); frame_nums.append(fnum)
        tl = fr.find("target_list")
        if tl is None:
            continue
        for t in tl.findall("target"):
            box, attr = t.find("box"), t.find("attribute")
            if box is None:
                continue
            gt.setdefault(fnum, []).append((int(t.get("id")),
                attr.get("vehicle_type", "others") if attr is not None else "others",
                [float(box.get(k)) for k in ("left", "top", "width", "height")]))
    H, W = frame_size(camera)
    mask = np.zeros((H, W), dtype=bool)
    for l, t, w, h in ignored:
        mask[max(0, int(t)):int(t + h), max(0, int(l)):int(l + w)] = True
    max_frame = max(frame_nums) if frame_nums else 0
    sampled = sorted(f for f in range(1, max_frame + 1, VID_STRIDE) if f in gt)
    return gt, mask, sampled, (H, W)


def load_preds(det_dir, camera):
    with open(det_dir / f"{camera}.json") as f:
        logs = json.load(f)
    preds = {}
    for idx, frame in enumerate(logs):
        preds[idx * VID_STRIDE + 1] = frame["detections"]
    return preds


def in_ignored(mask, xyxy, frac=IGNORE_FRACTION):
    H, W = mask.shape
    x1, y1 = max(0, int(round(xyxy[0]))), max(0, int(round(xyxy[1])))
    x2, y2 = min(W, int(round(xyxy[2]))), min(H, int(round(xyxy[3])))
    if x2 <= x1 or y2 <= y1:
        return False
    return mask[y1:y2, x1:x2].mean() >= frac


def build_coco(all_data):
    cats = ["car", "bus", "truck"]
    cid = {n: i + 1 for i, n in enumerate(cats)}
    images, anns, preds = [], [], []
    img_id = ann_id = 0
    for cam, (gt, mask, sampled, preds_by_frame) in all_data.items():
        for fnum in sampled:
            img_id += 1
            images.append({"id": img_id, "file_name": f"{cam}/img{fnum:05d}.jpg"})
            for _, vtype, (l, t, w, h) in gt.get(fnum, []):
                cat = ORIGINAL_GT_MAP.get(vtype)
                if cat is None:
                    continue
                ann_id += 1
                anns.append({"id": ann_id, "image_id": img_id, "category_id": cid[cat],
                             "bbox": [l, t, w, h], "area": w * h, "iscrowd": 0})
            for p in preds_by_frame.get(fnum, []):
                if p["class"] not in cid or in_ignored(mask, p["bbox_px"]):
                    continue
                x1, y1, x2, y2 = p["bbox_px"]
                preds.append({"image_id": img_id, "category_id": cid[p["class"]],
                              "bbox": [x1, y1, x2 - x1, y2 - y1], "score": p["confidence"]})
    return images, anns, preds, cats


def run_coco(images, anns, preds, cats):
    import contextlib
    import io
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"images": images, "annotations": anns,
                      "categories": [{"id": i + 1, "name": n} for i, n in enumerate(cats)]}
        gt.createIndex()
        dt = gt.loadRes(preds)
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
    per_class = {}
    for k, name in enumerate(cats):
        p = ev.eval["precision"][0, :, k, 0, 2]
        p = p[p > -1]
        per_class[name] = float(p.mean()) if p.size else float("nan")
    return {"mAP50": float(ev.stats[1]), "mAP50_95": float(ev.stats[0]), "per_class_AP50": per_class}


def run_tracking(all_data, det_dir):
    import motmetrics as mm
    accs, names = [], []
    for cam, (gt, mask, sampled, preds_by_frame) in all_data.items():
        acc = mm.MOTAccumulator(auto_id=True)
        for fnum in sampled:
            gt_here = gt.get(fnum, [])
            gt_ids = [g[0] for g in gt_here]
            gt_boxes = [g[2] for g in gt_here]
            pr = [p for p in preds_by_frame.get(fnum, []) if p["class"] in VEHICLE_CLASSES
                  and not in_ignored(mask, p["bbox_px"])]
            pr_ids = [p["track_id"] for p in pr]
            pr_boxes = [[p["bbox_px"][0], p["bbox_px"][1], p["bbox_px"][2] - p["bbox_px"][0],
                         p["bbox_px"][3] - p["bbox_px"][1]] for p in pr]
            dist = (mm.distances.iou_matrix(gt_boxes, pr_boxes, max_iou=0.5)
                    if gt_boxes and pr_boxes else np.empty((len(gt_boxes), len(pr_boxes))))
            acc.update(gt_ids, pr_ids, dist)
        accs.append(acc); names.append(cam)
    mh = mm.metrics.create()
    s = mh.compute_many(accs, names=names, generate_overall=True,
                        metrics=["mota", "idf1", "num_switches", "num_misses", "num_false_positives"])
    o = s.loc["OVERALL"]
    return {"MOTA": float(o["mota"]), "IDF1": float(o["idf1"]), "ID_switches": int(o["num_switches"]),
            "false_positives": int(o["num_false_positives"]), "misses": int(o["num_misses"])}


def evaluate(det_dir):
    all_data = {}
    for cam in CAMERAS:
        gt, mask, sampled, _ = load_gt_and_mask(cam)
        preds = load_preds(det_dir, cam)
        all_data[cam] = (gt, mask, sampled, preds)
    det = run_coco(*build_coco(all_data))
    trk = run_tracking(all_data, det_dir)
    return det, trk


def main():
    baseline_dir = ROOT / "data" / "detections"
    finetuned_dir = ROOT / "data" / "detections_finetuned"
    if not finetuned_dir.exists():
        sys.exit(f"{finetuned_dir} not found -- run `python -m eval.detect_finetuned` first")

    print("Evaluating BASELINE (original YOLOv8n)...")
    base_det, base_trk = evaluate(baseline_dir)
    print("Evaluating FINE-TUNED model...")
    ft_det, ft_trk = evaluate(finetuned_dir)

    L = ["# Fine-tune before/after (same 6 held-out eval sequences, same protocol-fair scoring)",
         "", "Both rows use: ignored-region masking + car/bus/truck (van->truck) mapping "
         "-- the D1/T1 protocol from eval_fair.py. Only the detector weights differ.", "",
         "## Detection (COCO AP)", "",
         "| Model | mAP@0.5 | mAP@0.5:0.95 | car AP | bus AP | truck AP |",
         "|---|---|---|---|---|---|",
         f"| Baseline (COCO-pretrained YOLOv8n) | {base_det['mAP50']:.3f} | {base_det['mAP50_95']:.3f} | "
         f"{base_det['per_class_AP50']['car']:.3f} | {base_det['per_class_AP50']['bus']:.3f} | "
         f"{base_det['per_class_AP50']['truck']:.3f} |",
         f"| Fine-tuned (54 UA-DETRAC sequences) | {ft_det['mAP50']:.3f} | {ft_det['mAP50_95']:.3f} | "
         f"{ft_det['per_class_AP50']['car']:.3f} | {ft_det['per_class_AP50']['bus']:.3f} | "
         f"{ft_det['per_class_AP50']['truck']:.3f} |",
         "", "## Tracking (motmetrics)", "",
         "| Model | MOTA | IDF1 | ID switches | false positives | misses |",
         "|---|---|---|---|---|---|",
         f"| Baseline | {base_trk['MOTA']:.3f} | {base_trk['IDF1']:.3f} | {base_trk['ID_switches']} | "
         f"{base_trk['false_positives']} | {base_trk['misses']} |",
         f"| Fine-tuned | {ft_trk['MOTA']:.3f} | {ft_trk['IDF1']:.3f} | {ft_trk['ID_switches']} | "
         f"{ft_trk['false_positives']} | {ft_trk['misses']} |"]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n")
    OUT_JSON.write_text(json.dumps({"baseline": {"detection": base_det, "tracking": base_trk},
                                    "finetuned": {"detection": ft_det, "tracking": ft_trk}}, indent=2))
    print("\n" + "\n".join(L) + f"\n\nSaved -> {OUT_MD}\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
