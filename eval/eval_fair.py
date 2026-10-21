"""
eval/eval_fair.py -- Phase 5: protocol-corrected detection + tracking evaluation.

The original scripts (xml_to_coco_gt.py, detection_eval.py, tracking_eval.py) are left untouched.
This script re-scores the SAME detections under a fairer UA-DETRAC protocol and prints every step,
so the effect of each correction is visible:

  Detection
    D0  original protocol : classes car/bus/truck, GT van->truck, GT 'others' dropped, no ignored regions
    D1  + ignored regions : predictions with >= 50% of their area inside an XML <ignored_region> are
                            discarded (UA-DETRAC's official rule: those areas are not annotated)
    D2  + single class    : class-agnostic 'vehicle' AP (car+bus+van+others vs all vehicle detections),
                            the standard UA-DETRAC detection protocol. Avoids penalising YOLO for calling a
                            GT 'van' a 'car', a labelling difference and not a localisation error.
  Tracking
    T0  original protocol (vehicle-class predictions, no ignored regions)
    T1  + ignored regions

Run from the project root:
    python -m eval.eval_fair
Outputs: report/figures/eval_protocol_comparison.md, eval/eval_fair_results.json
"""
import contextlib
import io
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

if not hasattr(np, "asfarray"):                         # motmetrics compatibility on numpy 2.x
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ANN_DIR = ROOT / "data" / "annotations"
DET_DIR = ROOT / "data" / "detections"
RAW_DIR = ROOT / "data" / "raw"
OUT_MD = ROOT / "report" / "figures" / "eval_protocol_comparison.md"
OUT_JSON = ROOT / "eval" / "eval_fair_results.json"

CAMERAS = ["MVI_20011", "MVI_20012", "MVI_20032", "MVI_20033", "MVI_20034", "MVI_40992"]
VID_STRIDE = 10
VEHICLE_CLASSES = {"car", "bus", "truck"}
IGNORE_FRACTION = 0.5


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def frame_size(camera):
    try:
        import cv2
        img = cv2.imread(str(RAW_DIR / camera / "img00001.jpg"))
        if img is not None:
            return img.shape[0], img.shape[1]
    except ImportError:
        pass
    return 540, 960                                     # UA-DETRAC standard


def load_camera(camera):
    root = ET.parse(ANN_DIR / f"{camera}.xml").getroot()
    ignored = []
    ig = root.find("ignored_region")
    if ig is not None:
        for b in ig.findall("box"):
            ignored.append([float(b.get(k)) for k in ("left", "top", "width", "height")])

    gt, frame_nums = {}, []
    for fr in root.findall("frame"):
        fnum = int(fr.get("num"))
        frame_nums.append(fnum)
        tl = fr.find("target_list")
        if tl is None:
            continue
        for t in tl.findall("target"):
            box, attr = t.find("box"), t.find("attribute")
            if box is None:
                continue
            gt.setdefault(fnum, []).append((
                int(t.get("id")),
                attr.get("vehicle_type", "others") if attr is not None else "others",
                [float(box.get(k)) for k in ("left", "top", "width", "height")]))

    with open(DET_DIR / f"{camera}.json") as f:
        logs = json.load(f)
    preds = {}
    for idx, frame in enumerate(logs):
        preds[idx * VID_STRIDE + 1] = [
            dict(cls=d["class"], conf=d["confidence"], tid=d["track_id"], xyxy=d["bbox_px"])
            for d in frame["detections"]]

    H, W = frame_size(camera)
    mask = np.zeros((H, W), dtype=bool)
    for l, t, w, h in ignored:
        mask[max(0, int(t)):int(t + h), max(0, int(l)):int(l + w)] = True

    max_frame = max(frame_nums) if frame_nums else 0
    valid = set(range(1, max_frame + 1, VID_STRIDE))
    return dict(gt=gt, sampled=sorted(f for f in frame_nums if f in valid), preds=preds,
                mask=mask, n_ignored=len(ignored))


def in_ignored(mask, xyxy, frac=IGNORE_FRACTION):
    H, W = mask.shape
    x1, y1 = max(0, int(round(xyxy[0]))), max(0, int(round(xyxy[1])))
    x2, y2 = min(W, int(round(xyxy[2]))), min(H, int(round(xyxy[3])))
    if x2 <= x1 or y2 <= y1:
        return False
    return mask[y1:y2, x1:x2].mean() >= frac


# --------------------------------------------------------------------------- #
# Detection (COCO AP)
# --------------------------------------------------------------------------- #
ORIGINAL_GT_MAP = {"car": "car", "bus": "bus", "van": "truck"}          # 'others' dropped
VEHICLE_GT_MAP = {"car": "vehicle", "bus": "vehicle", "van": "vehicle", "others": "vehicle"}


def build_coco(data, cats, gt_map, pred_map, use_ignore):
    cid = {n: i + 1 for i, n in enumerate(cats)}
    images, anns, preds = [], [], []
    img_id = ann_id = 0
    for cam, d in data.items():
        for fnum in d["sampled"]:
            img_id += 1
            images.append({"id": img_id, "file_name": f"{cam}/img{fnum:05d}.jpg", "width": d["mask"].shape[1],
                           "height": d["mask"].shape[0]})
            for _, vtype, (l, t, w, h) in d["gt"].get(fnum, []):
                cat = gt_map.get(vtype)
                if cat is None:
                    continue
                ann_id += 1
                anns.append({"id": ann_id, "image_id": img_id, "category_id": cid[cat],
                             "bbox": [l, t, w, h], "area": w * h, "iscrowd": 0})
            for p in d["preds"].get(fnum, []):
                cat = pred_map(p["cls"])
                if cat is None or (use_ignore and in_ignored(d["mask"], p["xyxy"])):
                    continue
                x1, y1, x2, y2 = p["xyxy"]
                preds.append({"image_id": img_id, "category_id": cid[cat],
                              "bbox": [x1, y1, x2 - x1, y2 - y1], "score": p["conf"]})
    return images, anns, preds, cats


def run_coco(images, anns, preds, cats):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    nan = float("nan")
    if not preds or not anns:
        return {"mAP50": 0.0, "mAP50_95": 0.0, "per_class_AP50": {c: nan for c in cats},
                "n_gt": len(anns), "n_pred": len(preds)}
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
        p = ev.eval["precision"][0, :, k, 0, 2]        # IoU 0.5, all areas, maxDets=100
        p = p[p > -1]
        per_class[name] = float(p.mean()) if p.size else nan
    return {"mAP50": float(ev.stats[1]), "mAP50_95": float(ev.stats[0]),
            "per_class_AP50": per_class, "n_gt": len(anns), "n_pred": len(preds)}


def detection_suite(data):
    cls3 = ["car", "bus", "truck"]
    ident = lambda c: c if c in VEHICLE_CLASSES else None
    anyveh = lambda c: "vehicle" if c in VEHICLE_CLASSES else None
    return {
        "D0 original (3 classes, no ignored regions)":
            run_coco(*build_coco(data, cls3, ORIGINAL_GT_MAP, ident, use_ignore=False)),
        "D1 + ignored regions":
            run_coco(*build_coco(data, cls3, ORIGINAL_GT_MAP, ident, use_ignore=True)),
        "D2 + single 'vehicle' class (standard protocol)":
            run_coco(*build_coco(data, ["vehicle"], VEHICLE_GT_MAP, anyveh, use_ignore=True)),
    }


# --------------------------------------------------------------------------- #
# Tracking (motmetrics)
# --------------------------------------------------------------------------- #
def tracking_eval(data, use_ignore):
    import motmetrics as mm
    accs, names = [], []
    for cam, d in data.items():
        acc = mm.MOTAccumulator(auto_id=True)
        for fnum in sorted(d["preds"]):
            gt_here = d["gt"].get(fnum, [])
            gt_ids = [g[0] for g in gt_here]
            gt_boxes = [g[2] for g in gt_here]
            pr = [p for p in d["preds"][fnum] if p["cls"] in VEHICLE_CLASSES
                  and not (use_ignore and in_ignored(d["mask"], p["xyxy"]))]
            pr_ids = [p["tid"] for p in pr]
            pr_boxes = [[p["xyxy"][0], p["xyxy"][1], p["xyxy"][2] - p["xyxy"][0],
                         p["xyxy"][3] - p["xyxy"][1]] for p in pr]
            dist = (mm.distances.iou_matrix(gt_boxes, pr_boxes, max_iou=0.5)
                    if gt_boxes and pr_boxes else np.empty((len(gt_boxes), len(pr_boxes))))
            acc.update(gt_ids, pr_ids, dist)
        accs.append(acc); names.append(cam)
    mh = mm.metrics.create()
    s = mh.compute_many(accs, names=names, generate_overall=True,
                        metrics=["mota", "idf1", "num_switches", "num_misses", "num_false_positives",
                                 "num_objects"])
    o = s.loc["OVERALL"]
    return {"MOTA": float(o["mota"]), "IDF1": float(o["idf1"]), "ID_switches": int(o["num_switches"]),
            "false_positives": int(o["num_false_positives"]), "misses": int(o["num_misses"]),
            "gt_objects": int(o["num_objects"])}


# --------------------------------------------------------------------------- #
def main():
    data = {cam: load_camera(cam) for cam in CAMERAS}
    print("ignored regions per camera:", {c: d["n_ignored"] for c, d in data.items()})
    all_conf = [p["conf"] for d in data.values() for fr in d["preds"].values() for p in fr]
    min_conf = min(all_conf)
    print(f"lowest logged detection confidence: {min_conf:.3f}")

    det = detection_suite(data)
    trk = {"T0 original (no ignored regions)": tracking_eval(data, False),
           "T1 + ignored regions": tracking_eval(data, True)}

    L = ["# Evaluation protocol comparison (same detections, different scoring rules)", "",
         f"Lowest logged detection confidence: {min_conf:.3f}. "
         + ("Predictions below this were never logged, so precision-recall curves are truncated "
            "and AP is understated relative to a low-threshold (0.001) evaluation." if min_conf > 0.01 else ""),
         "", "## Detection (COCO AP, sampled frames)", "",
         "| Protocol | mAP@0.5 | mAP@0.5:0.95 | per-class AP@0.5 | GT boxes | predictions |",
         "|---|---|---|---|---|---|"]
    for name, r in det.items():
        pc = ", ".join(f"{k} {v:.3f}" for k, v in r["per_class_AP50"].items())
        L.append(f"| {name} | {r['mAP50']:.3f} | {r['mAP50_95']:.3f} | {pc} | {r['n_gt']} | {r['n_pred']} |")
    L += ["", "## Tracking (motmetrics, sampled frames, all cameras)", "",
          "| Protocol | MOTA | IDF1 | ID switches | false positives | misses | GT objects |",
          "|---|---|---|---|---|---|---|"]
    for name, r in trk.items():
        L.append(f"| {name} | {r['MOTA']:.3f} | {r['IDF1']:.3f} | {r['ID_switches']} | "
                 f"{r['false_positives']} | {r['misses']} | {r['gt_objects']} |")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n")
    OUT_JSON.write_text(json.dumps({"min_logged_confidence": min_conf, "detection": det, "tracking": trk}, indent=2))
    print("\n" + "\n".join(L) + f"\n\nSaved -> {OUT_MD}\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
