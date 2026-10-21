"""COCO-style detection eval on the held-out UVH-26 test split, same protocol for every model.
  python -m indian.eval_detection --model yolov8n.pt --label coco_baseline --idmap coco
"""
import argparse, io, json, contextlib

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from indian.common import YOLO_DIR, CLASSES, COCO_TO_OURS, DETRAC_TO_OURS, FIG_DIR, EVAL_DIR


def build_gt(split):
    imgs, anns = [], []
    aid = 1
    for i, p in enumerate(sorted((YOLO_DIR / "images" / split).glob("*.jpg"))):
        with Image.open(p) as im:
            W, H = im.size
        imgs.append({"id": i, "file_name": p.name, "width": W, "height": H})
        lab = YOLO_DIR / "labels" / split / f"{p.stem}.txt"
        for ln in lab.read_text().split("\n") if lab.exists() else []:
            if not ln.strip():
                continue
            c, cx, cy, w, h = ln.split()
            w_, h_ = float(w) * W, float(h) * H
            anns.append({"id": aid, "image_id": i, "category_id": int(c),
                         "bbox": [float(cx) * W - w_ / 2, float(cy) * H - h_ / 2, w_, h_],
                         "area": w_ * h_, "iscrowd": 0})
            aid += 1
    cats = [{"id": i, "name": n} for i, n in enumerate(CLASSES)]
    return {"images": imgs, "annotations": anns, "categories": cats}


def coco_eval(gt_dict, preds, cat_ids):
    out = {}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = gt_dict
        gt.createIndex()
        dt = gt.loadRes(preds) if preds else None
    if dt is None:
        return {c: (0.0, 0.0) for c in cat_ids}
    for c in cat_ids:
        with contextlib.redirect_stdout(io.StringIO()):
            E = COCOeval(gt, dt, "bbox")
            E.params.catIds = [c]
            E.evaluate(); E.accumulate(); E.summarize()
        out[c] = (E.stats[1], E.stats[0])
    return out


def predict_all(model_path, idmap, gt_dict, split, imgsz, batch=16):
    from ultralytics import YOLO
    model = YOLO(model_path)
    mapping = {"coco": COCO_TO_OURS, "detrac": DETRAC_TO_OURS, "ours": {i: i for i in range(len(CLASSES))}}[idmap]
    paths = [str(YOLO_DIR / "images" / split / im["file_name"]) for im in gt_dict["images"]]
    preds = []
    for s in range(0, len(paths), batch):
        res = model.predict(paths[s:s + batch], imgsz=imgsz, conf=0.001, iou=0.6, max_det=300,
                            verbose=False, device=0)
        for k, r in enumerate(res):
            for b, c, sc in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                if int(c) in mapping:
                    x1, y1, x2, y2 = b
                    preds.append({"image_id": s + k, "category_id": mapping[int(c)],
                                  "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(sc)})
    return preds, sorted(set(mapping.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--idmap", choices=["coco", "detrac", "ours"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()

    gt = build_gt(args.split)
    print(f"{len(gt['images'])} images, {len(gt['annotations'])} GT boxes in '{args.split}'")
    preds, covered = predict_all(args.model, args.idmap, gt, args.split, args.imgsz)
    res = coco_eval(gt, preds, list(range(len(CLASSES))))

    rows, cov_ap50, cov_ap, all_ap50, all_ap = [], [], [], [], []
    for c, name in enumerate(CLASSES):
        ap50, ap = res[c]
        if ap50 < 0:
            rows.append((name, "no GT", "no GT")); continue
        if c not in covered:
            rows.append((name, "n/a (class not in model)", "n/a")); all_ap50.append(0.0); all_ap.append(0.0); continue
        rows.append((name, f"{ap50:.3f}", f"{ap:.3f}"))
        cov_ap50.append(ap50); cov_ap.append(ap); all_ap50.append(ap50); all_ap.append(ap)

    summary = {
        "label": args.label, "model": args.model, "split": args.split, "n_images": len(gt["images"]),
        "n_gt": len(gt["annotations"]),
        "mAP50_covered": float(np.mean(cov_ap50)) if cov_ap50 else 0.0,
        "mAP50_95_covered": float(np.mean(cov_ap)) if cov_ap else 0.0,
        "mAP50_all6": float(np.mean(all_ap50)), "mAP50_95_all6": float(np.mean(all_ap)),
        "per_class": {n: {"AP50": a, "AP50_95": b} for n, a, b in rows},
    }
    EVAL_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / f"indian_det_{args.label}.json").write_text(json.dumps(summary, indent=1))
    md = [f"### {args.label}  ({args.model}) on UVH-26 '{args.split}' ({summary['n_images']} imgs, {summary['n_gt']} boxes)", "",
          "| class | AP@0.5 | AP@0.5:0.95 |", "|---|---|---|"]
    md += [f"| {n} | {a} | {b} |" for n, a, b in rows]
    md += ["", f"mAP@0.5 (covered classes): **{summary['mAP50_covered']:.3f}**, mAP@0.5:0.95: {summary['mAP50_95_covered']:.3f}  ",
           f"mAP@0.5 (all 6, uncovered=0): {summary['mAP50_all6']:.3f}, mAP@0.5:0.95: {summary['mAP50_95_all6']:.3f}", ""]
    (FIG_DIR / f"indian_det_{args.label}.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
