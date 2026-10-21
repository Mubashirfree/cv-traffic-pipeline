"""
Phase 5 diagnostic: draw GT (green) vs predicted (red) boxes on one
frame to check for coordinate/scale/frame-matching bugs.
"""

import json
from pathlib import Path

import cv2

ROOT = Path(__file__).parent.parent

with open(ROOT / "eval" / "gt_coco.json") as f:
    gt = json.load(f)
with open(ROOT / "eval" / "pred_coco.json") as f:
    preds = json.load(f)

# pick the first image that has both GT and predictions
gt_by_img = {}
for ann in gt["annotations"]:
    gt_by_img.setdefault(ann["image_id"], []).append(ann["bbox"])

pred_by_img = {}
for p in preds:
    pred_by_img.setdefault(p["image_id"], []).append(p["bbox"])

target_id = None
for im in gt["images"]:
    if im["id"] in gt_by_img and im["id"] in pred_by_img:
        target_id = im["id"]
        target_meta = im
        break

print("Using image_id:", target_id, target_meta)

img_path = ROOT / "data" / "raw" / target_meta["file_name"]
img = cv2.imread(str(img_path))
print("Loaded image shape:", img.shape, "from", img_path)

for (x, y, w, h) in gt_by_img[target_id]:
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)  # green = GT

for (x, y, w, h) in pred_by_img[target_id]:
    cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)  # red = pred

out_path = ROOT / "report" / "figures" / "gt_vs_pred_overlay.jpg"
cv2.imwrite(str(out_path), img)
print("Saved ->", out_path)
