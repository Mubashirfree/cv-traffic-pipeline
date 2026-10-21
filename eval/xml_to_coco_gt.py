"""
Phase 5: Convert UA-DETRAC XML ground truth into COCO-format annotations,
restricted to the frames we actually sampled (matching detect_track.py's
2-3 FPS stride) and to classes our detector also predicts.
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).parent.parent
CAMERAS = ["MVI_20011", "MVI_20012", "MVI_20032", "MVI_20033", "MVI_20034", "MVI_40992"]

# UA-DETRAC vehicle_type -> our detector's class names
CLASS_MAP = {"car": "car", "bus": "bus", "van": "truck", "others": None}
CATEGORIES = [{"id": 1, "name": "car"}, {"id": 2, "name": "bus"}, {"id": 3, "name": "truck"}]
CAT_NAME_TO_ID = {c["name"]: c["id"] for c in CATEGORIES}

SOURCE_FPS = 25
VID_STRIDE = 10  # must match detect_track.py


def sampled_frame_numbers(xml_root):
    """Frame numbers we actually ran detection on (1-indexed, every VID_STRIDE-th)."""
    all_frames = [int(f.get("num")) for f in xml_root.findall("frame")]
    max_frame = max(all_frames) if all_frames else 0
    return set(range(1, max_frame + 1, VID_STRIDE))


def build_coco_gt():
    images, annotations = [], []
    img_id, ann_id = 0, 0

    for camera_id in CAMERAS:
        xml_path = ROOT / "data" / "annotations" / f"{camera_id}.xml"
        if not xml_path.exists():
            print(f"WARNING: no XML for {camera_id}, skipping")
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()
        valid_frames = sampled_frame_numbers(root)

        for frame in root.findall("frame"):
            fnum = int(frame.get("num"))
            if fnum not in valid_frames:
                continue

            img_id += 1
            images.append({
                "id": img_id,
                "file_name": f"{camera_id}/img{fnum:05d}.jpg",
                "camera_id": camera_id,
                "frame_num": fnum,
            })

            target_list = frame.find("target_list")
            if target_list is None:
                continue
            for target in target_list.findall("target"):
                box = target.find("box")
                attr = target.find("attribute")
                if box is None or attr is None:
                    continue
                vtype = attr.get("vehicle_type", "others")
                cls_name = CLASS_MAP.get(vtype)
                if cls_name is None:
                    continue

                x = float(box.get("left"))
                y = float(box.get("top"))
                w = float(box.get("width"))
                h = float(box.get("height"))

                ann_id += 1
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": CAT_NAME_TO_ID[cls_name],
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                })

    coco = {"images": images, "annotations": annotations, "categories": CATEGORIES}
    out_path = ROOT / "eval" / "gt_coco.json"
    with open(out_path, "w") as f:
        json.dump(coco, f)
    print(f"Wrote {len(images)} images, {len(annotations)} ground-truth boxes -> {out_path}")
    return coco


if __name__ == "__main__":
    build_coco_gt()
