"""
eval/clip_gt_from_xml.py -- derive per-clip vehicle ground truth from UA-DETRAC XML.

For every 3 s sub-clip in data/clips/manifest.json, reads the human-verified XML
annotations (independent of our YOLO detector) and reports, over the frames in the
clip window:
    bus/car/van/others : median and max number visible at the same time
    total_med/total_max: all vehicles visible at the same time
    near_bottom        : median (over frames) of the lowest box bottom edge / image height
                         (closer to 1.0 = a vehicle is right at the bottom of the frame = near camera)
    near_height        : median of the tallest box height / image height (large = vehicle close)

NOTE: UA-DETRAC annotates vehicles only. Pedestrians are NOT in the XML.

Run from the project root:
    python -m eval.clip_gt_from_xml
Outputs: eval/clip_gt_from_xml.csv (and a printed table)
"""
import csv
import json
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ANN_DIR = ROOT / "data" / "annotations"
RAW_DIR = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "clips" / "manifest.json"
OUT_CSV = ROOT / "eval" / "clip_gt_from_xml.csv"

SOURCE_FPS = 25
FRAME_STEP = 5            # look at every 5th frame in the window (5 samples/s)
TYPES = ["car", "van", "bus", "others"]


def image_height(seq):
    """Real frame height from the first raw image; UA-DETRAC standard is 540."""
    try:
        import cv2
        img = cv2.imread(str(RAW_DIR / seq / "img00001.jpg"))
        if img is not None:
            return img.shape[0]
    except ImportError:
        pass
    return 540


def load_sequence(seq):
    """-> {frame_num: [(vehicle_type, left, top, width, height), ...]}"""
    tree = ET.parse(ANN_DIR / f"{seq}.xml")
    frames = defaultdict(list)
    for fr in tree.getroot().iter("frame"):
        num = int(fr.get("num"))
        for t in fr.iter("target"):
            box, attr = t.find("box"), t.find("attribute")
            if box is None:
                continue
            vtype = attr.get("vehicle_type", "others") if attr is not None else "others"
            frames[num].append((vtype, float(box.get("left")), float(box.get("top")),
                                float(box.get("width")), float(box.get("height"))))
        frames.setdefault(num, [])
    return frames


def load_manifest():
    with open(MANIFEST) as f:
        m = json.load(f)
    if isinstance(m, dict):
        m = m.get("clips", list(m.values()))
    return m


def med(v):
    return statistics.median(v) if v else 0.0


def main():
    manifest = load_manifest()
    cache, heights, rows = {}, {}, []

    for c in manifest:
        seq = c["camera_id"]
        if seq not in cache:
            cache[seq] = load_sequence(seq)
            heights[seq] = image_height(seq)
        frames, H = cache[seq], heights[seq]

        first = int(round(c["start_ts"] * SOURCE_FPS)) + 1
        last = int(round(c["end_ts"] * SOURCE_FPS))          # exclusive-ish end of the 3 s window
        sample = [n for n in range(first, last + 1, FRAME_STEP) if n in frames]

        per_type = {t: [] for t in TYPES}
        totals, bottoms, heights_frac = [], [], []
        for n in sample:
            boxes = frames[n]
            for t in TYPES:
                per_type[t].append(sum(1 for b in boxes if b[0] == t))
            totals.append(len(boxes))
            if boxes:
                bottoms.append(max((b[2] + b[4]) for b in boxes) / H)
                heights_frac.append(max(b[4] for b in boxes) / H)
            else:
                bottoms.append(0.0)
                heights_frac.append(0.0)

        row = {"clip_id": c["clip_id"], "camera_id": seq, "frames_used": len(sample)}
        for t in TYPES:
            row[f"{t}_med"] = med(per_type[t])
            row[f"{t}_max"] = max(per_type[t]) if per_type[t] else 0
        row["total_med"] = med(totals)
        row["total_max"] = max(totals) if totals else 0
        row["near_bottom"] = round(med(bottoms), 2)
        row["near_height"] = round(med(heights_frac), 2)
        rows.append(row)

    rows.sort(key=lambda r: r["clip_id"])
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    cols = ["clip_id", "bus_med", "bus_max", "car_med", "car_max", "van_med",
            "total_med", "total_max", "near_bottom", "near_height"]
    print(" ".join(f"{c:<14}" if i == 0 else f"{c:>11}" for i, c in enumerate(cols)))
    for r in rows:
        print(" ".join(f"{r[c]:<14}" if i == 0 else f"{r[c]:>11}" for i, c in enumerate(cols)))
    print(f"\n{len(rows)} clips -> {OUT_CSV}")


if __name__ == "__main__":
    main()
