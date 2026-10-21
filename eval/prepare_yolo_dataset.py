"""
eval/prepare_yolo_dataset.py -- Phase: build a YOLO fine-tuning dataset from the 54 extra
UA-DETRAC train sequences, keeping the 6 evaluation sequences completely untouched.

- Reads XML from data/annotations_all/, images from data/raw/<seq>/imgNNNNN.jpg
- Classes: car, bus, truck (GT 'van' folded into truck, same as the eval scripts;
  GT 'others' is dropped -- consistent with the eval_fair.py finding that a merged
  single-class protocol did WORSE, so we keep the 3-class scheme the whole project uses).
- Split is done by SEQUENCE, not by frame, and stratified by weather condition, so
  train/val never share near-duplicate consecutive frames from the same clip.
- HARD SAFETY CHECK: raises immediately if any of the 6 evaluation sequences are found
  anywhere in the chosen training pool. This must never silently pass.
- Uses symlinks for images (not copies) to avoid duplicating several GB of JPEGs.

Run from the project root:
    python -m eval.prepare_yolo_dataset
    python -m eval.prepare_yolo_dataset --stride 5 --val-frac 0.15

Output: data/yolo_dataset/{images,labels}/{train,val}/, data/yolo_dataset/data.yaml
"""
import argparse
import json
import random
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ANN_DIR = ROOT / "data" / "annotations_all"
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "yolo_dataset"

# These must NEVER appear in the training/val pool -- they are the held-out eval set
# used by eval_fair.py and occlusion_check.py.
EVAL_SEQS = {"MVI_20011", "MVI_20012", "MVI_20032", "MVI_20033", "MVI_20034", "MVI_40992"}

CLASSES = ["car", "bus", "truck"]
GT_CLASS_MAP = {"car": "car", "bus": "bus", "van": "truck"}   # 'others' dropped
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}


def sequence_weather(xml_path):
    text = xml_path.read_text(errors="ignore")
    import re
    m = re.search(r'sequence_attribute[^/]*sence_weather="([a-zA-Z]*)"', text)
    return m.group(1) if m else "unknown"


def stratified_split(seqs_by_weather, val_frac, seed):
    rng = random.Random(seed)
    train, val = [], []
    for weather, seqs in seqs_by_weather.items():
        seqs = sorted(seqs)
        rng.shuffle(seqs)
        n_val = max(1, round(len(seqs) * val_frac))
        val += seqs[:n_val]
        train += seqs[n_val:]
    return sorted(train), sorted(val)


def image_size(seq):
    from PIL import Image
    imgs = sorted((RAW_DIR / seq).glob("img*.jpg"))
    if not imgs:
        return None
    with Image.open(imgs[0]) as im:
        return im.size  # (W, H)


def build_split(seqs, split_name, stride, dry_run=False):
    img_out = OUT_DIR / "images" / split_name
    lbl_out = OUT_DIR / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n_images, n_boxes, class_counts = 0, 0, defaultdict(int)
    for seq in seqs:
        size = image_size(seq)
        if size is None:
            print(f"[skip] {seq}: no images found in {RAW_DIR / seq}")
            continue
        W, H = size
        root = ET.parse(ANN_DIR / f"{seq}.xml").getroot()
        frames = root.findall("frame")
        max_frame = max(int(f.get("num")) for f in frames) if frames else 0
        by_num = {int(f.get("num")): f for f in frames}

        for fnum in range(1, max_frame + 1, stride):
            fr = by_num.get(fnum)
            img_path = RAW_DIR / seq / f"img{fnum:05d}.jpg"
            if fr is None or not img_path.exists():
                continue
            lines = []
            tl = fr.find("target_list")
            if tl is not None:
                for t in tl.findall("target"):
                    box, attr = t.find("box"), t.find("attribute")
                    if box is None:
                        continue
                    vtype = attr.get("vehicle_type", "others") if attr is not None else "others"
                    cls = GT_CLASS_MAP.get(vtype)
                    if cls is None:
                        continue
                    l, t_, w, h = (float(box.get(k)) for k in ("left", "top", "width", "height"))
                    cx, cy = (l + w / 2) / W, (t_ + h / 2) / H
                    nw, nh = w / W, h / H
                    if not (0 < cx < 1 and 0 < cy < 1 and nw > 0 and nh > 0):
                        continue
                    lines.append(f"{CLASS_ID[cls]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                    class_counts[cls] += 1
                    n_boxes += 1
            stem = f"{seq}_{fnum:05d}"
            if not dry_run:
                link = img_out / f"{stem}.jpg"
                if not link.exists():
                    link.symlink_to(img_path.resolve())
                (lbl_out / f"{stem}.txt").write_text("\n".join(lines))
            n_images += 1
    return {"n_images": n_images, "n_boxes": n_boxes, "class_counts": dict(class_counts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="compute stats only, write no files")
    args = ap.parse_args()

    xmls = sorted(ANN_DIR.glob("*.xml"))
    pool = [x.stem for x in xmls if x.stem not in EVAL_SEQS]
    assert not (set(pool) & EVAL_SEQS), "BUG: eval sequence leaked into pool"
    print(f"{len(xmls)} XML files found, {len(pool)} usable after excluding {len(EVAL_SEQS)} eval sequences")

    by_weather = defaultdict(list)
    for seq in pool:
        by_weather[sequence_weather(ANN_DIR / f"{seq}.xml")].append(seq)
    for w, seqs in sorted(by_weather.items()):
        print(f"  {w}: {len(seqs)} sequences")

    train_seqs, val_seqs = stratified_split(by_weather, args.val_frac, args.seed)

    # HARD SAFETY CHECK -- must never pass silently
    leaked = (set(train_seqs) | set(val_seqs)) & EVAL_SEQS
    if leaked:
        raise RuntimeError(f"REFUSING TO CONTINUE: eval sequence(s) {leaked} ended up in the "
                           "training pool. This would invalidate every before/after comparison.")
    missing_raw = [s for s in train_seqs + val_seqs if not (RAW_DIR / s).exists()]
    if missing_raw:
        print(f"[warn] {len(missing_raw)} sequences have no extracted images yet: {missing_raw[:5]}...")

    print(f"\ntrain: {len(train_seqs)} sequences, val: {len(val_seqs)} sequences")
    print("train seqs:", train_seqs)
    print("val seqs:  ", val_seqs)

    train_stats = build_split(train_seqs, "train", args.stride, args.dry_run)
    val_stats = build_split(val_seqs, "val", args.stride, args.dry_run)

    print(f"\ntrain: {train_stats['n_images']} images, {train_stats['n_boxes']} boxes, "
          f"{train_stats['class_counts']}")
    print(f"val:   {val_stats['n_images']} images, {val_stats['n_boxes']} boxes, "
          f"{val_stats['class_counts']}")

    if args.dry_run:
        print("\n[dry run] no files written")
        return

    yaml_text = (f"path: {OUT_DIR}\ntrain: images/train\nval: images/val\n"
                f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    (OUT_DIR / "data.yaml").write_text(yaml_text)
    manifest = {"eval_seqs_excluded": sorted(EVAL_SEQS), "train_seqs": train_seqs, "val_seqs": val_seqs,
                "stride": args.stride, "train_stats": train_stats, "val_stats": val_stats}
    (OUT_DIR / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {OUT_DIR / 'data.yaml'}\nWrote {OUT_DIR / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
