"""UVH-26 COCO json -> YOLO dataset (6 classes). train=official train sample; val/test = 25%/75% of official val sample.
  python -m indian.to_yolo
"""
import hashlib, json, os
from collections import Counter
from pathlib import Path

from PIL import Image

from indian.common import UVH_DIR, YOLO_DIR, CLASSES, CLASS_MAP, norm_name


def split_of(official, stem):
    if official == "train":
        return "train"
    h = int(hashlib.md5(stem.encode()).hexdigest(), 16) % 4
    return "val" if h == 0 else "test"


def main():
    counts = {s: Counter() for s in ("train", "val", "test")}
    n_img = Counter()
    for official in ("train", "val"):
        coco = json.load(open(UVH_DIR / f"{official}.json"))
        cat_name = {c["id"]: norm_name(c["name"]) for c in coco["categories"]}
        unknown = sorted({n for n in cat_name.values() if n not in CLASS_MAP})
        if unknown:
            raise SystemExit(f"Unmapped categories {unknown}: add them to CLASS_MAP in indian/common.py")
        by_img = {}
        for a in coco["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        for im in coco["images"]:
            stem = Path(im["file_name"]).stem
            src = UVH_DIR / "images" / official / f"{stem}.jpg"
            if not src.exists():
                continue
            sp = split_of(official, stem)
            W, H = im["width"], im["height"]
            if not W or not H:
                with Image.open(src) as t:
                    W, H = t.size
            lines = []
            for a in by_img.get(im["id"], []):
                target = CLASS_MAP[cat_name[a["category_id"]]]
                if target is None:
                    continue
                x, y, w, h = a["bbox"]
                x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                k = CLASSES.index(target)
                lines.append(f"{k} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
                counts[sp][target] += 1
            img_dir = YOLO_DIR / "images" / sp
            lab_dir = YOLO_DIR / "labels" / sp
            img_dir.mkdir(parents=True, exist_ok=True)
            lab_dir.mkdir(parents=True, exist_ok=True)
            dst = img_dir / src.name
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.symlink(src.resolve(), dst)
            (lab_dir / f"{stem}.txt").write_text("\n".join(lines))
            n_img[sp] += 1

    yaml = f"path: {YOLO_DIR.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
    yaml += "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES))
    (YOLO_DIR / "data.yaml").write_text(yaml)
    print("images per split:", dict(n_img))
    for sp in ("train", "val", "test"):
        print(f"{sp:5s} boxes:", dict(counts[sp]))
    print("wrote", YOLO_DIR / "data.yaml")


if __name__ == "__main__":
    main()
