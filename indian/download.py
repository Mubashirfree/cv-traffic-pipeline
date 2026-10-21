"""Download a SAMPLE of IISc UVH-26, resize to 960 px wide JPEG, delete originals. Resumable.
  python -m indian.download --n-train 4000 --n-val 1200
"""
import argparse, json, os, random, shutil, sys, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from huggingface_hub import hf_hub_download, list_repo_files

from indian.common import UVH_DIR

REPO = "iisc-aim/UVH-26"
ANN = {"train": "UVH-26-Train/UVH-26-MV-Train.json", "val": "UVH-26-Val/UVH-26-MV-Val.json"}
IMG_PREFIX = {"train": "UVH-26-Train/images/", "val": "UVH-26-Val/images/"}
EXTS = (".png", ".jpg", ".jpeg")


def fetch_one(repo_file, out_path, tmp_dir, width, tries=3):
    if out_path.exists():
        return True
    last = None
    for _ in range(tries):
        try:
            p = hf_hub_download(REPO, repo_file, repo_type="dataset", local_dir=tmp_dir)
            with Image.open(p) as im:
                im = im.convert("RGB")
                w, h = im.size
                if w > width:
                    im = im.resize((width, round(h * width / w)), Image.LANCZOS)
                im.save(out_path, "JPEG", quality=92)
            os.remove(p)
            return True
        except Exception as e:
            last = e
    print(f"  FAILED {repo_file}: {last}")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=1200)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    UVH_DIR.mkdir(parents=True, exist_ok=True)
    print("listing repo files ...")
    try:
        files = list_repo_files(REPO, repo_type="dataset")
    except Exception as e:
        sys.exit(f"Could not list {REPO}: {e}\nIf gated/401: accept terms on the dataset page, then run `huggingface-cli login`.")

    tmp = Path(tempfile.mkdtemp(prefix="uvh_dl_", dir=UVH_DIR))
    rng = random.Random(args.seed)
    try:
        for split, n in (("train", args.n_train), ("val", args.n_val)):
            ann_path = hf_hub_download(REPO, ANN[split], repo_type="dataset", local_dir=UVH_DIR / "ann_raw")
            shutil.copy(ann_path, UVH_DIR / f"{split}.json")
            coco = json.load(open(UVH_DIR / f"{split}.json"))
            has_ann = {a["image_id"] for a in coco["annotations"]}
            annotated = {Path(i["file_name"]).name for i in coco["images"] if i["id"] in has_ann}
            repo_imgs = [f for f in files if f.startswith(IMG_PREFIX[split]) and f.lower().endswith(EXTS)
                         and Path(f).name in annotated]
            print(f"[{split}] {len(repo_imgs)} annotated images in repo; sampling {min(n, len(repo_imgs))}")
            rng.shuffle(repo_imgs)
            pick = sorted(repo_imgs[:n])
            out_dir = UVH_DIR / "images" / split
            out_dir.mkdir(parents=True, exist_ok=True)
            done = 0
            with ThreadPoolExecutor(args.workers) as ex:
                futs = [ex.submit(fetch_one, f, out_dir / (Path(f).stem + ".jpg"), tmp, args.width) for f in pick]
                for k, fu in enumerate(as_completed(futs), 1):
                    done += bool(fu.result())
                    if k % 100 == 0:
                        print(f"  {split}: {k}/{len(pick)}")
            print(f"[{split}] saved {done}/{len(pick)} -> {out_dir}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(UVH_DIR / "ann_raw", ignore_errors=True)

    cats = json.load(open(UVH_DIR / "train.json"))["categories"]
    print("\nCategories in the annotation file (check CLASS_MAP in indian/common.py):")
    for c in cats:
        print("  ", c["id"], c["name"])


if __name__ == "__main__":
    main()
