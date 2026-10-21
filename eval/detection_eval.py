"""
Phase 5: Compute mAP@0.5 and mAP@0.5:0.95 using pycocotools.
"""

import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).parent.parent


def run_eval():
    coco_gt = COCO(str(ROOT / "eval" / "gt_coco.json"))
    coco_dt = coco_gt.loadRes(str(ROOT / "eval" / "pred_coco.json"))

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map_50_95 = coco_eval.stats[0]
    map_50 = coco_eval.stats[1]

    results = {"mAP@0.5": round(float(map_50), 4), "mAP@0.5:0.95": round(float(map_50_95), 4)}
    with open(ROOT / "eval" / "detection_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n", results)


if __name__ == "__main__":
    run_eval()
