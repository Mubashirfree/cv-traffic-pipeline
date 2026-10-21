"""
eval/occlusion_check.py -- Phase 5: does ByteTrack keep the same ID across occlusion?

An "occlusion event" for a GT object = a run of consecutive SAMPLED frames where its box
overlaps another GT box above IOU_THRESH (one vehicle passing behind/beside another).
For each event we take the predicted track_id matched (by IoU) to that GT object on the
frame right before the event starts and right after it ends, and call it:
    survived  : same pred track_id before and after
    switched  : different (but present) pred track_id before and after
    lost      : no matched prediction on one or both sides
Events touching the start/end of a clip (no "before" or "after" frame) are skipped.

This directly tests the ByteTrack design choice cited in the report: does its motion-based
re-identification actually keep IDs stable through occlusion, on our footage?

Run from the project root:
    python -m eval.occlusion_check                                    # baseline detections
    python -m eval.occlusion_check --det-dir data/detections_finetuned --label finetuned
Outputs: report/figures/occlusion_check[_<label>].md, eval/occlusion_events[_<label>].json
"""
import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ANN_DIR = ROOT / "data" / "annotations"
DET_DIR = ROOT / "data" / "detections"          # overridden in main() via --det-dir

CAMERAS = ["MVI_20011", "MVI_20012", "MVI_20032", "MVI_20033", "MVI_20034", "MVI_40992"]
VID_STRIDE = 10
IOU_THRESH = 0.15     # GT-GT overlap counted as "occluded" (UA-DETRAC vehicles are dense; a
                       # high threshold like 0.5 would miss partial/side occlusion)
MATCH_THRESH = 0.3    # GT-to-prediction IoU to call it "the same object"


def iou(a, b):
    ax1, ay1, aw, ah = a; bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def load_camera(camera):
    root = ET.parse(ANN_DIR / f"{camera}.xml").getroot()
    gt, frames = {}, []
    for fr in root.findall("frame"):
        fnum = int(fr.get("num")); frames.append(fnum)
        tl = fr.find("target_list")
        objs = {}
        if tl is not None:
            for t in tl.findall("target"):
                box = t.find("box")
                if box is None:
                    continue
                objs[int(t.get("id"))] = [float(box.get(k)) for k in ("left", "top", "width", "height")]
        gt[fnum] = objs
    with open(DET_DIR / f"{camera}.json") as f:
        logs = json.load(f)
    preds = {}
    for idx, frame in enumerate(logs):
        fnum = idx * VID_STRIDE + 1
        preds[fnum] = [(d["track_id"], [d["bbox_px"][0], d["bbox_px"][1],
                                        d["bbox_px"][2] - d["bbox_px"][0], d["bbox_px"][3] - d["bbox_px"][1]])
                       for d in frame["detections"]]
    max_frame = max(frames) if frames else 0
    sampled = sorted(f for f in range(1, max_frame + 1, VID_STRIDE) if f in gt)
    return gt, preds, sampled


def matched_track_id(gt_box, preds_here):
    best_id, best_iou = None, MATCH_THRESH
    for tid, box in preds_here:
        v = iou(gt_box, box)
        if v > best_iou:
            best_iou, best_id = v, tid
    return best_id


def occluded_gt_ids(objs):
    ids = list(objs)
    occ = set()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if iou(objs[ids[i]], objs[ids[j]]) >= IOU_THRESH:
                occ.add(ids[i]); occ.add(ids[j])
    return occ


def find_events(gt, preds, sampled):
    """-> list of dicts, one per occlusion run, for every GT object."""
    all_ids = sorted({oid for f in sampled for oid in gt[f]})
    events = []
    for oid in all_ids:
        timeline = []
        for f in sampled:
            if oid not in gt[f]:
                timeline.append((f, None, None))
                continue
            occ = oid in occluded_gt_ids(gt[f])
            pid = matched_track_id(gt[f][oid], preds.get(f, []))
            timeline.append((f, occ, pid))
        i = 0
        while i < len(timeline):
            f, occ, pid = timeline[i]
            if occ:
                j = i
                while j < len(timeline) and timeline[j][1]:
                    j += 1
                before = timeline[i - 1] if i - 1 >= 0 else None
                after = timeline[j] if j < len(timeline) else None
                if before and before[1] is False and after and after[1] is False:
                    events.append({
                        "gt_id": oid, "start_frame": timeline[i][0], "end_frame": timeline[j - 1][0],
                        "n_frames": j - i, "pred_before": before[2], "pred_after": after[2]})
                i = j
            else:
                i += 1
    return events


def classify(events):
    out = []
    for e in events:
        b, a = e["pred_before"], e["pred_after"]
        status = "lost" if (b is None or a is None) else ("survived" if b == a else "switched")
        out.append({**e, "status": status})
    return out


def main():
    global DET_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-dir", default=str(DET_DIR), help="directory of per-camera detection JSON logs")
    ap.add_argument("--label", default="", help="suffix for output filenames, e.g. 'finetuned'")
    args = ap.parse_args()
    DET_DIR = Path(args.det_dir)
    suffix = f"_{args.label}" if args.label else ""
    out_md = ROOT / "report" / "figures" / f"occlusion_check{suffix}.md"
    out_json = ROOT / "eval" / f"occlusion_events{suffix}.json"

    all_events, per_cam = [], {}
    for cam in CAMERAS:
        gt, preds, sampled = load_camera(cam)
        ev = classify(find_events(gt, preds, sampled))
        for e in ev:
            e["camera"] = cam
        all_events += ev
        counts = {"survived": 0, "switched": 0, "lost": 0}
        for e in ev:
            counts[e["status"]] += 1
        per_cam[cam] = counts
        print(f"{cam}: {len(ev)} occlusion events -> {counts}")

    total = {"survived": 0, "switched": 0, "lost": 0}
    for c in per_cam.values():
        for k in total:
            total[k] += c[k]
    n = sum(total.values())
    rate = total["survived"] / n if n else float("nan")

    L = ["# Occlusion check: does the tracker keep the same ID across occlusion?", "",
         f"IoU >= {IOU_THRESH} between two GT boxes on a sampled frame = occluded. "
         f"For each occlusion run we compare the matched prediction ID on the frame right "
         f"before it starts and right after it ends (matched to GT by IoU >= {MATCH_THRESH}).", "",
         f"**{n} occlusion events found. ID survived in {total['survived']} "
         + (f"({rate:.1%}).**" if n else "(n/a).**"), "",
         "| Camera | events | survived | switched | lost (no match one side) |",
         "|---|---|---|---|---|"]
    for cam, c in per_cam.items():
        tot = sum(c.values())
        L.append(f"| {cam} | {tot} | {c['survived']} | {c['switched']} | {c['lost']} |")
    L.append(f"| **overall** | **{n}** | **{total['survived']}** | **{total['switched']}** | **{total['lost']}** |")
    if n == 0:
        L += ["", f"No qualifying events at IoU >= {IOU_THRESH} and stride {VID_STRIDE} "
                  "(occlusions this brief may fall between sampled frames). Consider lowering "
                  "IOU_THRESH or evaluating at native frame rate for this check specifically."]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L) + "\n")
    out_json.write_text(json.dumps(all_events, indent=2))
    print("\n" + "\n".join(L) + f"\n\nSaved -> {out_md}\nSaved -> {out_json}")


if __name__ == "__main__":
    main()
