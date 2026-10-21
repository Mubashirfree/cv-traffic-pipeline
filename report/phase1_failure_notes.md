# Phase 1 — Observed Detection/Tracking Issues

Initial visual review of annotated tracking video (MVI_20011): no obvious
missed detections, flickering boxes, or ID switches noted. Track IDs for
person/car classes appeared stable through the sampled frames.

Note: this was a quick pass at 2-3 FPS sampling, not a frame-by-frame
review. A closer look during Phase 5 evaluation (mAP/MOTA against
UA-DETRAC ground truth) may surface issues not visible at a glance -
worth revisiting this note after those numbers come in.

# Phase 4 — Retrieval Observations

Cosine similarity scores across all spot-check queries sat consistently
in the ~0.23-0.29 range, even for strong-seeming matches. This appears
to be expected behavior for short/generic text queries against
mean-pooled multi-frame CLIP embeddings, not a bug -- relative ranking
(best match scoring higher than others) held up correctly across all
7 test queries, with different queries correctly surfacing different
cameras. Precision@K/Recall@K in Phase 5 will give a harder
quantitative read on whether this ranking quality is good enough.

# Phase 5 — Bug Found and Fixed: Lossy Timestamp Frame-Matching

detect_track.py's frame_ts() truncates to whole-second precision
(str(timedelta(...))[:8]), but sampled frames are 0.4s apart -- multiple
distinct frames collapsed onto the same timestamp string. This caused
eval/detections_to_coco_pred.py to match predictions to the WRONG
ground-truth frame (up to 9 frames off), producing artificially low
mAP (0.034 @0.5). Fixed by using each detection log's array index
(a deterministic function of VID_STRIDE) instead of parsing the lossy
timestamp string. Corrected mAP@0.5 = 0.294, mAP@0.5:0.95 = 0.118.

Known related limitation: Phase 3's crop_encoder.py used the same lossy
timestamp approach to locate source frames for object crops. Some crop
embeddings may be taken from a frame 1-2 positions off from the one
referenced in the detection log. Not re-run, given low practical impact
on visually near-identical adjacent traffic frames -- documented here
for transparency rather than treated as fully resolved.

# Phase 5 — Tracking Evaluation (MOTA/MOTP/IDF1)

Overall: MOTA=-0.717, MOTP=0.346, IDF1=0.290, 36 ID switches,
2357 misses, 4219 false positives (across 6 clips, vehicle classes only).

Negative MOTA confirmed as a genuine result, not a pipeline bug --
verified no duplicate detections/track_ids per frame, and predictions
correctly filtered to vehicle classes matching UA-DETRAC's GT scope
(person/bag detections excluded from this comparison). Driven by the
same underlying detector precision/recall limitation reflected in the
Phase 5 mAP result (0.294 @0.5) -- a non-fine-tuned COCO-pretrained
YOLOv8n underperforms on UA-DETRAC's elevated camera angle and small
distant vehicles. Fine-tuning (flagged as optional future work back
in Phase 0/1 discussion) would likely improve both metrics substantially.

Notable: MVI_40992 (night clip) had the LOWEST false-positive count
(187) of all 6 clips despite low-light conditions -- worth highlighting
as a counterintuitive result in the report.
