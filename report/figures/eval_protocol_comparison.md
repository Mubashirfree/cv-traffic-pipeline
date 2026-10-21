# Evaluation protocol comparison (same detections, different scoring rules)

Lowest logged detection confidence: 0.100. Predictions below this were never logged, so precision-recall curves are truncated and AP is understated relative to a low-threshold (0.001) evaluation.

## Detection (COCO AP, sampled frames)

| Protocol | mAP@0.5 | mAP@0.5:0.95 | per-class AP@0.5 | GT boxes | predictions |
|---|---|---|---|---|---|
| D0 original (3 classes, no ignored regions) | 0.294 | 0.118 | car 0.148, bus 0.704, truck 0.030 | 3851 | 5709 |
| D1 + ignored regions | 0.314 | 0.122 | car 0.209, bus 0.704, truck 0.030 | 3851 | 1844 |
| D2 + single 'vehicle' class (standard protocol) | 0.273 | 0.069 | vehicle 0.273 | 3872 | 1844 |

## Tracking (motmetrics, sampled frames, all cameras)

| Protocol | MOTA | IDF1 | ID switches | false positives | misses | GT objects |
|---|---|---|---|---|---|---|
| T0 original (no ignored regions) | -0.717 | 0.290 | 36 | 4219 | 2357 | 3851 |
| T1 + ignored regions | 0.222 | 0.448 | 31 | 480 | 2487 | 3851 |
