# Fine-tune before/after (same 6 held-out eval sequences, same protocol-fair scoring)

Both rows use: ignored-region masking + car/bus/truck (van->truck) mapping -- the D1/T1 protocol from eval_fair.py. Only the detector weights differ.

## Detection (COCO AP)

| Model | mAP@0.5 | mAP@0.5:0.95 | car AP | bus AP | truck AP |
|---|---|---|---|---|---|
| Baseline (COCO-pretrained YOLOv8n) | 0.314 | 0.122 | 0.209 | 0.704 | 0.030 |
| Fine-tuned (54 UA-DETRAC sequences) | 0.614 | 0.347 | 0.523 | 0.818 | 0.500 |

## Tracking (motmetrics)

| Model | MOTA | IDF1 | ID switches | false positives | misses |
|---|---|---|---|---|---|
| Baseline | 0.220 | 0.446 | 31 | 480 | 2508 |
| Fine-tuned | 0.483 | 0.629 | 173 | 173 | 1654 |
