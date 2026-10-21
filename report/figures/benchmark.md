# Benchmark

Hardware: NVIDIA GeForce RTX 3050 Laptop GPU / torch 2.14.0+cu130 / Python 3.14.4

## Detection throughput (YOLOv8n, consecutive native-rate frames)

| Setting | FPS | ms/frame (mean) | median | p95 | preprocess | inference | postprocess | x real-time (25 fps) |
|---|---|---|---|---|---|---|---|---|
| GPU detect only | 118.8 | 8.4 | 7.6 | 11.3 | 1.0 | 5.5 | 1.6 | 4.75 |
| GPU detect + ByteTrack | 90.6 | 11.0 | 9.7 | 14.8 | 1.1 | 5.8 | 1.7 | 3.62 |
| CPU detect only | 34.1 | 29.3 | 29.1 | 33.5 | 1.5 | 26.6 | 0.8 | 1.36 |
| CPU detect + ByteTrack | 32.1 | 31.2 | 31.3 | 35.4 | 1.4 | 25.7 | 0.9 | 1.28 |

## Query latency (top_k=5, 7 queries x 10 repeats)

One-off cold start (imports + CLIP load): 1.2 s

| Embedding | total mean (ms) | median | p95 | text encode, median (ms) | Qdrant + overhead, median (ms) |
|---|---|---|---|---|---|
| frame | 13.6 | 10.7 | 27.0 | 12.7 | -2.0 |
| crop | 12.9 | 9.2 | 25.7 | 7.4 | 1.9 |
