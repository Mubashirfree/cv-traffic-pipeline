### Integration test

- checks passed: 47/47
- /search latency (warm, incl. CLIP text encode + Qdrant): median 20 ms, p95 34 ms, n=42

| check | result | detail | ms |
|---|---|---|---|
| GET /health | PASS |  |  |
| search[detrac/frame] heavy traffic with many vehicles | PASS | 200 n=5 | 27 |
| search[detrac/crop] heavy traffic with many vehicles | PASS | 200 n=5 | 23 |
| search[detrac/frame] multiple buses on the road | PASS | 200 n=5 | 24 |
| search[detrac/crop] multiple buses on the road | PASS | 200 n=5 | 30 |
| search[detrac/frame] an almost empty road with hardly any traffic | PASS | 200 n=5 | 28 |
| search[detrac/crop] an almost empty road with hardly any traffic | PASS | 200 n=5 | 24 |
| search[detrac/frame] a night road with several cars | PASS | 200 n=5 | 23 |
| search[detrac/crop] a night road with several cars | PASS | 200 n=5 | 22 |
| search[detrac/frame] cars close to the camera | PASS | 200 n=5 | 24 |
| search[detrac/crop] cars close to the camera | PASS | 200 n=5 | 25 |
| search[detrac/frame] a truck on the road | PASS | 200 n=5 | 20 |
| search[detrac/crop] a truck on the road | PASS | 200 n=5 | 20 |
| search[detrac/frame] a crowded junction | PASS | 200 n=5 | 21 |
| search[detrac/crop] a crowded junction | PASS | 200 n=5 | 18 |
| search[indian/frame] many motorcycles and scooters on the road | PASS | 200 n=5 | 19 |
| search[indian/crop] many motorcycles and scooters on the road | PASS | 200 n=5 | 20 |
| search[indian/frame] auto rickshaws on the road | PASS | 200 n=5 | 20 |
| search[indian/crop] auto rickshaws on the road | PASS | 200 n=5 | 20 |
| search[indian/frame] a bus on the road | PASS | 200 n=5 | 19 |
| search[indian/crop] a bus on the road | PASS | 200 n=5 | 19 |
| search[indian/frame] heavy traffic with many vehicles | PASS | 200 n=5 | 19 |
| search[indian/crop] heavy traffic with many vehicles | PASS | 200 n=5 | 20 |
| search[indian/frame] trucks on the road | PASS | 200 n=5 | 34 |
| search[indian/crop] trucks on the road | PASS | 200 n=5 | 18 |
| search[indian/frame] bicycles on the road | PASS | 200 n=5 | 34 |
| search[indian/crop] bicycles on the road | PASS | 200 n=5 | 34 |
| search[indian/frame] an almost empty road | PASS | 200 n=5 | 20 |
| search[indian/crop] an almost empty road | PASS | 200 n=5 | 19 |
| search[indian_video/frame] many motorcycles and scooters on the road | PASS | 200 n=5 | 35 |
| search[indian_video/crop] many motorcycles and scooters on the road | PASS | 200 n=5 | 33 |
| search[indian_video/frame] auto rickshaws on the road | PASS | 200 n=5 | 34 |
| search[indian_video/crop] auto rickshaws on the road | PASS | 200 n=5 | 18 |
| search[indian_video/frame] a bus on the road | PASS | 200 n=5 | 19 |
| search[indian_video/crop] a bus on the road | PASS | 200 n=5 | 18 |
| search[indian_video/frame] heavy traffic with many vehicles | PASS | 200 n=5 | 18 |
| search[indian_video/crop] heavy traffic with many vehicles | PASS | 200 n=5 | 19 |
| search[indian_video/frame] trucks on the road | PASS | 200 n=5 | 18 |
| search[indian_video/crop] trucks on the road | PASS | 200 n=5 | 18 |
| search[indian_video/frame] bicycles on the road | PASS | 200 n=5 | 19 |
| search[indian_video/crop] bicycles on the road | PASS | 200 n=5 | 18 |
| search[indian_video/frame] an almost empty road | PASS | 200 n=5 | 19 |
| search[indian_video/crop] an almost empty road | PASS | 200 n=5 | 18 |
| empty query -> 400 | PASS |  |  |
| bad dataset -> 400 | PASS |  |  |
| camera filter | PASS |  |  |
| keyframe outside project -> 403 | PASS |  |  |