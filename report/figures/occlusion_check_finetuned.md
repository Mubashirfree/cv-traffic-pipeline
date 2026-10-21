# Occlusion check: does the tracker keep the same ID across occlusion?

IoU >= 0.15 between two GT boxes on a sampled frame = occluded. For each occlusion run we compare the matched prediction ID on the frame right before it starts and right after it ends (matched to GT by IoU >= 0.3).

**50 occlusion events found. ID survived in 22 (44.0%).**

| Camera | events | survived | switched | lost (no match one side) |
|---|---|---|---|---|
| MVI_20011 | 8 | 4 | 2 | 2 |
| MVI_20012 | 17 | 12 | 0 | 5 |
| MVI_20032 | 0 | 0 | 0 | 0 |
| MVI_20033 | 0 | 0 | 0 | 0 |
| MVI_20034 | 10 | 6 | 0 | 4 |
| MVI_40992 | 15 | 0 | 0 | 15 |
| **overall** | **50** | **22** | **2** | **26** |
