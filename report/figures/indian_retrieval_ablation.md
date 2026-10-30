### Indian-domain retrieval ablation (UVH-26 test images, N=400, 8 queries)

| embedding | prompt ensemble | P@1 | P@3 | P@5 | P@10 | MRR |
|---|---|---|---|---|---|---|
| frame | no | 0.375 | 0.417 | 0.300 | 0.325 | 0.568 |
| frame | yes | 0.375 | 0.458 | 0.425 | 0.287 | 0.627 |
| crop | no | 0.125 | 0.292 | 0.300 | 0.212 | 0.369 |
| crop | yes | 0.125 | 0.250 | 0.300 | 0.275 | 0.337 |
| frame+crop (combined) | no | 0.375 | 0.458 | 0.400 | 0.325 | 0.584 |
| frame+crop (combined) | yes | 0.500 | 0.583 | 0.450 | 0.375 | 0.688 |

Random-ranking baseline (mean fraction of relevant images): 0.123

Queries and number of relevant images (from annotation counts):

- "many motorcycles and scooters on the road": 40 relevant
- "auto rickshaws on the road": 42 relevant
- "a bus on the road": 120 relevant
- "trucks on the road": 48 relevant
- "bicycles on the road": 51 relevant
- "many cars on the road": 44 relevant
- "heavy traffic with many vehicles": 44 relevant
- "an almost empty road with hardly any vehicles": 4 relevant