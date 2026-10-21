# Retrieval ablation: frame vs crop embeddings

7 labelled queries, corpus of 73 sub-clips, no camera/zone filters. Random-ranking baseline P@K ~ 0.149.

**Overall (mean over queries)**

| Embedding | P@1 | P@3 | P@5 | P@10 | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|---|---|---|---|
| frame | 0.571 | 0.619 | 0.486 | 0.414 | 0.059 | 0.185 | 0.218 | 0.347 | 0.643 |
| crop | 0.714 | 0.524 | 0.371 | 0.300 | 0.123 | 0.208 | 0.230 | 0.336 | 0.786 |

**Category: object (2 queries)**

| Embedding | P@1 | P@3 | P@5 | P@10 | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|---|---|---|---|
| frame | 1.000 | 0.833 | 0.800 | 0.750 | 0.061 | 0.141 | 0.221 | 0.401 | 1.000 |
| crop | 1.000 | 0.833 | 0.600 | 0.500 | 0.061 | 0.141 | 0.183 | 0.282 | 1.000 |

**Category: scene (5 queries)**

| Embedding | P@1 | P@3 | P@5 | P@10 | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|---|---|---|---|
| frame | 0.400 | 0.533 | 0.360 | 0.280 | 0.058 | 0.203 | 0.217 | 0.326 | 0.500 |
| crop | 0.600 | 0.400 | 0.280 | 0.220 | 0.148 | 0.235 | 0.250 | 0.358 | 0.700 |
