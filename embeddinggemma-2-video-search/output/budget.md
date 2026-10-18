# Vision token budget per frame

The laptop clip (7431681, 8 frames at 640x360), int8 on the iGPU at f16: 3 warm-ups, median of 10, all budgets in one session. Search over 43 clips and 23 queries; fp32 is the iGPU at f32 (matches PyTorch at cosine 1.0000 at budget 140).

| Budget | Tokens per frame | Sequence | Clip, frames → vector (ms) | Vision (ms) | Top-1 int8 | Top-1 fp32 | MRR int8 | MRR fp32 | int8 vs fp32 cos (median, worst) | vs budget 140 cos (median) | Paired, fp32 | fwd vs reversed cos (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 70 | 66 | 546 | 1,013 | 789 | 22/23 | 22/23 | 0.978 | 0.978 | 0.9934, 0.9741 | 0.9786 | 17/32 = 53% (95% CI 35%–71%, p = 0.86 vs chance) | 0.9982 |
| 140 | 120 | 978 | 2,533 | 2,286 | 22/23 | 22/23 | 0.978 | 0.978 | 0.9945, 0.9745 | 1.0000 | 16/32 = 50% (95% CI 32%–68%, p = 1.00 vs chance) | 0.9974 |
| 280 | 264 | 2130 | 9,038 | 8,046 | 5/23 | 23/23 | 0.349 | 1.000 | 0.6756, 0.5458 | 0.9700 | 12/32 = 38% (95% CI 21%–56%, p = 0.22 vs chance) | 0.9956 |
