# Where frame order gets lost

## 1 fps (4–8 frames per clip)

- Search, clip: top-1 22/23, top-3 23/23, MRR 0.978
- Search, bag of frames: top-1 23/23, top-3 23/23, MRR 1.000
- Search, clip with timestamps: top-1 23/23, top-3 23/23, MRR 1.000
- Direction, paired, same frames reversed: 16/32 = 50% (95% CI 32%–68%, p = 1.00 vs chance)
- Control, paired, shuffled frames in place of the reversed ones: 15/32 = 47% (95% CI 29%–65%, p = 0.86 vs chance)
- Direction, ranking (forward caption scores the clip above its reverse): 18/32 = 56% (95% CI 38%–74%, p = 0.60 vs chance)
- Direction, paired, reversed with timestamps kept ascending: 21/32 = 66% (95% CI 47%–81%, p = 0.11 vs chance)
- Cosine of the pooled vectors, median: fwd vs rfr 0.9974, fwd vs shf 0.9981, fwd vs bag 0.9588, ts_fwd vs ts_rfr 0.9966
- Cosine per token (the same frame's tokens, before pooling), median over clips: fwd vs rfr 0.9292, fwd vs shf 0.9344; for scale, the frame in the clip vs the frame alone 0.8275

## 2 fps (9–16 frames per clip)

- Search, clip: top-1 22/23, top-3 23/23, MRR 0.978
- Search, bag of frames: top-1 23/23, top-3 23/23, MRR 1.000
- Search, clip with timestamps: top-1 23/23, top-3 23/23, MRR 1.000
- Direction, paired, same frames reversed: 15/32 = 47% (95% CI 29%–65%, p = 0.86 vs chance)
- Control, paired, shuffled frames in place of the reversed ones: 14/32 = 44% (95% CI 26%–62%, p = 0.60 vs chance)
- Direction, ranking (forward caption scores the clip above its reverse): 13/32 = 41% (95% CI 24%–59%, p = 0.38 vs chance)
- Direction, paired, reversed with timestamps kept ascending: 21/32 = 66% (95% CI 47%–81%, p = 0.11 vs chance)
- Cosine of the pooled vectors, median: fwd vs rfr 0.9968, fwd vs shf 0.9963, fwd vs bag 0.9545, ts_fwd vs ts_rfr 0.9971
- Cosine per token (the same frame's tokens, before pooling), median over clips: fwd vs rfr 0.9086, fwd vs shf 0.8843; for scale, the frame in the clip vs the frame alone 0.7782

## 4 fps (19–32 frames per clip)

- Search, clip: top-1 23/23, top-3 23/23, MRR 1.000
- Search, bag of frames: top-1 23/23, top-3 23/23, MRR 1.000
- Search, clip with timestamps: top-1 23/23, top-3 23/23, MRR 1.000
- Direction, paired, same frames reversed: 8/32 = 25% (95% CI 11%–43%, p = 0.01 vs chance)
- Control, paired, shuffled frames in place of the reversed ones: 17/32 = 53% (95% CI 35%–71%, p = 0.86 vs chance)
- Direction, ranking (forward caption scores the clip above its reverse): 17/32 = 53% (95% CI 35%–71%, p = 0.86 vs chance)
- Direction, paired, reversed with timestamps kept ascending: 10/32 = 31% (95% CI 16%–50%, p = 0.05 vs chance)
- Cosine of the pooled vectors, median: fwd vs rfr 0.9963, fwd vs shf 0.9937, fwd vs bag 0.9509, ts_fwd vs ts_rfr 0.9972
- Cosine per token (the same frame's tokens, before pooling), median over clips: fwd vs rfr 0.8761, fwd vs shf 0.8192; for scale, the frame in the clip vs the frame alone 0.7185

## Does a clip's paired reading repeat across frame rates?

- 1 vs 2 fps: same answer for 17/32 clips
- 1 vs 4 fps: same answer for 14/32 clips
- 2 vs 4 fps: same answer for 23/32 clips
