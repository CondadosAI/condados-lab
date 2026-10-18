# Results for `pytorch_fp32_fps1`

## Search

23 queries over 43 clips: top-1 22/23, top-3 23/23, MRR 0.978

| Query | Rank of the right clip | Top result | Score |
|---|---|---|---|
| a swimmer dives off the starting block | 1 | a swimmer dives into a pool | 0.728 |
| glass shattering into pieces | 1 | a glass jar shatters | 0.772 |
| cracking an egg into a steel bowl | 1 | an egg is cracked into a bowl | 0.729 |
| espresso dripping into a white cup | 1 | coffee pours into a cup | 0.778 |
| a red train pulling into a station | 1 | a train arrives at the station | 0.748 |
| an airplane landing at an airport | 1 | an airplane lands on the runway | 0.709 |
| a man inflating a purple balloon | 1 | a man blows up a balloon | 0.775 |
| a man on the beach putting on a straw hat | 1 | a man puts on a hat | 0.747 |
| packing clothes into a suitcase | 1 | someone packs a suitcase | 0.729 |
| a car on a winding country road | 1 | a car drives away down the road | 0.727 |
| riding an escalator in a station | 1 | riding an escalator up | 0.740 |
| a woman wearing red headphones | 1 | a woman puts on headphones | 0.719 |
| striking a match | 1 | a match is struck and lights | 0.662 |
| a candle flame going out | 1 | a candle is blown out | 0.696 |
| dominoes toppling | 2 | a glass jar shatters | 0.656 |
| black and white waves hitting rocks | 1 | a wave crashes on rocks | 0.775 |
| snapping a pencil | 1 | a pencil is snapped in half | 0.734 |
| pulling on a black glove | 1 | a woman puts on a glove | 0.723 |
| a man getting into a black car | 1 | a man gets into a car | 0.707 |
| drawing back the curtains to let daylight in | 1 | a person opens the curtains | 0.657 |
| a laptop showing a world map | 1 | someone closes a laptop | 0.746 |
| a woman putting on glasses | 1 | a woman puts on glasses | 0.738 |
| a person walking along a dirt path | 1 | a person walks away from the camera | 0.699 |

## Direction of time

### Reverse = same frames, reversed order (`rfr`)

- Paired (reversible clips): 16/32 = 50% (95% CI 32%–68%, p = 1.00 vs chance)
- Ranking, reversible clips: 18/32 = 56% (95% CI 38%–74%, p = 0.60 vs chance)
- Ranking, one-way clips: 7/11 = 64% (95% CI 31%–89%, p = 0.55 vs chance)
- Decodable, held-out classifier on the vectors: 45/86 = 52% of vectors

### Reverse = reversed file, resampled (`rev`)

- Paired (reversible clips): 17/32 = 53% (95% CI 35%–71%, p = 0.86 vs chance)
- Ranking, reversible clips: 16/32 = 50% (95% CI 32%–68%, p = 1.00 vs chance)
- Ranking, one-way clips: 5/11 = 45% (95% CI 17%–77%, p = 1.00 vs chance)
- Decodable, held-out classifier on the vectors: 44/86 = 51% of vectors

How far apart the versions are (cosine):

| Pair | Median | Min | Max |
|---|---|---|---|
| clip vs same frames, reversed order | 0.9974 | 0.9879 | 0.9993 |
| clip vs reversed file, resampled | 0.9890 | 0.9538 | 0.9973 |
| clip vs its frames shuffled | 0.9980 | 0.9893 | 0.9998 |
| two different clips | 0.6130 | 0.4929 | 0.9853 |

Filmed mirrors (the same set, the action really done both ways):

- laptop opened vs closed: cos(rev 7431681, real 7431863) = 0.9634, cos(fwd 7431681, real 7431863) = 0.9554
- mug: cos(rev 39173954a, real 39173954b) = 0.9869, cos(fwd 39173954a, real 39173954b) = 0.9853
