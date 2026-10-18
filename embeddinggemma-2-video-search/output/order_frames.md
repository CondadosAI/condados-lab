# Per-frame readings

## 1 fps

Clip vector vs each frame encoded alone, mean cosine by the quarter of the sequence the frame sits in (1st → 4th):

- fwd: 0.9175 · 0.9396 · 0.9366 · 0.9228
- rfr: 0.9276 · 0.9381 · 0.9337 · 0.9150

Forward vs reversed, linear classifier, accuracy with each clip held out (chance 50%):

- from a frame's token mean (same frame in both): 51%
- from the pooled clip vector: 52%

## 4 fps

Clip vector vs each frame encoded alone, mean cosine by the quarter of the sequence the frame sits in (1st → 4th):

- fwd: 0.9169 · 0.9277 · 0.9251 · 0.9107
- rfr: 0.9193 · 0.9285 · 0.9246 · 0.9085

Forward vs reversed, linear classifier, accuracy with each clip held out (chance 50%):

- from a frame's token mean (same frame in both): 51%
- from the pooled clip vector: 50%
