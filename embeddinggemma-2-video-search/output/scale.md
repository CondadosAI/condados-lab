# Exact search, no index

One query, top-10, float32 numpy, one CPU thread (i7-12700H); median of 20 runs after 3 warm-ups.

| Vectors | 768-d: ms per query | 768-d: memory | 256-d: ms per query | 256-d: memory |
|---|---|---|---|---|
| 43 | 0.010 | 0.13 MB | 0.010 | 0.04 MB |
| 1,000 | 0.064 | 2.93 MB | 0.029 | 0.98 MB |
| 10,000 | 0.961 | 29.30 MB | 0.197 | 9.77 MB |
| 100,000 | 10.466 | 292.97 MB | 4.389 | 97.66 MB |
| 1,000,000 | 104.126 | 2,929.69 MB | 45.385 | 976.56 MB |

Search quality with shorter vectors: 23 queries over 43 clips.

| Dimensions | Top-1 | Top-3 | MRR |
|---|---|---|---|
| 768 | 22/23 | 23/23 | 0.978 |
| 512 | 23/23 | 23/23 | 1.000 |
| 256 | 23/23 | 23/23 | 1.000 |
| 128 | 23/23 | 23/23 | 1.000 |
