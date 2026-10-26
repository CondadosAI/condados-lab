# condados-lab

Companion code for the posts on [condados.ai](https://condados.ai). Each subfolder belongs to one post and holds a
few scripts that run top to bottom, one notebook that walks the result end to end, and an `output/` folder with the
numbers the post publishes.

| Folder | Post |
|---|---|
| [`embeddinggemma-2-video-search/`](embeddinggemma-2-video-search/) | [EmbeddingGemma 2 Video Search on a Laptop iGPU](https://condados.ai/blog/embeddinggemma-2-video-search) |
| [`embeddinggemma-2-basketball-moments/`](embeddinggemma-2-basketball-moments/) | What a free video model finds in your footage, and what it misses (in progress) |
| [`valdrada-traffic-safety/`](valdrada-traffic-safety/) | Traffic-safety cameras in a simulated New York (in progress) |

```bash
uv sync
```

Code is Apache-2.0 (see `LICENSE`). Models and datasets keep their own licences, listed in each subfolder's README.
