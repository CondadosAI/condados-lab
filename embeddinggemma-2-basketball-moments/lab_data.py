"""Export the QVHighlights scores for the post's interactive lab: per query, its video's segments,
both models' cosine scores and the annotated spans. No frames, no vectors.

    uv run python lab_data.py          # -> output/qvh_lab.json

Queries and spans are QVHighlights annotations (CC BY-NC-SA 4.0, Lei et al. 2021), as in
qvh_sample.json; the scores are this project's.
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
INDEX, OUT = HERE / "index", HERE / "output"


def main() -> None:
    sample = json.loads((HERE / "qvh_sample.json").read_text())
    idx = {m: np.load(INDEX / f"qvh_{m}_visual.npz") for m in ("marengo", "gemma")}
    tv = {m: dict(zip(d["texts"].tolist(), d["text_emb"], strict=True)) for m, d in idx.items()}
    out = []
    for q in sample["queries"]:
        rows = {}
        for m, d in idx.items():
            r = np.where(d["vid"] == q["vid"])[0]
            rows[m] = r[~np.isnan(d["emb"][r]).any(axis=1)]
        mk, gm = idx["marengo"], idx["gemma"]
        # Both files hold the same segments (Gemma ran on Marengo's cuts); check it.
        assert np.allclose(mk["start"][rows["marengo"]], gm["start"][rows["gemma"]])
        s, e = mk["start"][rows["marengo"]], mk["end"][rows["marengo"]]
        out.append(
            {
                "qid": q["qid"],
                "query": q["query"],
                "duration": q["duration"],
                "windows": q["relevant_windows"],
                "seg": [[round(float(a), 2), round(float(b), 2)] for a, b in zip(s, e)],
                "marengo": [round(float(x), 4) for x in mk["emb"][rows["marengo"]] @ tv["marengo"][q["query"]]],
                "gemma": [round(float(x), 4) for x in gm["emb"][rows["gemma"]] @ tv["gemma"][q["query"]]],
            }
        )
    OUT.mkdir(exist_ok=True)
    (OUT / "qvh_lab.json").write_text(json.dumps(out, separators=(",", ":")))
    print(len(out), "queries,", (OUT / "qvh_lab.json").stat().st_size, "bytes")


if __name__ == "__main__":
    main()
