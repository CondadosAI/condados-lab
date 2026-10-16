"""Export EmbeddingGemma 2 to OpenVINO IR (models/fp32/) and check it against PyTorch.

The vision graph is traced with a batch of 8 real 640x360 frames: the tower drops padded patches
inside the graph, and the trace fixes how many tokens each frame keeps (120). The encoder is
traced with a dynamic shape and reshaped to a static one when it is compiled (ov_model.py).

    uv run python export_openvino.py
"""

import json
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from loguru import logger
from ov_model import MODEL_ID, MODELS, VISION_BATCH, OVEmbedder
from smoke_export import EncoderPath, VisionPath, sample_frames
from transformers import AutoModel, AutoProcessor

HERE = Path(__file__).parent


def main() -> None:
    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    assert proc.tokenizer.pad_token_id == model.config.text_config.pad_token_id
    out = MODELS / "fp32"
    out.mkdir(parents=True, exist_ok=True)

    clips = json.loads((HERE / "clips.json").read_text())["clips"]
    frames = sample_frames(HERE / "data" / "clips" / f"{clips[0]['key']}_fwd.mp4")
    frames = np.concatenate([frames, frames])[:VISION_BATCH]
    x = proc(text=["<|video|>"], videos=[frames], do_sample_frames=False, return_tensors="pt")
    vis = ov.convert_model(
        VisionPath(model).eval(), example_input=(x["pixel_values_videos"], x["video_position_ids"])
    )
    ov.save_model(vis, out / "vision.xml", compress_to_fp16=False)
    enc_example = (torch.randn(1, 64, 512), torch.ones(1, 64, dtype=torch.long))
    enc = ov.convert_model(EncoderPath(model).eval(), example_input=enc_example)
    ov.save_model(enc, out / "encoder.xml", compress_to_fp16=False)

    emb = model.get_input_embeddings()
    table = (emb.weight * emb.embed_scale).numpy().astype(np.float32)
    np.save(MODELS / "token_embeddings.npy", table)
    logger.info(f"exported to {out}; token table {table.shape}")

    # Check: the OpenVINO pipeline against the saved PyTorch reference, on 3 clips and 3 queries.
    ref = np.load(HERE / "index" / "pytorch_fp32_fps1.npz")
    ov_e = OVEmbedder("fp32", "CPU")
    queries = json.loads((HERE / "clips.json").read_text())["queries"]
    for c in clips[:3]:
        v = ov_e.embed_videos([sample_frames(HERE / "data" / "clips" / f"{c['key']}_fwd.mp4")])[0]
        logger.info(f"{c['key']}: cos vs PyTorch {v @ ref[f'fwd/{c["key"]}']:.6f}")
    for i, q in enumerate(queries[:3]):
        logger.info(
            f"query {i}: cos vs PyTorch {ov_e.embed_text(q['query']) @ ref[f'query/{i}']:.6f}"
        )


if __name__ == "__main__":
    main()
