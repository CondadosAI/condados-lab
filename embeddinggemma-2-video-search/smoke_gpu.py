"""P0, iGPU half: does the exported model survive the GPU plugin's default f16 precision?

The model card warns that float16 gives NaN or quietly degraded embeddings, and OpenVINO's GPU
plugin runs in f16 unless told otherwise. Same clip and query as smoke_export.py; reference is the
PyTorch float32 embedding it saved.

    uv run python smoke_gpu.py
"""

import time

import numpy as np
import openvino as ov
import torch
from loguru import logger
from smoke_export import MODEL_ID, QUERY, WORK, cos, pooled, sample_frames
from transformers import AutoModel, AutoProcessor


def main() -> None:
    torch.set_grad_enabled(False)
    core = ov.Core()
    gpus = [d for d in core.available_devices if d.startswith("GPU")]
    logger.info(
        f"devices: {core.available_devices}; GPU name: "
        f"{core.get_property(gpus[0], 'FULL_DEVICE_NAME') if gpus else 'none'}"
    )
    if not gpus:
        return
    ref = np.load(WORK / "smoke.npz")
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    vid = proc(
        text=["<|video|>"],
        videos=[sample_frames(WORK / "testsrc2_6s.mp4")],
        do_sample_frames=False,
        return_tensors="pt",
    )
    txt = proc(text=[QUERY], return_tensors="pt")
    embed = model.get_input_embeddings()
    pv, pos, ids = (
        vid["pixel_values_videos"].numpy(),
        vid["video_position_ids"].numpy(),
        vid["input_ids"],
    )
    mask = vid["attention_mask"].numpy()

    for precision in ("f16", "f32"):
        cfg = {"INFERENCE_PRECISION_HINT": precision}
        vis = core.compile_model(WORK / "vision.xml", gpus[0], cfg)
        enc = core.compile_model(WORK / "encoder.xml", gpus[0], cfg)
        for _ in range(2):  # second pass is warm
            t = time.perf_counter()
            frames = np.concatenate(
                [vis([pv[i : i + 1], pos[i : i + 1]])[0] for i in range(len(pv))]
            )
            t_vis = time.perf_counter() - t
            vmask = ids == proc.video_token_id
            emb = embed(torch.where(vmask, model.config.text_config.pad_token_id, ids)).numpy()
            emb[vmask.numpy()] = frames
            t = time.perf_counter()
            hid = enc([emb, mask])[0]
            t_enc = time.perf_counter() - t
        v = pooled(hid, mask)
        q = pooled(
            enc([embed(txt["input_ids"]).numpy(), txt["attention_mask"].numpy()])[0],
            txt["attention_mask"].numpy(),
        )
        logger.info(
            f"{gpus[0]} {precision}: NaN frames={np.isnan(frames).any()} "
            f"NaN out={np.isnan(hid).any()} max|frame token|={np.nanmax(np.abs(frames)):.1f} "
            f"| cos video vs fp32 {cos(v[0], ref['ref_vid'][0]):.6f} "
            f"| cos text vs fp32 {cos(q[0], ref['ref_txt'][0]):.6f} "
            f"| warm: vision {t_vis * 1000 / len(pv):.0f} ms/frame, encoder {t_enc * 1000:.0f} ms"
        )


if __name__ == "__main__":
    main()
