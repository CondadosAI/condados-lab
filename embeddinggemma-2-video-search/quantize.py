"""INT8 versions of the exported graphs, with NNCF.

- int8w: weights only (`compress_weights`). Activations stay in float, so the iGPU still computes
  in f16/f32; what shrinks is the model file and the memory traffic.
- int8: weights and activations (`quantize`, post-training). Needs calibration data to set each
  activation's range: frames for the vision graph, padded clip sequences for the encoder.

Calibration uses 16 corpus clips. It only sets ranges, no labels are involved, but the clips are
also evaluated later, so the post states it.

    uv run python quantize.py
"""

import json
from pathlib import Path

import nncf
import numpy as np
import openvino as ov
from loguru import logger
from ov_model import MODELS, VISION_BATCH, OVEmbedder
from smoke_export import sample_frames

HERE = Path(__file__).parent
N_CALIB = 16


def main() -> None:
    core = ov.Core()
    clips = json.loads((HERE / "clips.json").read_text())["clips"]
    calib = clips[:: max(1, len(clips) // N_CALIB)][:N_CALIB]
    ref = OVEmbedder("fp32", "CPU")

    vis_items, enc_items = [], []
    for c in calib:
        frames = sample_frames(HERE / "data" / "clips" / f"{c['key']}_fwd.mp4")
        x = ref.prompt(frames)
        pv, pos = x["pixel_values_videos"], x["video_position_ids"]
        k = VISION_BATCH - len(pv)
        if k > 0:
            pv = np.concatenate([pv, np.repeat(pv[-1:], k, 0)])
            pos = np.concatenate([pos, np.repeat(pos[-1:], k, 0)])
        vis_items.append((pv[:VISION_BATCH], pos[:VISION_BATCH]))
        emb = ref.video_inputs(frames)  # calibrated at the 1024 bucket every 8-frame clip uses
        e, m = np.zeros((1, 1024, emb.shape[1]), np.float32), np.zeros((1, 1024), np.int64)
        e[0, : len(emb)], m[0, : len(emb)] = emb, 1
        enc_items.append((e, m))
    logger.info(f"calibration: {len(vis_items)} clips")

    for name, items in (("vision", vis_items), ("encoder", enc_items)):
        model = core.read_model(MODELS / "fp32" / f"{name}.xml")
        w = nncf.compress_weights(model)
        (MODELS / "int8w").mkdir(exist_ok=True)
        ov.save_model(w, MODELS / "int8w" / f"{name}.xml", compress_to_fp16=False)

        model = core.read_model(MODELS / "fp32" / f"{name}.xml")
        q = nncf.quantize(
            model,
            nncf.Dataset(items, lambda it: list(it)),
            model_type=nncf.ModelType.TRANSFORMER,
            subset_size=len(items),
        )
        (MODELS / "int8").mkdir(exist_ok=True)
        ov.save_model(q, MODELS / "int8" / f"{name}.xml", compress_to_fp16=False)
        logger.info(f"{name}: int8w and int8 saved")


if __name__ == "__main__":
    main()
