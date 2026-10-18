"""P0: can EmbeddingGemma 2's video path run in OpenVINO at all?

The model is two networks in a row. The vision tower turns each sampled frame into soft tokens
(120 for a 640x360 frame) on its own, with no view of the other frames. The 24-layer text encoder
then reads every frame's tokens together, bidirectionally, with rotary positions, and the
embedding is the mean of its outputs over the attention mask. We export the two separately,
because the vision tower runs once per frame with a fixed shape while the encoder's sequence
length depends on the clip.

Reference: the PyTorch model in float32, pooled the way SentenceTransformers pools it. Pass: cosine
>= 0.9999 against that reference, for a video and for a text query.

    uv run python smoke_export.py
"""

import subprocess
import time
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from loguru import logger
from transformers import AutoModel, AutoProcessor

MODEL_ID = "google/embeddinggemma-2"
HERE = Path(__file__).parent
WORK = HERE / "cache" / "smoke"
QUERY = "task: search result | query: a colourful test pattern with a moving gradient"


def make_test_clip(path: Path, seconds: int = 6) -> Path:
    """A synthetic clip, so P0 needs no download: ffmpeg's moving test pattern, no audio."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"testsrc2=size=640x360:rate=25:duration={seconds}",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=True,
        )
    return path


def sample_frames(path: Path, fps: float = 1.0) -> np.ndarray:
    """Decode with PyAV and keep the frame nearest each 1/fps tick: (T, H, W, 3) uint8.

    The processor's own decoder needs torchcodec, which does not load against this system's
    ffmpeg; sampling here also makes the frame choice explicit, which the direction test needs.
    """
    import av

    with av.open(str(path)) as c:
        stream = c.streams.video[0]
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(stream)]
        native = float(stream.average_rate)
    step = native / fps
    idx = [min(round(k * step), len(frames) - 1) for k in range(int(len(frames) / step))]
    return np.stack([frames[i] for i in idx])


def pooled(last_hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean over real tokens, then L2: what SentenceTransformers does after the encoder."""
    m = mask[..., None].astype(np.float32)
    v = (last_hidden * m).sum(axis=1) / m.sum(axis=1)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))


class VisionPath(torch.nn.Module):
    """One frame of patches -> its soft tokens in the encoder's input space."""

    def __init__(self, model):
        super().__init__()
        self.tower, self.proj = model.vision_tower, model.embed_vision

    def forward(self, pixel_values, position_ids):
        hidden = self.tower(
            pixel_values=pixel_values, pixel_position_ids=position_ids
        ).last_hidden_state
        return self.proj(inputs_embeds=hidden)


class EncoderPath(torch.nn.Module):
    """Input embeddings + mask -> projected per-token outputs (pooling stays outside)."""

    def __init__(self, model):
        super().__init__()
        self.lm = model.language_model

    def forward(self, inputs_embeds, attention_mask):
        return self.lm(inputs_embeds=inputs_embeds, attention_mask=attention_mask).last_hidden_state


def main() -> None:
    torch.set_grad_enabled(False)
    clip = make_test_clip(WORK / "testsrc2_6s.mp4")
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    proc = AutoProcessor.from_pretrained(MODEL_ID)

    # --- PyTorch reference -------------------------------------------------------------
    clip_frames = sample_frames(clip)
    vid = proc(
        text=["<|video|>"], videos=[clip_frames], do_sample_frames=False, return_tensors="pt"
    )
    txt = proc(text=[QUERY], return_tensors="pt")
    n_frames = int(vid["num_frames_per_video"][0])
    logger.info(f"clip: {n_frames} frames, sequence {vid['input_ids'].shape[1]} tokens")

    t = time.perf_counter()
    ref_vid_hidden = model(**vid).last_hidden_state.numpy()
    logger.info(f"PyTorch video forward: {time.perf_counter() - t:.2f} s")
    ref_vid = pooled(ref_vid_hidden, vid["attention_mask"].numpy())
    ref_txt = pooled(model(**txt).last_hidden_state.numpy(), txt["attention_mask"].numpy())

    # Is this pooling the one SentenceTransformers uses? Check on text, where it is unambiguous.
    try:
        from sentence_transformers import SentenceTransformer

        st = SentenceTransformer(MODEL_ID, model_kwargs={"dtype": torch.float32})
        st_txt = st.encode(QUERY, normalize_embeddings=True)
        logger.info(
            f"pooling check, ours vs SentenceTransformers (text): {cos(ref_txt[0], st_txt):.6f}"
        )
        st_vid = st.encode({"video": clip_frames}, normalize_embeddings=True)
        logger.info(
            f"pooling check, ours vs SentenceTransformers (video): {cos(ref_vid[0], st_vid):.6f}"
        )
        del st
    except Exception as e:  # the check is informative, not required for P0
        logger.warning(f"SentenceTransformers check skipped: {type(e).__name__}: {e}")

    # --- Export ------------------------------------------------------------------------
    core = ov.Core()
    pv, pos = vid["pixel_values_videos"], vid["video_position_ids"]
    WORK.mkdir(parents=True, exist_ok=True)

    t = time.perf_counter()
    vis_ov = ov.convert_model(VisionPath(model).eval(), example_input=(pv[:1], pos[:1]))
    vis_ov.reshape({0: list(pv[:1].shape), 1: list(pos[:1].shape)})  # static: one frame
    ov.save_model(vis_ov, WORK / "vision.xml", compress_to_fp16=False)
    logger.info(f"vision exported in {time.perf_counter() - t:.1f} s, input {list(pv[:1].shape)}")

    # Rebuild the encoder's input the way EmbeddingGemma2Model.forward does: text-token embeddings,
    # with the video placeholders overwritten by the frames' soft tokens.
    def encoder_inputs(frame_tokens: np.ndarray, ids: torch.Tensor) -> np.ndarray:
        pad = model.config.text_config.pad_token_id
        vmask = ids == proc.video_token_id
        emb = model.get_input_embeddings()(torch.where(vmask, pad, ids)).numpy()
        emb[vmask.numpy()] = frame_tokens.reshape(-1, emb.shape[-1])
        return emb

    t = time.perf_counter()
    enc_example = (
        torch.randn(1, 64, model.config.text_config.hidden_size),
        torch.ones(1, 64, dtype=torch.long),
    )
    enc_ov = ov.convert_model(EncoderPath(model).eval(), example_input=enc_example)
    ov.save_model(enc_ov, WORK / "encoder.xml", compress_to_fp16=False)
    logger.info(f"encoder exported in {time.perf_counter() - t:.1f} s (dynamic sequence length)")

    # --- OpenVINO on CPU vs the reference ----------------------------------------------
    vis_c = core.compile_model(WORK / "vision.xml", "CPU")
    enc_c = core.compile_model(WORK / "encoder.xml", "CPU")

    t = time.perf_counter()
    frames = [vis_c([pv[i : i + 1].numpy(), pos[i : i + 1].numpy()])[0] for i in range(n_frames)]
    # The tower drops padded patches itself, so each frame comes back as (tokens, hidden). The trace
    # fixes that count, so every clip must share one frame size (640x360 -> 120 tokens).
    tok_per_frame = int(((pos[0] != -1).all(-1).sum()) // 9)
    assert all(f.shape[0] == tok_per_frame for f in frames), [f.shape for f in frames]
    frames = np.concatenate(frames)
    emb = encoder_inputs(frames, vid["input_ids"])
    ov_vid_hidden = enc_c([emb, vid["attention_mask"].numpy()])[0]
    logger.info(f"OpenVINO CPU video: {time.perf_counter() - t:.2f} s")
    ov_vid = pooled(ov_vid_hidden, vid["attention_mask"].numpy())

    ov_txt = pooled(
        enc_c(
            [model.get_input_embeddings()(txt["input_ids"]).numpy(), txt["attention_mask"].numpy()]
        )[0],
        txt["attention_mask"].numpy(),
    )
    logger.info(f"cosine OV vs PyTorch, video: {cos(ov_vid[0], ref_vid[0]):.6f}")
    logger.info(f"cosine OV vs PyTorch, text:  {cos(ov_txt[0], ref_txt[0]):.6f}")
    logger.info(
        f"cosine query vs clip, PyTorch {cos(ref_txt[0], ref_vid[0]):.4f}, "
        f"OV {cos(ov_txt[0], ov_vid[0]):.4f}"
    )

    # Static shape for the iGPU: pad the sequence and check the mask keeps padding out of the mean.
    L = emb.shape[1]
    Lp = L + 37
    emb_p = np.concatenate([emb, np.zeros((1, Lp - L, emb.shape[-1]), np.float32)], axis=1)
    mask_p = np.concatenate(
        [vid["attention_mask"].numpy(), np.zeros((1, Lp - L), np.int64)], axis=1
    )
    ov_pad = pooled(enc_c([emb_p, mask_p])[0], mask_p)
    logger.info(f"cosine padded ({L}->{Lp}) vs unpadded: {cos(ov_pad[0], ov_vid[0]):.6f}")

    np.savez(WORK / "smoke.npz", ref_vid=ref_vid, ref_txt=ref_txt, ov_vid=ov_vid, ov_txt=ov_txt)


if __name__ == "__main__":
    main()
