"""EmbeddingGemma 2 video and text embeddings on OpenVINO, with static shapes.

Three graphs, each compiled once:

- vision: `VISION_BATCH` frames of patches -> 120 soft tokens each (640x360 frames). A clip with
  fewer frames is padded with copies of its last frame, and their tokens are dropped.
- encoder: `enc_batch` sequences of input embeddings + mask -> per-token outputs, compiled per
  length bucket (128, 256, 512, 1024, ... 4608) the first time a clip needs it; each clip goes to
  the shortest bucket that fits. The mask keeps padding out of the attention and out of the mean.
- the same encoder graph reshaped to `TEXT_LEN` for queries.

The token-embedding table stays in numpy (memory-mapped): a video sequence only looks up its few
special tokens, so the 262k x 512 table never has to sit on the GPU.

Static shapes matter on the iGPU: a dynamic graph is recompiled or run through slower kernels.

Why two lengths and not one: most encoder layers attend within a sliding window of 512 tokens. Pad
a 490-token clip to 1024 and the last padding positions have a window holding only padding, so
every key is masked. In f32 the mask value is -3.4e38, that softmax row turns to NaN, and the NaN
reaches the real tokens through a 0 x NaN in the next attention. In f16 the mask value is finite
and the bug hides. Buckets 512 apart keep padding under the window (`bucket - length < WINDOW`).
"""

from pathlib import Path

import numpy as np
import openvino as ov
from transformers import AutoProcessor

MODEL_ID = "google/embeddinggemma-2"
HERE = Path(__file__).parent
MODELS = HERE / "models"
VISION_BATCH, TOK_PER_FRAME, HIDDEN = 8, 120, 512
VIDEO_LEN, TEXT_LEN, WINDOW = 4608, 64, 512  # 32 frames (4 fps x 8 s) with timestamps = 4129
# 128 and 256 hold one frame (~124 tokens) without padding it to 512.
BUCKETS = (128, 256, *range(WINDOW, VIDEO_LEN + 1, WINDOW))
QUERY_PREFIX = "task: search result | query: "


def igpu() -> str:
    """The Intel integrated GPU's device name (GPU.0 here). A discrete GPU, if any, is GPU.1."""
    core = ov.Core()
    for d in core.available_devices:
        if d.startswith("GPU") and "iGPU" in core.get_property(d, "FULL_DEVICE_NAME"):
            return d
    raise RuntimeError("no Intel iGPU found")


def pooled(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    m = mask[..., None].astype(np.float32)
    v = (hidden * m).sum(axis=1) / m.sum(axis=1)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class OVEmbedder:
    def __init__(self, variant: str = "fp32", device: str = "CPU", precision: str | None = None,
                 vision_batch: int = VISION_BATCH, enc_batch: int = 1):  # fmt: skip
        """`variant` is a folder under models/ (fp32, int8w, int8); `precision` is the GPU's
        INFERENCE_PRECISION_HINT (f16 is the plugin default, f32 the safe choice)."""
        core = ov.Core()
        core.set_property({"CACHE_DIR": str(MODELS / "cache")})
        cfg = {"PERFORMANCE_HINT": "LATENCY"}
        if precision:
            cfg["INFERENCE_PRECISION_HINT"] = precision
        d = MODELS / variant
        self.vision_batch, self.enc_batch = vision_batch, enc_batch

        vis = core.read_model(d / "vision.xml")
        vis.reshape({0: [vision_batch, 1260, 768], 1: [vision_batch, 1260, 2]})
        self.vision = core.compile_model(vis, device, cfg)

        self._core, self._device, self._cfg, self._enc_xml = core, device, cfg, d / "encoder.xml"
        self._encoders: dict[int, ov.CompiledModel] = {}
        txt = core.read_model(d / "encoder.xml")
        txt.reshape({0: [1, TEXT_LEN, HIDDEN], 1: [1, TEXT_LEN]})
        self.text_encoder = core.compile_model(txt, device, cfg)

        self.table = np.load(MODELS / "token_embeddings.npy", mmap_mode="r")
        self.proc = AutoProcessor.from_pretrained(MODEL_ID)
        self.video_id = self.proc.video_token_id
        self.pad_id = self.proc.tokenizer.pad_token_id

    # --- video ------------------------------------------------------------------------------
    def frame_tokens(self, pixel_values: np.ndarray, position_ids: np.ndarray) -> np.ndarray:
        """All frames of a clip -> (n_frames * 120, 512), in calls of `vision_batch` frames."""
        n, out = len(pixel_values), []
        for s in range(0, n, self.vision_batch):
            pv, pos = (
                pixel_values[s : s + self.vision_batch],
                position_ids[s : s + self.vision_batch],
            )
            real = len(pv)
            if real < self.vision_batch:  # pad with the last frame; its tokens are dropped
                k = self.vision_batch - real
                pv = np.concatenate([pv, np.repeat(pv[-1:], k, 0)])
                pos = np.concatenate([pos, np.repeat(pos[-1:], k, 0)])
            tok = self.vision([pv, pos])[0].reshape(self.vision_batch, TOK_PER_FRAME, HIDDEN)
            out.append(tok[:real].reshape(-1, HIDDEN))
        return np.concatenate(out)

    def encoder(self, n: int) -> ov.CompiledModel:
        """The encoder compiled for `enc_batch` sequences of length `n` (compiled on first use)."""
        if n not in self._encoders:
            enc = self._core.read_model(self._enc_xml)
            enc.reshape({0: [self.enc_batch, n, HIDDEN], 1: [self.enc_batch, n]})
            self._encoders[n] = self._core.compile_model(enc, self._device, self._cfg)
        return self._encoders[n]

    def prompt(self, frames: np.ndarray, **kwargs) -> dict:
        """The processor's output for one clip: token ids, patches, patch positions."""
        x = self.proc(
            text=["<|video|>"], videos=[frames], do_sample_frames=False, return_tensors="np",
            **kwargs,
        )  # fmt: skip
        assert len(x["input_ids"][0]) <= VIDEO_LEN, f"{len(x['input_ids'][0])} > {VIDEO_LEN}"
        assert (x["video_position_ids"][0] != -1).all(-1).sum() // 9 == TOK_PER_FRAME, "frame size"
        return x

    def assemble(self, ids: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        """Token ids + the frames' soft tokens (n_frames * 120, 512), in the order they should
        appear -> the encoder's input embeddings (L, 512)."""
        vmask = ids == self.video_id
        emb = self.table[np.where(vmask, self.pad_id, ids)].astype(np.float32)
        emb[vmask] = tokens
        return emb

    def video_inputs(self, frames: np.ndarray, **kwargs) -> np.ndarray:
        """Frames -> the encoder's input embeddings (L, 512), unpadded."""
        x = self.prompt(frames, **kwargs)
        tokens = self.frame_tokens(x["pixel_values_videos"], x["video_position_ids"])
        return self.assemble(x["input_ids"][0], tokens)

    def hidden(self, embs: list[np.ndarray]) -> list[np.ndarray]:
        """Unpadded sequences (L_i, 512) -> per-token encoder outputs (L_i, D), in calls of
        `enc_batch` sequences of one length bucket."""
        lengths = np.array([len(e) for e in embs])
        assert lengths.max() <= VIDEO_LEN, f"{lengths.max()} tokens > {VIDEO_LEN}"
        bucket = np.array([min(b for b in BUCKETS if b >= n) for n in lengths])
        assert (bucket - lengths < WINDOW).all()
        out: list[np.ndarray] = [np.empty(0)] * len(embs)
        for b in np.unique(bucket):
            idx = list(np.flatnonzero(bucket == b))
            for s in range(0, len(idx), self.enc_batch):
                i = idx[s : s + self.enc_batch]
                i_full = i + [i[-1]] * (self.enc_batch - len(i))  # pad the batch with repeats
                e = np.zeros((self.enc_batch, b, HIDDEN), np.float32)
                m = np.zeros((self.enc_batch, b), np.int64)
                for r, k in enumerate(i_full):
                    e[r, : lengths[k]], m[r, : lengths[k]] = embs[k], 1
                h = self.encoder(int(b))([e, m])[0]
                for r, k in enumerate(i):
                    out[k] = h[r, : lengths[k]]
        return out

    def encode(self, embs: list[np.ndarray]) -> np.ndarray:
        """Unpadded sequences (L_i, 512) -> (N, 768) unit vectors (masked mean, then L2)."""
        v = np.stack([h.mean(0) for h in self.hidden(embs)])
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def embed_videos(self, clips: list[np.ndarray]) -> np.ndarray:
        return self.encode([self.video_inputs(f) for f in clips])

    # --- text -------------------------------------------------------------------------------
    def embed_text(self, text: str) -> np.ndarray:
        ids = self.proc.tokenizer(QUERY_PREFIX + text)["input_ids"]
        emb = np.zeros((1, TEXT_LEN, HIDDEN), np.float32)
        emb[0, : len(ids)] = self.table[ids]
        mask = np.zeros((1, TEXT_LEN), np.int64)
        mask[0, : len(ids)] = 1
        return pooled(self.text_encoder([emb, mask])[0], mask)[0]
