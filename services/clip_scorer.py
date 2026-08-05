"""CLIP text-image scoring for the automatic tournament.

Depends only on onnxruntime, tokenizers, numpy and PIL. Knows nothing about
tournaments, tiles or OpenGL.

ONNX tensor names, verified against Xenova/clip-vit-base-patch32 on 2026-08-04:
  vision: in 'pixel_values' (N,3,224,224) tensor(float) -> out 'image_embeds'
  text:   in 'input_ids'    (N,77) int64               -> out 'text_embeds'

Two non-obvious facts about these exports:
  - the fp16 files have fp16 WEIGHTS but float32 I/O, so the input dtype is read
    from the session descriptor rather than assumed
  - loading them with the default ORT_ENABLE_ALL crashes on the CPU provider
    (SimplifiedLayerNormFusion), which would break the required CPU fallback,
    so both sessions pin ORT_ENABLE_BASIC
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# CLIP preprocessing constants, from preprocessor_config.json
CLIP_MEAN = np.array([0.48145466, 0.45782750, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# CLIP's learned logit_scale.exp()
LOGIT_SCALE = 100.0

CONTEXT_LENGTH = 77

_ORT_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
}

DEFAULT_DISTRACTORS = [
    "a blank image",
    "random noise",
    "a blurry photograph",
    "a screenshot of text",
    "a solid color",
    # Without this one, nearly every trail pattern scores highly on nearly every
    # prompt, because "abstract texture" is the honest description of most of
    # the search space.
    "an abstract texture",
    # The last three exist because a DEAD CANVAS is a degenerate attractor.
    # Measured on 32 real tiles (Task 3 gate): with only the six distractors
    # above, a pure black image scored 0.37 on "flowing water" and 0.35 on
    # "tree branches", outranking 31 of 32 genuine tiles. The optimizer would
    # have driven straight to an empty simulation. "a blank image" and "a solid
    # color" alone did not catch it; these do, dropping black to 0.03 while
    # leaving real-signal spread essentially unchanged.
    "a black image",
    "an empty black background",
    "a dark empty scene",
]


def preprocess(crops: np.ndarray, dtype=np.float32) -> np.ndarray:
    """uint8 (B,224,224,3) -> (B,3,224,224) in `dtype`, C-contiguous."""
    x = crops.astype(np.float32) / 255.0
    x = (x - CLIP_MEAN) / CLIP_STD
    # ascontiguousarray is required, not cosmetic: a non-contiguous array either
    # forces a silent copy inside ONNX Runtime or errors, depending on provider.
    x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))
    return x.astype(dtype)


def augment(crops: np.ndarray, n_views: int, rng) -> np.ndarray:
    """Expand each image into n_views. View 0 is the untouched frame; the rest
    are random 160-224px sub-crops resampled back to 224.

    Averaging over random views is the standard defense against a search process
    finding adversarial texture that satisfies one fixed view.
    Output is image-major: [img0 v0, img0 v1, ..., img1 v0, ...].
    """
    if n_views <= 1:
        return crops
    b, h, w, _ = crops.shape
    out = np.empty((b * n_views, h, w, 3), dtype=np.uint8)
    for i in range(b):
        out[i * n_views] = crops[i]
        for v in range(1, n_views):
            size = int(rng.integers(160, h + 1))
            y = int(rng.integers(0, h - size + 1))
            x = int(rng.integers(0, w - size + 1))
            patch = crops[i, y:y + size, x:x + size]
            if size == h:
                out[i * n_views + v] = patch
            else:
                img = Image.fromarray(patch).resize((w, h), Image.BILINEAR)
                out[i * n_views + v] = np.asarray(img)
    return out


def _l2(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)


class CLIPScorer:
    """Turns a text prompt into a per-image fitness in [0, 1]."""

    MAX_CHUNK = 64  # images per session.run; 64 fp32 NCHW inputs ~= 38 MB

    def __init__(self, model_dir, providers=None, n_views: int = 3, seed: int = 0):
        self._available = False
        self._text_emb = None
        self._prompt = ""
        self._n_views = n_views
        self._rng = np.random.default_rng(seed)

        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = Path(model_dir)
        if providers is None:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        have = set(ort.get_available_providers())
        providers = [p for p in providers if p in have] or ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        # ORT_ENABLE_ALL crashes the fp16 vision model on the CPU provider.
        # Pinned for both providers so they cannot diverge in behaviour.
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.log_severity_level = 3

        self._vision = ort.InferenceSession(
            str(d / "vision_model_fp16.onnx"), opts, providers=providers
        )
        self._text = ort.InferenceSession(
            str(d / "text_model_fp16.onnx"), opts, providers=providers
        )

        vin = self._vision.get_inputs()[0]
        self._vision_in = vin.name
        self._vision_dtype = _ORT_DTYPES.get(vin.type, np.float32)
        self._vision_out = self._vision.get_outputs()[0].name
        self._text_in = self._text.get_inputs()[0].name
        self._text_out = self._text.get_outputs()[0].name

        self._tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        self._tokenizer.enable_truncation(CONTEXT_LENGTH)
        self._tokenizer.enable_padding(length=CONTEXT_LENGTH)

        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    @property
    def prompt(self) -> str:
        return self._prompt

    def set_prompt(self, text: str, distractors: list[str] | None = None) -> None:
        """Embed the target prompt (index 0) plus distractors. Cached until the
        next call, so retyping the goal costs one text-encoder pass."""
        prompts = [text] + list(
            DEFAULT_DISTRACTORS if distractors is None else distractors
        )
        encoded = self._tokenizer.encode_batch(prompts)
        feed = {self._text_in: np.array([e.ids for e in encoded], dtype=np.int64)}
        for inp in self._text.get_inputs()[1:]:
            if inp.name == "attention_mask":
                feed["attention_mask"] = np.array(
                    [e.attention_mask for e in encoded], dtype=np.int64
                )
        emb = self._text.run([self._text_out], feed)[0].astype(np.float32)
        self._text_emb = _l2(emb)
        self._prompt = text

    def _embed_images(self, crops: np.ndarray) -> np.ndarray:
        chunks = []
        for i in range(0, len(crops), self.MAX_CHUNK):
            batch = preprocess(crops[i:i + self.MAX_CHUNK], self._vision_dtype)
            out = self._vision.run([self._vision_out], {self._vision_in: batch})[0]
            chunks.append(out.astype(np.float32))
        return _l2(np.concatenate(chunks, axis=0))

    def score(self, images: np.ndarray) -> np.ndarray:
        """uint8 (B,224,224,3) -> float32 (B,). Softmax probability of the
        target prompt against the distractor set, averaged over augmented views.
        """
        if self._text_emb is None:
            raise RuntimeError("set_prompt() must be called before score()")
        b = len(images)
        views = augment(images, self._n_views, self._rng)
        emb = self._embed_images(views)
        logits = LOGIT_SCALE * (emb @ self._text_emb.T)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        target = probs[:, 0].reshape(b, -1)  # (B, n_views)
        return target.mean(axis=1).astype(np.float32)
