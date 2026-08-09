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


# The normalisation folded into one multiply-add, in NCHW, so it can be applied
# in place after the transpose instead of as three whole-array passes.
_SCALE_CHW = (1.0 / 255.0 / CLIP_STD).reshape(1, 3, 1, 1).astype(np.float32)
_OFFSET_CHW = (-CLIP_MEAN / CLIP_STD).reshape(1, 3, 1, 1).astype(np.float32)


def preprocess(crops: np.ndarray, dtype=np.float32) -> np.ndarray:
    """uint8 (B,224,224,3) -> (B,3,224,224) in `dtype`, C-contiguous.

    TRANSPOSE FIRST, while the data is still uint8. This was the single largest
    cost in the whole CLIP path - 54-58% of it, more than the GPU - because the
    old order converted to float32, made two more full-array temporaries for
    the mean and the std, and only then did a strided transpose of 38 MB per
    64 images. Transposing one byte per element instead of four, and folding
    the normalisation into one in-place multiply-add, measures 2.3x faster
    (112.6 ms -> 48.7 ms per 64 images).

    EXACT in fp16, which is the dtype the shipped model takes: 0 of 4.8M values
    differ from the old formula. In float32 it differs on 67% of values, but by
    at most 4.8e-7 - one ulp, from `x*(1/std) + (-mean/std)` associating
    differently to `(x - mean)/std`. Worth stating precisely, because every
    embedding already in an archive was produced by the old order.

    Not done in fp16 throughout, which looks tempting since the model is fp16:
    numpy emulates fp16 arithmetic, so it measured 150.6 ms - slower than the
    original - and lost 0.002 of accuracy for the privilege.

    ascontiguousarray is required, not cosmetic: a non-contiguous array either
    forces a silent copy inside ONNX Runtime or errors, depending on provider.
    """
    x = np.ascontiguousarray(crops.transpose(0, 3, 1, 2)).astype(np.float32)
    x *= _SCALE_CHW
    x += _OFFSET_CHW
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
        self._pool = None            # made on first multi-chunk embed

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
        self._text_emb = self.embed_text(prompts)
        self._prompt = text

    def embed_text(self, prompts: list[str]) -> np.ndarray:
        """(P,) strings -> float32 (P, 512), L2-normalised.

        Deliberately does NOT write self._text_emb: the exploration archive's
        goal embeddings and score()'s prompt+distractor cache are different
        things and must not be able to clobber each other.
        """
        encoded = self._tokenizer.encode_batch(list(prompts))
        feed = {self._text_in: np.array([e.ids for e in encoded], dtype=np.int64)}
        for inp in self._text.get_inputs()[1:]:
            if inp.name == "attention_mask":
                feed["attention_mask"] = np.array(
                    [e.attention_mask for e in encoded], dtype=np.int64
                )
        emb = self._text.run([self._text_out], feed)[0].astype(np.float32)
        return _l2(emb)

    def _prep_pool(self):
        """One worker, made on first use. Idle between generations, so it costs
        a parked thread and nothing else."""
        # getattr, because the tests build a scorer without running __init__.
        if getattr(self, "_pool", None) is None:
            from concurrent.futures import ThreadPoolExecutor

            self._pool = ThreadPoolExecutor(max_workers=1,
                                            thread_name_prefix="clip-prep")
        return self._pool

    def _embed_images(self, crops: np.ndarray) -> np.ndarray:
        """PIPELINED: the next chunk is normalised on a worker thread while the
        GPU runs the current one.

        The two stages are close in cost after the preprocess rewrite - 48.7 ms
        of numpy against 58.4 ms of DirectML per 64 images - and numpy releases
        the GIL for work this size, so they genuinely overlap. Measured 1.39x
        at 1152 images, which is one generation at grid 8, 6 snapshots, 3 views.

        A lookahead of exactly ONE chunk. Submitting every chunk up front is
        simpler and marginally faster, but it would hold the whole generation
        preprocessed in memory at once - 345 MB at grid 8 - to save nothing
        beyond the first chunk's latency.

        Only preprocess() moves off the main thread. augment() draws from
        self._rng and must not race, and only one thread ever calls
        session.run.
        """
        starts = list(range(0, len(crops), self.MAX_CHUNK))
        if len(starts) <= 1:
            batch = preprocess(crops, self._vision_dtype)
            out = self._vision.run([self._vision_out], {self._vision_in: batch})[0]
            return _l2(out.astype(np.float32))

        ex = self._prep_pool()
        c = self.MAX_CHUNK

        def prep(s):
            return ex.submit(preprocess, crops[s:s + c], self._vision_dtype)

        pending = prep(starts[0])
        chunks = []
        for j, _s in enumerate(starts):
            batch = pending.result()
            pending = prep(starts[j + 1]) if j + 1 < len(starts) else None
            out = self._vision.run([self._vision_out], {self._vision_in: batch})[0]
            chunks.append(out.astype(np.float32))
        return _l2(np.concatenate(chunks, axis=0))

    def embed(self, images: np.ndarray, n_views: int | None = None) -> np.ndarray:
        """uint8 (B,224,224,3) -> float32 (B*n_views, 512), L2-normalised.

        Output is image-major: [img0 v0, img0 v1, ..., img1 v0, ...].
        n_views=None uses the instance default.

        n_views=1 skips augmentation entirely. Augmentation is an
        anti-adversarial defence for a DIRECTED objective; for novelty search it
        would turn three random sub-crops of one tile into three competing
        archive descriptors for one behaviour.
        """
        v = self._n_views if n_views is None else int(n_views)
        return self._embed_images(augment(images, v, self._rng))

    def embed_mean(self, images: np.ndarray, n_views: int = 3) -> np.ndarray:
        """uint8 (B,224,224,3) -> float32 (B, 512): ONE embedding per image,
        averaged over n_views and renormalised.

        The averaging is the point. embed() returns the views un-reduced, and
        handing those to a novelty archive would make three sub-crops of one
        tile into three competing descriptors for one behaviour - which is why
        the search used n_views=1 and took the raw frame.

        But CLIP ViT-B/32 is strongly position-dependent: measured 2026-08-09 on
        real archive thumbnails, shifting one 16px on the torus moves its
        embedding 0.078-0.088, which is 2.5-2.7x the distance to its nearest
        genuine neighbour and past the 0.02 separation bar for 100% of tiles.
        Averaging over random sub-crops buys back some of that invariance:

            views   roll 16px   repeat noise   same/unrelated
                1      0.0784        0.0000            0.459
                3      0.0319        0.0069            0.425
                5      0.0236        0.0050            0.445
                8      0.0197        0.0037            0.424

        `repeat` is what the averaging COSTS - the same image embedded twice no
        longer agrees, because views 1..n are random draws. At 3 views the
        nuisance falls by 0.046 and the new noise is 0.007, so the trade is
        about 7:1 in favour; and 0.0069 is a third of the separation bar, so a
        true duplicate still cannot pass it on noise alone.

        (Centring on the centre of mass was measured first and rejected: it
        cancels a shift exactly, but 26-34% of tiles have no well-posed centre -
        these are space-filling textures, not localised objects - so it pushed
        real near-duplicates APART by 15-19%.)
        """
        b = np.asarray(images)
        v = max(1, int(n_views))
        e = self.embed(b, v)
        if v == 1:
            return e
        return _l2(e.reshape(len(b), v, -1).mean(axis=1))

    def score(self, images: np.ndarray) -> np.ndarray:
        """uint8 (B,224,224,3) -> float32 (B,). Softmax probability of the
        target prompt against the distractor set, averaged over augmented views.
        """
        if self._text_emb is None:
            raise RuntimeError("set_prompt() must be called before score()")
        b = len(images)
        emb = self.embed(images, self._n_views)
        logits = LOGIT_SCALE * (emb @ self._text_emb.T)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        target = probs[:, 0].reshape(b, -1)  # (B, n_views)
        return target.mean(axis=1).astype(np.float32)
