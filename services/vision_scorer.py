"""Goal scoring for the automatic tournament: a text prompt or a picture.

Depends only on onnxruntime, tokenizers, numpy and PIL. Knows nothing about
tournaments, tiles or OpenGL. Everything model-specific - paths, preprocessing,
tokenizer context, logit scale - comes from services/vision_models.py.

ONNX tensor names, verified against Xenova/clip-vit-base-patch32:
  vision: in 'pixel_values' (N,3,224,224) tensor(float) -> out 'image_embeds'
  text:   in 'input_ids'    (N,77) int64               -> out 'text_embeds'

Three non-obvious facts about these exports:
  - the fp16 files have fp16 WEIGHTS but float32 I/O, so the input dtype is read
    from the session descriptor rather than assumed
  - loading them with the default ORT_ENABLE_ALL crashes on the CPU provider
    (SimplifiedLayerNormFusion), which would break the required CPU fallback,
    so both sessions pin ORT_ENABLE_BASIC
  - the pooled image embedding is not always output 0: SigLIP's export puts
    last_hidden_state first, so the output is chosen by name
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from services.vision_models import DEFAULT_KEY, get
from utilities.paths import get_models_root

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
    # Without this one, nearly every trail pattern scores highly on nearly
    # every prompt - "abstract texture" is the honest description of most of
    # the search space.
    "an abstract texture",
    # The last three exist because a DEAD CANVAS is a degenerate attractor:
    # without them a pure black image can outrank genuine tiles and the
    # optimizer drives straight to an empty simulation. "a blank image" and
    # "a solid color" alone did not catch it.
    "a black image",
    "an empty black background",
    "a dark empty scene",
]


DEFAULT_CROP = (0.5, 0.5, 1.0)


def load_goal_image(path: str, px: int, crop=DEFAULT_CROP) -> np.ndarray:
    """A picture on disk -> uint8 (1, px, px, 3), a square crop of it.

    `crop` is (x, y, zoom): the square's side is the shorter side over zoom,
    and x, y in 0..1 slide it across whatever room is left on each axis. The
    default is the centre crop. Nothing is squeezed.
    """
    cx, cy, zoom = crop
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = max(1, int(round(min(w, h) / max(1.0, float(zoom)))))
    left = int(round((w - side) * min(1.0, max(0.0, float(cx)))))
    top = int(round((h - side) * min(1.0, max(0.0, float(cy)))))
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((int(px), int(px)), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)[None]


def to_grayscale(images: np.ndarray) -> np.ndarray:
    """uint8 (B, px, px, 3) -> the same shape with every channel the luma."""
    x = np.asarray(images, dtype=np.float32)
    luma = x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114
    g = np.clip(np.rint(luma), 0, 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=-1)


def distractor_images(px: int) -> np.ndarray:
    """The pictures an image goal is scored against: black, white, mid-grey and
    seeded uniform noise, uint8 (4, px, px, 3). Same arrays every call."""
    px = int(px)
    flat = [np.full((px, px, 3), v, dtype=np.uint8) for v in (0, 255, 128)]
    noise = np.random.default_rng(0).integers(0, 256, (px, px, 3), dtype=np.uint8)
    return np.stack(flat + [noise], axis=0)


def _scale_offset(model):
    """The normalisation folded into one multiply-add, in NCHW, so it can be
    applied in place after the transpose instead of as three whole-array
    passes."""
    mean = np.asarray(model.mean, dtype=np.float32)
    std = np.asarray(model.std, dtype=np.float32)
    return ((1.0 / 255.0 / std).reshape(1, 3, 1, 1).astype(np.float32),
            (-mean / std).reshape(1, 3, 1, 1).astype(np.float32))


def preprocess(crops: np.ndarray, model, dtype=np.float32) -> np.ndarray:
    """uint8 (B,px,px,3) -> (B,3,px,px) in `dtype`, C-contiguous.

    TRANSPOSE FIRST, while the data is still uint8 - this was the single
    largest cost in the scoring path. Folds normalisation into one in-place
    multiply-add rather than three full-array passes.

    In float32 this differs from the old per-channel formula by at most one
    ulp (associativity), which matters because every embedding already in an
    archive was produced by the old order. Not done in fp16 throughout: numpy
    emulates fp16 arithmetic and that measured slower, not faster.

    ascontiguousarray is required, not cosmetic: a non-contiguous array either
    forces a silent copy inside ONNX Runtime or errors, depending on provider.
    """
    scale, offset = _scale_offset(model)
    x = np.ascontiguousarray(crops.transpose(0, 3, 1, 2)).astype(np.float32)
    x *= scale
    x += offset
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


def pick_embedding_output(outputs, preferred) -> str:
    """The POOLED embedding among an ONNX session's outputs.

    Never output 0 by position: SigLIP's exports put last_hidden_state first on
    both towers, which is a (N, tokens, dim) tensor, and the mismatch only
    surfaces as a matmul error much further downstream.
    """
    names = {o.name for o in outputs}
    for pref in preferred:
        if pref in names:
            return pref
    for o in outputs:
        if len(o.shape) == 2:
            return o.name
    return outputs[0].name


class VisionScorer:
    """Turns a goal - a text prompt or a picture - into a per-image fitness.

    ONE goal at a time: a target at row 0 of `_goal_emb`, the references after
    it, and the logit scale that goes with them. A text goal is scored against
    the text distractors at the text scale; a picture against the synthesized
    pictures at the image scale, or against nothing at all, which is the plain
    cosine.
    """

    MAX_CHUNK = 64  # images per session.run; 64 fp32 NCHW inputs ~= 38 MB

    def __init__(self, model_key: str = DEFAULT_KEY, providers=None,
                 n_views: int = 3, seed: int = 0):
        self._available = False
        self._model = get(model_key)
        self._goal_emb = None
        self._goal_scale = None
        self._distractor_emb = {}      # grayscale flag -> (4, dim)
        self._prompt = ""
        self._n_views = n_views
        self._rng = np.random.default_rng(seed)
        self._pool = None            # made on first multi-chunk embed

        import onnxruntime as ort
        from tokenizers import Tokenizer

        d = get_models_root() / self._model.subdir
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
        self._outputs = self._vision.get_outputs()
        self._vision_out = self._pick_output()
        self._text_in = self._text.get_inputs()[0].name
        self._text_out = pick_embedding_output(
            self._text.get_outputs(), ("text_embeds", "pooler_output"))

        self._tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        self._tokenizer.enable_truncation(self._model.context)
        self._tokenizer.enable_padding(length=self._model.context)

        self._available = True

    @property
    def available(self) -> bool:
        return self._available

    @property
    def model(self):
        return self._model

    @property
    def prompt(self) -> str:
        return self._prompt

    def _pick_output(self) -> str:
        """The pooled IMAGE embedding."""
        return pick_embedding_output(self._outputs,
                                     ("image_embeds", "pooler_output"))

    def set_prompt(self, text: str, distractors: list[str] | None = None) -> None:
        """Embed the target prompt (index 0) plus distractors. Cached until the
        next call, so retyping the goal costs one text-encoder pass."""
        prompts = [text] + list(
            DEFAULT_DISTRACTORS if distractors is None else distractors
        )
        self._goal_emb = self.embed_text(prompts)
        self._goal_scale = self._model.text_logit_scale
        self._prompt = text

    def set_image_goal(self, image: np.ndarray, distractors: bool = True,
                       grayscale: bool = False) -> None:
        """Make a picture the goal: uint8 (1, px, px, 3), see load_goal_image.

        The reference is embedded as the untouched frame - the user chose the
        framing - and the synthesized distractors once per scorer. With
        `distractors` off the goal is the target alone and score() is the
        cosine to it. `grayscale` reads the picture and the distractors by
        luma, to match a capture rendered as density.
        """
        target = np.asarray(image, dtype=np.uint8)
        if grayscale:
            target = to_grayscale(target)
        rows = [self.embed(target, n_views=1)]
        if distractors:
            rows.append(self._distractor_embeddings(bool(grayscale)))
        self._goal_emb = np.concatenate(rows, axis=0)
        self._goal_scale = self._model.image_logit_scale
        self._prompt = ""

    def _distractor_embeddings(self, grayscale: bool = False) -> np.ndarray:
        # getattr, because the tests build a scorer without running __init__.
        cache = getattr(self, "_distractor_emb", None)
        if not isinstance(cache, dict):
            cache = {}
            self._distractor_emb = cache
        if grayscale not in cache:
            pics = distractor_images(self._model.px)
            if grayscale:
                pics = to_grayscale(pics)
            cache[grayscale] = self._embed_images(pics)
        return cache[grayscale]

    def embed_text(self, prompts: list[str]) -> np.ndarray:
        """(P,) strings -> float32 (P, dim), L2-normalised.

        Deliberately does NOT write self._goal_emb: the exploration archive's
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
        GPU runs the current one. numpy releases the GIL for work this size, so
        they genuinely overlap.

        Lookahead of exactly ONE chunk - submitting every chunk up front would
        hold the whole generation preprocessed in memory at once, to save
        nothing beyond the first chunk's latency.

        Only preprocess() moves off the main thread. augment() draws from
        self._rng and must not race, and only one thread ever calls
        session.run.
        """
        starts = list(range(0, len(crops), self.MAX_CHUNK))
        if len(starts) <= 1:
            batch = preprocess(crops, self._model, self._vision_dtype)
            out = self._vision.run([self._vision_out], {self._vision_in: batch})[0]
            return _l2(out.astype(np.float32))

        ex = self._prep_pool()
        c = self.MAX_CHUNK

        def prep(s):
            return ex.submit(preprocess, crops[s:s + c], self._model,
                             self._vision_dtype)

        pending = prep(starts[0])
        chunks = []
        for j, _s in enumerate(starts):
            batch = pending.result()
            pending = prep(starts[j + 1]) if j + 1 < len(starts) else None
            out = self._vision.run([self._vision_out], {self._vision_in: batch})[0]
            chunks.append(out.astype(np.float32))
        return _l2(np.concatenate(chunks, axis=0))

    def embed(self, images: np.ndarray, n_views: int | None = None) -> np.ndarray:
        """uint8 (B,px,px,3) -> float32 (B*n_views, dim), L2-normalised.

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
        """uint8 (B,px,px,3) -> float32 (B, dim): ONE embedding per image,
        averaged over n_views and renormalised.

        The averaging is the point: embed() returns views un-reduced, which
        would make sub-crops of one tile into competing archive descriptors.
        These encoders are strongly position-dependent (a roll on the torus
        moves the embedding well past the separation bar), and averaging over
        random sub-crops buys back most of that invariance. See CLAUDE.md.
        """
        b = np.asarray(images)
        v = max(1, int(n_views))
        e = self.embed(b, v)
        if v == 1:
            return e
        return _l2(e.reshape(len(b), v, -1).mean(axis=1))

    def score(self, images: np.ndarray) -> np.ndarray:
        """uint8 (B,px,px,3) -> float32 (B,), averaged over augmented views.

        Softmax probability of the target against the goal's references; with
        no references, the cosine to the target.
        """
        if self._goal_emb is None:
            raise RuntimeError(
                "set_prompt() or set_image_goal() must be called before score()")
        b = len(images)
        emb = self.embed(images, self._n_views)
        if len(self._goal_emb) == 1:
            cos = emb @ self._goal_emb[0]
            return cos.reshape(b, -1).mean(axis=1).astype(np.float32)
        logits = float(self._goal_scale) * (emb @ self._goal_emb.T)
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        target = probs[:, 0].reshape(b, -1)  # (B, n_views)
        return target.mean(axis=1).astype(np.float32)
