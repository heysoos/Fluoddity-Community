"""Sampling an archive's thumbnails and embedding them, for the measurement
tools. Knows nothing about what is being measured.

Images are archive thumbnails upscaled to the encoder's input size. That
understates a fine-patch model's discrimination, but identically for every
model, so a comparison between them is fair.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

MODELS_ROOT = "models"
CLEARED_MARK = ".cleared-"

_ORT_DTYPES = {"tensor(float)": np.float32, "tensor(float16)": np.float16}


def recent_layouts(root, count: int) -> list[Path]:
    """The `count` most recently written layout directories holding thumbnails,
    newest first.

    Ranked on the layout that HAS entries, never on the archive: switching
    brain creates an empty sibling and writes to it, so the most recently
    touched directory is reliably one nothing was explored in.
    """
    found = []
    try:
        archives = [p for p in Path(root).iterdir()
                    if p.is_dir() and CLEARED_MARK not in p.name]
    except OSError:
        return []
    for archive in archives:
        try:
            subs = [p for p in archive.iterdir() if p.is_dir()]
        except OSError:
            continue
        for sub in subs:
            index = sub / "index.jsonl"
            thumbs = sub / "thumbs"
            if not index.is_file() or not thumbs.is_dir():
                continue
            if not any(thumbs.glob("*.jpg")):
                continue
            found.append((index.stat().st_mtime, str(sub), sub))
    # str(sub) breaks mtime ties deterministically - the test fixtures write
    # several layouts inside one filesystem timestamp tick.
    found.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return [p for _mt, _s, p in found[:count]]


def load_thumbs(layout_dir, n: int, px: int, rng) -> np.ndarray:
    """A random sample of one layout's thumbnails, resized to px."""
    files = sorted((Path(layout_dir) / "thumbs").glob("*.jpg"))
    if not files:
        return np.empty((0, px, px, 3), dtype=np.uint8)
    if len(files) > n:
        files = [files[i] for i in sorted(rng.choice(len(files), n, replace=False))]
    out = np.empty((len(files), px, px, 3), dtype=np.uint8)
    keep = 0
    for f in files:
        try:
            img = Image.open(f).convert("RGB")
        except OSError:
            continue        # one unreadable thumbnail is not a failed run
        if img.size != (px, px):
            img = img.resize((px, px), Image.BILINEAR)
        out[keep] = np.asarray(img, dtype=np.uint8)
        keep += 1
    return out[:keep]


def nn_distance(emb: np.ndarray, block: int = 512) -> np.ndarray:
    """Cosine distance from each row to its nearest OTHER row.

    Blocked over query rows: the full (n, n) similarity matrix is scratch, and
    only one column of it survives per query.
    """
    e = np.asarray(emb, dtype=np.float32)
    n = len(e)
    out = np.empty(n, dtype=np.float32)
    for i in range(0, n, block):
        sim = e[i:i + block] @ e.T
        for j in range(len(sim)):
            sim[j, i + j] = -np.inf          # never itself
        out[i:i + block] = 1.0 - sim.max(axis=1)
    return out


class Encoder:
    """One ONNX vision tower, timed. Takes a VisionModel, so preprocessing and
    input size cannot drift from the registry."""

    def __init__(self, model, providers):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.log_severity_level = 3
        self.model = model
        self.key = model.key
        self.px = model.px
        path = Path(MODELS_ROOT) / model.subdir / "vision_model_fp16.onnx"
        self.sess = ort.InferenceSession(str(path), opts, providers=providers)
        self.provider = self.sess.get_providers()[0]

        vin = self.sess.get_inputs()[0]
        self.in_name = vin.name
        self.dtype = _ORT_DTYPES.get(vin.type, np.float32)
        self.out_name = self._pick_output()

        mean = np.asarray(model.mean, dtype=np.float32)
        std = np.asarray(model.std, dtype=np.float32)
        self._scale = (1.0 / 255.0 / std).reshape(1, 3, 1, 1)
        self._offset = (-mean / std).reshape(1, 3, 1, 1)

        self.seconds = 0.0
        self.images = 0
        self.dim = None

    def _pick_output(self) -> str:
        """Shared with the scorer rather than reimplemented - picking the wrong
        output surfaces as a matmul error far from its cause."""
        from services.vision_scorer import pick_embedding_output

        return pick_embedding_output(self.sess.get_outputs(),
                                     ("image_embeds", "pooler_output"))

    def embed(self, images: np.ndarray, chunk: int = 64) -> np.ndarray:
        """uint8 (N,px,px,3) -> float32 (N,D) L2-normalised."""
        parts = []
        for i in range(0, len(images), chunk):
            block = images[i:i + chunk]
            x = np.ascontiguousarray(block.transpose(0, 3, 1, 2)).astype(np.float32)
            x *= self._scale
            x += self._offset
            x = x.astype(self.dtype)
            t0 = time.perf_counter()
            out = self.sess.run([self.out_name], {self.in_name: x})[0]
            self.seconds += time.perf_counter() - t0
            self.images += len(block)
            parts.append(out.astype(np.float32))
        e = np.concatenate(parts, axis=0)
        self.dim = e.shape[1]
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)

    @property
    def ms_per_image(self) -> float:
        return 1000.0 * self.seconds / max(1, self.images)


def build_encoders(keys, providers=None) -> list[Encoder]:
    """One Encoder per key whose weights are on disk. Missing models are
    skipped with a note rather than raising, so a run with only the default
    encoder downloaded still works."""
    import onnxruntime as ort

    from services.vision_models import get

    if providers is None:
        have = set(ort.get_available_providers())
        providers = [p for p in ("DmlExecutionProvider", "CPUExecutionProvider")
                     if p in have] or ["CPUExecutionProvider"]
    out = []
    for key in keys:
        model = get(key)
        path = Path(MODELS_ROOT) / model.subdir / "vision_model_fp16.onnx"
        if not path.is_file():
            print(f"[skip] {key}: {path} not present")
            continue
        enc = Encoder(model, providers)
        print(f"[load] {key:12s} provider={enc.provider}  out={enc.out_name}")
        out.append(enc)
    return out
