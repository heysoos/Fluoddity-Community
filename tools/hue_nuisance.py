"""How much of the archive's novelty signal is spent on hue.

Rotates each sampled thumbnail's HSV hue and measures how far that moves its
CLIP embedding, against the distance to a genuinely different entry. Reports
one scale-free ratio per encoder, plus wall-clock cost per image, so the hue
question and the "is a better encoder affordable" question are answered off one
pass.

The ratio is scale-free on purpose: `min_separation` is calibrated in one
encoder's space and means nothing in another's. A ratio at or above 1 means a
pure hue rotation looks at least as different as a new creature. The figures
this produced live in the colour caveat in CLAUDE.md.

    python -m tools.hue_nuisance --entries 700

Vision towers only - this never loads a text encoder. Each is expected at
`models/<subdir>/vision_model_fp16.onnx`; missing ones are skipped, so the
default CLIP B/32 alone is a valid run. Images come from archive thumbnails
upscaled to the encoder's input size, which understates a fine-patch model's
discrimination but is identical for every model, so the comparison is fair.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from utilities.paths import get_archives_root

CLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
HALF = (0.5, 0.5, 0.5)

# (key, subdir, mean, std, input px)
MODELS = [
    ("clip-b32", "clip-vit-b32", CLIP_MEAN, CLIP_STD, 224),
    ("clip-b16", "clip-vit-b16", CLIP_MEAN, CLIP_STD, 224),
    ("siglip2-b16", "siglip2-b16-224", HALF, HALF, 224),
    ("clip-l14", "clip-vit-l14", CLIP_MEAN, CLIP_STD, 224),
]

# 74 deg is |dhue| at expedition_sigma with the gain at 0.5 - the angle the
# search actually produces, and the one the caveat quotes.
ANGLES = (15.0, 30.0, 60.0, 74.0, 90.0, 180.0)

MIN_SEPARATION = 0.02  # the live admission bar; valid for clip-b32 only

_ORT_DTYPES = {"tensor(float)": np.float32, "tensor(float16)": np.float16}


# ---- hue rotation -------------------------------------------------------

def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """(N,H,W,3) float32 in [0,1] -> HSV, h in [0,1)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    d = mx - mn
    safe_d = np.where(d == 0, 1.0, d)

    h = np.zeros_like(mx)
    h = np.where(mx == r, ((g - b) / safe_d) % 6.0, h)
    h = np.where(mx == g, (b - r) / safe_d + 2.0, h)
    h = np.where(mx == b, (r - g) / safe_d + 4.0, h)
    h = np.where(d == 0, 0.0, h / 6.0)

    s = np.where(mx == 0, 0.0, d / np.where(mx == 0, 1.0, mx))
    return np.stack([h, s, mx], axis=-1)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Inverse of rgb_to_hsv. -> (N,H,W,3) float32 in [0,1]."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    idx = (i % 6).astype(np.int32)

    sel = [idx == 0, idx == 1, idx == 2, idx == 3, idx == 4]
    r = np.select(sel, [v, q, p, p, t], v)
    g = np.select(sel, [t, v, v, q, p], p)
    b = np.select(sel, [p, p, t, v, v], q)
    return np.stack([r, g, b], axis=-1)


def rotate_hue_all(images: np.ndarray, angles) -> dict:
    """uint8 (N,H,W,3) -> {angle: uint8}, hue shifted, S and V untouched.

    An HSV shift, not an RGB-space matrix: `e.color.x` IS an HSV hue read
    modulo 1, so this is the faithful analogue of a mutation sliding it. The
    forward RGB->HSV runs once for all angles; only the inverse repeats.
    """
    hsv = rgb_to_hsv(images.astype(np.float32) / 255.0)
    h0 = hsv[..., 0].copy()
    out = {}
    for a in angles:
        hsv[..., 0] = (h0 + a / 360.0) % 1.0
        out[a] = np.clip(hsv_to_rgb(hsv) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return out


# ---- encoders -----------------------------------------------------------

class Encoder:
    """One ONNX vision tower. Times only session.run."""

    def __init__(self, key: str, path: Path, mean, std, px: int, providers):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.log_severity_level = 3
        self.key = key
        self.px = px
        self.sess = ort.InferenceSession(str(path), opts, providers=providers)
        self.provider = self.sess.get_providers()[0]

        vin = self.sess.get_inputs()[0]
        self.in_name = vin.name
        self.dtype = _ORT_DTYPES.get(vin.type, np.float32)
        self.out_name = self._pick_output()

        m = np.asarray(mean, dtype=np.float32)
        s = np.asarray(std, dtype=np.float32)
        self._scale = (1.0 / 255.0 / s).reshape(1, 3, 1, 1)
        self._offset = (-m / s).reshape(1, 3, 1, 1)

        self.seconds = 0.0
        self.images = 0
        self.dim = None

    def _pick_output(self) -> str:
        """The pooled image embedding. CLIP exports name it image_embeds,
        SigLIP's is pooler_output; falling back to output 0 by position would
        pick last_hidden_state on some exports."""
        outs = self.sess.get_outputs()
        by_name = {o.name: o for o in outs}
        for pref in ("image_embeds", "pooler_output"):
            if pref in by_name:
                return pref
        for o in outs:
            if len(o.shape) == 2:
                return o.name
        return outs[0].name

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


# ---- data ---------------------------------------------------------------

def recent_layouts(root: Path, count: int) -> list[Path]:
    """The `count` most recently written layout directories that hold
    thumbnails, newest first.

    Ranked on the layout that has entries, never on the archive: switching
    brain creates an EMPTY sibling and writes to it, so the most recently
    touched directory is reliably one nothing was explored in.
    """
    found = []
    try:
        archives = [p for p in root.iterdir()
                    if p.is_dir() and ".cleared-" not in p.name]
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
            found.append((index.stat().st_mtime, sub))
    found.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in found[:count]]


def load_thumbs(layout_dir: Path, n: int, px: int, rng) -> np.ndarray:
    """A random sample of one layout's thumbnails, resized to px."""
    files = sorted((layout_dir / "thumbs").glob("*.jpg"))
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
    n = len(emb)
    out = np.empty(n, dtype=np.float32)
    for i in range(0, n, block):
        sim = emb[i:i + block] @ emb.T
        for j in range(len(sim)):
            sim[j, i + j] = -np.inf          # never itself
        out[i:i + block] = 1.0 - sim.max(axis=1)
    return out


# ---- main ---------------------------------------------------------------

def build_encoders(models_root: Path, only: set) -> list:
    import onnxruntime as ort

    have = set(ort.get_available_providers())
    providers = [p for p in ("DmlExecutionProvider", "CPUExecutionProvider")
                 if p in have] or ["CPUExecutionProvider"]
    encoders = []
    for key, sub, mean, std, px in MODELS:
        if only and key not in only:
            continue
        path = models_root / sub / "vision_model_fp16.onnx"
        if not path.is_file():
            print(f"[skip] {key}: {path} not present")
            continue
        enc = Encoder(key, path, mean, std, px, providers)
        print(f"[load] {key:12s} provider={enc.provider}  out={enc.out_name}")
        encoders.append(enc)
    return encoders


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entries", type=int, default=700, help="per archive")
    ap.add_argument("--archives", type=int, default=3, help="most recent N")
    ap.add_argument("--models-root", default="models")
    ap.add_argument("--only", default="", help="comma-separated model keys")
    ap.add_argument("--out", default="hue_nuisance.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    encoders = build_encoders(Path(args.models_root).resolve(),
                              {k.strip() for k in args.only.split(",") if k.strip()})
    if not encoders:
        raise SystemExit("no models available")
    px = encoders[0].px
    if any(e.px != px for e in encoders):
        raise SystemExit("models disagree on input size")

    layouts = recent_layouts(get_archives_root(), args.archives)
    if not layouts:
        raise SystemExit("no archive holds any thumbnails")

    rng = np.random.default_rng(args.seed)
    results = {"angles": list(ANGLES), "entries_per_archive": args.entries,
               "archives": {}, "timing": {}}
    started = time.perf_counter()

    for layout in layouts:
        base = load_thumbs(layout, args.entries, px, rng)
        name = f"{layout.parent.name}/{layout.name}"
        print(f"\n=== {name}: {len(base)} entries ===")
        if len(base) < 2:
            continue
        rotated = rotate_hue_all(base, ANGLES)

        per_model = {}
        for enc in encoders:
            e0 = enc.embed(base)
            nn = nn_distance(e0)
            nn_med = float(np.median(nn))
            hue = {}
            for a in ANGLES:
                d = 1.0 - np.sum(e0 * enc.embed(rotated[a]), axis=1)
                hue[str(a)] = {
                    "median": float(np.median(d)),
                    "p10": float(np.percentile(d, 10)),
                    "p90": float(np.percentile(d, 90)),
                    "ratio": float(np.median(d) / max(1e-9, nn_med)),
                    "frac_over_min_sep": float(np.mean(d > MIN_SEPARATION)),
                }
            per_model[enc.key] = {"dim": int(enc.dim), "nn_median": nn_med,
                                  "hue": hue}
            r = hue["74.0"]
            print(f"  {enc.key:12s} dim={enc.dim:4d}  nn_med={nn_med:.4f}  "
                  f"hue74={r['median']:.4f}  ratio={r['ratio']:.2f}  "
                  f"over_bar={100 * r['frac_over_min_sep']:.1f}%")

        results["archives"][name] = {"n": int(len(base)), "models": per_model}

    for enc in encoders:
        results["timing"][enc.key] = {
            "ms_per_image": enc.ms_per_image, "images": enc.images,
            "provider": enc.provider,
            "dim": int(enc.dim) if enc.dim else None,
        }
    results["wall_seconds"] = time.perf_counter() - started

    print("\n=== ms/image ===")
    for enc in encoders:
        print(f"  {enc.key:12s} {enc.ms_per_image:6.2f}")

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}  ({results['wall_seconds'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
