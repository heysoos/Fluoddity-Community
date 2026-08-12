"""How much of the archive's novelty signal is spent on hue.

Rotates each sampled thumbnail's HSV hue and measures how far that moves its
embedding, against the distance to a genuinely different entry. Reports one
scale-free ratio per encoder, plus wall-clock cost per image, so the hue
question and the "is a better encoder affordable" question are answered off one
pass.

The ratio is scale-free on purpose: a separation bar is calibrated in one
encoder's space and means nothing in another's. A ratio at or above 1 means a
pure hue rotation looks at least as different as a new creature. The figures
this produced live in the colour caveat in CLAUDE.md.

    python -m tools.hue_nuisance --entries 700

Vision towers only - this never loads a text encoder. Each is expected at
`models/<subdir>/vision_model_fp16.onnx`; missing ones are skipped, so the
default encoder alone is a valid run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# The app has a pre-existing import cycle: services/__init__ -> config_saver ->
# ui -> services.config_saver. It resolves only when `ui` is imported first.
# Prime it, as tests/conftest.py and tools/brain_search_bench.py do.
import ui  # noqa: F401,E402

from services.vision_models import DEFAULT_KEY, REGISTRY, get  # noqa: E402
from tools.archive_sample import (
    build_encoders,
    load_thumbs,
    nn_distance,
    recent_layouts,
)
from utilities.paths import get_archives_root

# 74 deg is |dhue| at expedition_sigma with the gain at 0.5 - the angle the
# search actually produces, and the one the caveat quotes.
ANGLES = (15.0, 30.0, 60.0, 74.0, 90.0, 180.0)


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


# ---- main ---------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entries", type=int, default=700, help="per archive")
    ap.add_argument("--archives", type=int, default=3, help="most recent N")
    ap.add_argument("--only", default="", help="comma-separated model keys")
    ap.add_argument("--out", default="hue_nuisance.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    wanted = [k.strip() for k in args.only.split(",") if k.strip()]
    encoders = build_encoders(wanted or sorted(REGISTRY))
    if not encoders:
        raise SystemExit("no models available")
    px = encoders[0].px
    if any(e.px != px for e in encoders):
        raise SystemExit("models disagree on input size")

    # The admission bar the percentages are reported against. Valid for the
    # encoder that owns it; every other model needs its own.
    bar = get(DEFAULT_KEY).default_min_separation

    layouts = recent_layouts(get_archives_root(), args.archives)
    if not layouts:
        raise SystemExit("no archive holds any thumbnails")

    rng = np.random.default_rng(args.seed)
    results = {"angles": list(ANGLES), "entries_per_archive": args.entries,
               "separation_bar": bar, "archives": {}, "timing": {}}
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
                    "frac_over_bar": float(np.mean(d > bar)),
                }
            per_model[enc.key] = {"dim": int(enc.dim), "nn_median": nn_med,
                                  "hue": hue}
            r = hue["74.0"]
            print(f"  {enc.key:12s} dim={enc.dim:4d}  nn_med={nn_med:.4f}  "
                  f"hue74={r['median']:.4f}  ratio={r['ratio']:.2f}  "
                  f"over_bar={100 * r['frac_over_bar']:.1f}%")

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
