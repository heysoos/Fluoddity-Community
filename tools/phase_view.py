"""Turn a phase-diagram sweep into a picture.

    python -m tools.phase_view phase_out --list
    python -m tools.phase_view phase_out --feature participation_ratio
    python -m tools.phase_view phase_out --rgb change_rate,structure,participation_ratio

The sweep is the expensive part and the picture is free, so the picture is never
baked into the sweep - re-render as often as you like without re-running.

An RGB render is the point rather than a garnish: choosing one good scalar in
advance is the hard part, and three channels show a phase boundary wherever any
of them has an edge.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Perceptually ordered enough to read a landscape off, and defined here rather
# than pulled from matplotlib, which is not a dependency of this project.
_VIRIDIS = np.array([
    (68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
    (38, 130, 142), (31, 158, 137), (53, 183, 121), (109, 205, 89),
    (180, 222, 44), (253, 231, 37),
], dtype=np.float64)


def load(path: Path):
    data = np.load(path / "features.npz", allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    return data, meta


def fill_holes(plane: np.ndarray, done: np.ndarray) -> np.ndarray:
    """Spread completed cells into the ones the sweep has not reached yet.

    A progressive sweep holds a complete coarse lattice at every checkpoint, so
    the honest picture of a partial run is that lattice at its own resolution -
    not a field of isolated dots on grey, which is what plotting the raw array
    would show. Each pass doubles the filled radius, so this converges in
    log2(grid) passes.
    """
    out = np.where(done, plane, np.nan)
    if done.all():
        return out
    while np.isnan(out).any():
        before = np.isnan(out).sum()
        for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            donor = np.roll(out, shift, axis=axis)
            out = np.where(np.isnan(out), donor, out)
        if np.isnan(out).sum() == before:
            break
    return out


def normalise(plane: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
    """-> [0, 1], NaN preserved.

    Percentile clipping rather than min/max: one dead cell at an extreme
    otherwise compresses the whole rest of the diagram into a few levels.
    """
    finite = plane[np.isfinite(plane)]
    if not finite.size:
        return np.zeros_like(plane)
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    if hi <= lo:
        return np.where(np.isfinite(plane), 0.5, np.nan)
    return np.clip((plane - lo) / (hi - lo), 0.0, 1.0)


def colormap(norm: np.ndarray) -> np.ndarray:
    """(H, W) in [0,1] -> (H, W, 3) uint8. NaN renders as mid grey."""
    x = np.nan_to_num(norm, nan=0.0) * (len(_VIRIDIS) - 1)
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, len(_VIRIDIS) - 1)
    t = (x - lo)[..., None]
    rgb = _VIRIDIS[lo] * (1 - t) + _VIRIDIS[hi] * t
    rgb[~np.isfinite(norm)] = 110.0
    return rgb.astype(np.uint8)


def render(data, meta, feature: str | None, rgb: list[str] | None,
           lo_pct: float, hi_pct: float, scale: int) -> tuple[np.ndarray, str]:
    names = [str(n) for n in data["cell_names"]]
    feats = data["cell_features"]
    done = data["done"]

    def plane(name: str) -> np.ndarray:
        if name not in names:
            raise SystemExit(f"no feature {name!r}; try --list")
        # Row 0 is the LOW end of the y axis, so the array is flipped for
        # display: an image's first row is its top.
        return np.flipud(fill_holes(feats[:, :, names.index(name)], done))

    if rgb:
        if len(rgb) != 3:
            raise SystemExit("--rgb needs exactly three comma-separated features")
        img = np.stack([
            (normalise(plane(n), lo_pct, hi_pct) * 255).astype(np.uint8)
            for n in rgb], axis=-1)
        title = "-".join(rgb)
    else:
        img = colormap(normalise(plane(feature), lo_pct, hi_pct))
        title = feature

    if scale > 1:
        img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    return img, title


def summarise(feats: np.ndarray, strip: np.ndarray, names: list[str],
              pr_bracket=None) -> None:
    """Every feature's spread against the noise floor.

    The one home for this table: the sweep prints it on finishing and the
    viewer prints it on --list, and a second copy would drift.
    """
    flat = feats.reshape(-1, len(names))
    print(f"{'feature':<22}{'p05':>10}{'p50':>10}{'p95':>10}"
          f"{'spread':>10}{'noise sd':>10}{'snr':>8}")
    for i, name in enumerate(names):
        col = flat[:, i]
        col = col[np.isfinite(col)]
        if not col.size:
            continue
        lo, mid, hi = np.percentile(col, [5, 50, 95])
        spread = hi - lo
        if len(strip) > 1:
            sd = float(strip[:, i].std())
            snr = spread / sd if sd > 0 else float("inf")
            print(f"{name:<22}{lo:10.4f}{mid:10.4f}{hi:10.4f}"
                  f"{spread:10.4f}{sd:10.4f}{snr:8.1f}")
        else:
            print(f"{name:<22}{lo:10.4f}{mid:10.4f}{hi:10.4f}{spread:10.4f}"
                  f"{'-':>10}{'-':>8}")
    if len(strip) > 1:
        print("\nsnr is the p05..p95 spread over the noise floor's standard "
              "deviation.\nA channel near 1 is speckle, not landscape.")

    # alive_steps is the one feature that can silently measure nothing: a
    # bracket no cell ever leaves reports the budget everywhere.
    if "alive_steps" in names:
        col = flat[:, names.index("alive_steps")]
        col = col[np.isfinite(col)]
        if col.size and col.std() == 0.0:
            pr = flat[:, names.index("participation_ratio")]
            pr = pr[np.isfinite(pr)]
            lo, hi = np.percentile(pr, [1, 99]) if pr.size else (0.0, 1.0)
            print(f"\nalive_steps is flat: no cell left "
                  f"{list(pr_bracket) if pr_bracket else 'the bracket'}. "
                  f"Participation ratio spans {lo:.3f}..{hi:.3f} here,\n"
                  f"so try --pr-lo/--pr-hi inside that to make the feature "
                  f"discriminate.")


def describe(data, meta) -> None:
    names = [str(n) for n in data["cell_names"]]
    print(f"{meta['preset_name']}  {meta['brain_layout']}")
    print(f"  {meta['x_param']} {meta['x_range']} x "
          f"{meta['y_param']} {meta['y_range']}")
    print(f"  grid {meta['grid']}, {meta['steps']} steps, "
          f"world_size {meta['world_size']}, canvas {meta['canvas'][0]}")
    done = data["done"]
    print(f"  {int(done.sum())}/{done.size} cells complete\n")
    summarise(data["cell_features"], data["noise_strip"], names,
              meta.get("pr_bracket"))


def main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="the sweep's output directory")
    ap.add_argument("--feature", default="participation_ratio")
    ap.add_argument("--rgb", default="", help="three features, comma separated")
    ap.add_argument("--list", action="store_true",
                    help="print every feature's spread against the noise floor")
    ap.add_argument("--lo-pct", type=float, default=2.0)
    ap.add_argument("--hi-pct", type=float, default=98.0)
    ap.add_argument("--scale", type=int, default=4, help="nearest-neighbour zoom")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    path = Path(args.path)
    data, meta = load(path)
    if args.list:
        describe(data, meta)
        return 0

    rgb = [s.strip() for s in args.rgb.split(",") if s.strip()]
    img, title = render(data, meta, args.feature, rgb or None,
                        args.lo_pct, args.hi_pct, args.scale)
    dest = Path(args.out) if args.out else path / f"{title}.png"
    Image.fromarray(img).save(dest)
    print(f"wrote {dest}  ({img.shape[1]}x{img.shape[0]})")
    print(f"  x = {meta['x_param']} {meta['x_range']} left to right")
    print(f"  y = {meta['y_param']} {meta['y_range']} bottom to top")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
