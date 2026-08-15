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

import ui  # noqa: F401,E402  prime the services/ui import cycle

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


def recompute_series_columns(data, meta, pr_lo=None, pr_hi=None,
                             rho_floor=None) -> np.ndarray:
    """Re-derive the series-backed columns from the stored probe series.

    `change_rate` and `alive_steps` are functions of `probe_series`, which is
    kept in full - so their definitions can change after a sweep has run
    without re-running it. That is the whole reason the series is stored rather
    than only the scalars it reduces to, and it is what let the mass floor be
    added to `alive_steps` mid-sweep.
    """
    from services import phase_metrics as pm

    feats = np.array(data["cell_features"])
    series = data["probe_series"]
    steps = data["probe_steps"]
    done = data["done"]
    lo, hi = meta.get("pr_bracket", (0.02, 0.98))
    lo = lo if pr_lo is None else pr_lo
    hi = hi if pr_hi is None else pr_hi
    floor = (meta.get("rho_floor", pm.DEFAULT_RHO_FLOOR)
             if rho_floor is None else rho_floor)
    budget = int(meta["steps"])

    ci = list(pm.CELL_NAMES).index("change_rate")
    ai = list(pm.CELL_NAMES).index("alive_steps")
    for iy, ix in zip(*np.nonzero(done)):
        ser = np.asarray(series[iy, ix], dtype=np.float64)
        keep = np.isfinite(ser).all(axis=1)
        if not keep.any():
            continue
        ser, st = ser[keep], np.asarray(steps)[keep]
        ch = ser[:, pm.PROBE_NAMES.index("change")]
        tail = max(1, int(round(len(ch) * 0.25)))
        feats[iy, ix, ci] = ch[-tail:].mean()
        feats[iy, ix, ai] = pm.alive_steps(st, ser, lo, hi, budget, floor)
    return feats


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
            # np.roll WRAPS, and the two ends of an axis are opposite ends of a
            # parameter range - the top row must never be filled from the
            # bottom one. Blank whichever edge the roll brought around.
            edge = 0 if shift > 0 else -1
            if axis == 0:
                donor[edge, :] = np.nan
            else:
                donor[:, edge] = np.nan
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
           lo_pct: float, hi_pct: float, scale: int,
           feats: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    names = [str(n) for n in data["cell_names"]]
    feats = data["cell_features"] if feats is None else feats
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


def bifurcation(data, meta, feature: str, bins: int, scale: int,
                feats: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    """One parameter across, the feature's DISTRIBUTION up, count as brightness.

    The sweep behind this has a collapsed y range, so every row of a column is
    the same parameters run again and differs only by the splat race. A column
    that stays a thin line is reproducible; one that spreads into a band has
    amplified a float-level perturbation into a visible difference, which is
    what makes this the diagram that separates chaos from measurement noise.

    Counts are log-scaled: a band's rare edges are the interesting part and a
    linear scale renders them as black.
    """
    names = [str(n) for n in data["cell_names"]]
    if feature not in names:
        raise SystemExit(f"no feature {feature!r}; try --list")
    feats = data["cell_features"] if feats is None else feats
    col = feats[:, :, names.index(feature)]
    done = data["done"]

    ys = data["y_values"]
    if float(np.ptp(ys)) > 0.0:
        raise SystemExit(
            "--bifurcation wants a sweep whose y range is collapsed, so that "
            f"its rows are repeats rather than a second parameter. This one "
            f"varies {meta['y_param']} over {meta['y_range']}.")

    good = np.isfinite(col) & done
    vals = col[good]
    if not vals.size:
        raise SystemExit(f"{feature} is empty")
    lo, hi = np.percentile(vals, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(vals.min()), float(vals.max()) + 1e-9

    hist = np.zeros((bins, col.shape[1]), dtype=np.float64)
    for ix in range(col.shape[1]):
        v = col[:, ix][good[:, ix]]
        if not v.size:
            continue
        idx = np.clip(((v - lo) / (hi - lo) * bins).astype(int), 0, bins - 1)
        hist[:, ix] = np.bincount(idx, minlength=bins)

    img = colormap(normalise(np.flipud(np.log1p(hist)), 0.0, 100.0))
    if scale > 1:
        img = np.repeat(img, scale, axis=1)
    return img, f"bifurcation-{feature}"


def mark_preset(img: np.ndarray, meta: dict) -> np.ndarray:
    """Draw a crosshair where the preset's OWN parameters sit.

    The diagram exists to explain one preset, and without this it is a landscape
    with no "you are here" - which quadrant the thing you actually run lives in
    is the first question anyone asks of it.
    """
    phys = (meta.get("config") or {}).get("physics") or {}
    xv = phys.get(meta["x_param"].lower())
    yv = phys.get(meta["y_param"].lower())
    if xv is None or yv is None:
        return img
    h, w = img.shape[:2]
    (x0, x1), (y0, y1) = meta["x_range"], meta["y_range"]
    cx = int(round((xv - x0) / (x1 - x0) * (w - 1)))
    cy = int(round((1.0 - (yv - y0) / (y1 - y0)) * (h - 1)))   # y is flipped
    if not (0 <= cx < w and 0 <= cy < h):
        return img

    out = img.copy()
    arm, gap = max(6, h // 40), max(2, h // 160)
    for d in range(gap, arm):
        for py, px in ((cy - d, cx), (cy + d, cx), (cy, cx - d), (cy, cx + d)):
            if 0 <= py < h and 0 <= px < w:
                # White core, black shoulders, so it reads on any colormap.
                out[py, px] = 255
                for oy, ox in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    qy, qx = py + oy, px + ox
                    if 0 <= qy < h and 0 <= qx < w and out[qy, qx].max() != 255:
                        out[qy, qx] = 0
    return out


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


def _binned_r2(key: np.ndarray, v: np.ndarray, nb: int) -> float:
    """Share of v's variance recovered by binning on `key` alone."""
    idx = np.clip(((key - key.min()) / (np.ptp(key) + 1e-12) * nb).astype(int),
                  0, nb - 1)
    cnt = np.bincount(idx, minlength=nb)
    mean = np.bincount(idx, weights=v, minlength=nb) / np.maximum(cnt, 1)
    return float(1.0 - np.var(v - mean[idx]) / np.var(v))


def best_1d(plane: np.ndarray, done: np.ndarray, xs: np.ndarray,
            ys: np.ndarray, nb: int = 0, ndir: int = 180) -> tuple[float, str]:
    """-> (best single coordinate's share of the variance, what that was).

    Two candidate coordinates, both cheap and both physical:

      LINEAR, constant along parallel lines. What a plane looks like when one
      of its axes does nothing, or when the two trade off against each other.
      Searched over direction, on axes rescaled to the unit square so that two
      parameters with different units are still comparable.

      POLAR, constant along rays from the parameter origin. What a plane looks
      like when its axes are really a magnitude and a ratio - scaling both
      together is then some third parameter the sweep never varied, and only
      the ratio was ever swept. Tested at the parameter origin, which is the
      one point where a multiplicative pair degenerates.

    A high score means the sweep spent two axes measuring a curve.
    """
    r, c = plane.shape[:2]
    nb = nb or max(16, min(r, c) // 3)
    m = done & np.isfinite(plane)
    v = plane[m].astype(np.float64)
    # Relative, because a channel in step counts sits at 2500 and one in
    # cosine sits at 0.5 - an absolute floor calls only the second one flat.
    if v.size < 4 * nb or np.std(v) <= 1e-6 * max(1.0, abs(float(np.mean(v)))):
        return float("nan"), "flat"

    gy, gx = np.mgrid[0:r, 0:c]
    x, y = (gx / max(c - 1, 1))[m], (gy / max(r - 1, 1))[m]
    best, label = -np.inf, "?"
    for phi in np.linspace(0.0, np.pi, ndir, endpoint=False):
        score = _binned_r2(x * np.cos(phi) + y * np.sin(phi), v, nb)
        if score > best:
            best, label = score, f"linear @{np.degrees(phi):3.0f} deg"

    # Only meaningful when the rays actually fan out inside the swept box.
    if xs.min() < 0.0 < xs.max() and ys.min() < 0.0 < ys.max():
        XV, YV = np.meshgrid(xs, ys)
        score = _binned_r2(np.arctan2(YV[m], XV[m]), v, nb)
        if score > best:
            best, label = score, "polar"
    return float(best), label


def dimensionality(data, meta, feats=None) -> None:
    """Whether the plane earned its second axis.

    A sweep is expensive and a plane that turns out to be a line is the
    expensive way to measure a curve, so this is worth a coarse pilot before a
    long run rather than a post-mortem after one.
    """
    names = [str(n) for n in data["cell_names"]]
    feats = data["cell_features"] if feats is None else feats
    done, xs, ys = data["done"], data["x_values"], data["y_values"]
    print(f"{meta['x_param']} x {meta['y_param']}   "
          f"(a linear angle of 0 deg lies along {meta['x_param']})")
    print(f"{'feature':<22}{'1-D R2':>8}   {'coordinate':<18}verdict")
    scores = []
    for i, name in enumerate(names):
        r2, label = best_1d(feats[..., i], done, xs, ys)
        if not np.isfinite(r2):
            print(f"{name:<22}{'flat':>8}")
            continue
        scores.append(r2)
        verdict = ("1-D" if r2 >= 0.90 else
                   "mostly 1-D" if r2 >= 0.75 else
                   "2-D" if r2 < 0.60 else "partly 2-D")
        print(f"{name:<22}{r2:>8.3f}   {label:<18}{verdict}")
    if scores:
        med = float(np.median(scores))
        print(f"\nmedian 1-D R2 {med:.3f} - "
              + ("this plane is a LINE; one axis is redundant with the other "
                 "or with a parameter outside the sweep."
                 if med >= 0.85 else
                 "the second axis carries structure of its own."))


def describe(data, meta, feats=None) -> None:
    names = [str(n) for n in data["cell_names"]]
    feats = data["cell_features"] if feats is None else feats
    print(f"{meta['preset_name']}  {meta['brain_layout']}")
    print(f"  {meta['x_param']} {meta['x_range']} x "
          f"{meta['y_param']} {meta['y_range']}")
    print(f"  grid {meta['grid']}, {meta['steps']} steps, "
          f"world_size {meta['world_size']}, canvas {meta['canvas'][0]}")
    done = data["done"]
    print(f"  {int(done.sum())}/{done.size} cells complete\n")
    summarise(feats[done], data["noise_strip"], names, meta.get("pr_bracket"))


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
    ap.add_argument("--mark", action="store_true",
                    help="crosshair at the preset's own parameter values")
    ap.add_argument("--bifurcation", action="store_true",
                    help="for a sweep whose y range is collapsed to repeats: "
                         "--feature's distribution against the x parameter")
    ap.add_argument("--bins", type=int, default=256,
                    help="vertical resolution of --bifurcation")
    ap.add_argument("--ridge", action="store_true",
                    help="whether the plane earned its second axis: the best "
                         "single direction's share of each feature's variance")
    ap.add_argument("--recompute", action="store_true",
                    help="re-derive change_rate and alive_steps from the stored "
                         "probe series, under the thresholds given below")
    ap.add_argument("--pr-lo", type=float, default=None)
    ap.add_argument("--pr-hi", type=float, default=None)
    ap.add_argument("--rho-floor", type=float, default=None)
    args = ap.parse_args(argv)

    path = Path(args.path)
    data, meta = load(path)
    feats = None
    if args.recompute or args.rho_floor is not None:
        feats = recompute_series_columns(data, meta, args.pr_lo, args.pr_hi,
                                         args.rho_floor)
    if args.ridge:
        dimensionality(data, meta, feats)
        return 0
    if args.list:
        describe(data, meta, feats)
        return 0

    rgb = [s.strip() for s in args.rgb.split(",") if s.strip()]
    if args.bifurcation:
        img, title = bifurcation(data, meta, args.feature, args.bins,
                                 args.scale, feats)
    else:
        img, title = render(data, meta, args.feature, rgb or None,
                            args.lo_pct, args.hi_pct, args.scale, feats)
        if args.mark:
            img = mark_preset(img, meta)
    dest = Path(args.out) if args.out else path / f"{title}.png"
    Image.fromarray(img).save(dest)
    print(f"wrote {dest}  ({img.shape[1]}x{img.shape[0]})")
    print(f"  x = {meta['x_param']} {meta['x_range']} left to right")
    if args.bifurcation:
        print(f"  y = {args.feature} value, low to high, brightness = count")
    else:
        print(f"  y = {meta['y_param']} {meta['y_range']} bottom to top")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
