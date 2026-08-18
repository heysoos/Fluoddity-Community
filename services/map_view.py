"""What the archive map draws: which entries, what colour, and density bins.

Pure over plain entry data, so the rules are testable without an ImGui context.

The map is a 2-D PCA of the 512-d descriptors. What makes a large archive
unreadable is CROWDING, not the projection, which is why filtering exists here.
Never read local density off dot overlap - `novelty` is the real measure and it
is computed in the full space. See CLAUDE.md.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np


class MapPoints(NamedTuple):
    """One frame's worth of map geometry, after filtering."""
    unit: np.ndarray            # (n, 2) positions in [0, 1] over the FILTERED set
    lo: np.ndarray              # projection-space offset, for the goal marker
    span: np.ndarray
    colors: np.ndarray          # (n,) packed RGBA
    idx: np.ndarray             # (n,) archive indices - row i is NOT entry i
    tvals: np.ndarray | None    # (n,) the normalised scalar being coloured
    novelty: np.ndarray         # (n,) float32, for the atlas's cell winner

# UI-facing option lists. The first of each is the historical behaviour and
# stays the default: these are additions to the map, not a replacement for it.
COLOR_MODES = ("source", "novelty", "liveness")
FILTER_MODES = ("all", "recent", "novel", "kept", "goal", "source")
RENDER_MODES = ("points", "density", "points+density")

FILTER_LABELS = {
    "all": "All entries",
    "recent": "Recent generations",
    "novel": "Most novel",
    "kept": "Kept for matching",
    "goal": "One goal",
    "source": "One regime",
}


def pack(r: int, g: int, b: int, a: int = 255) -> int:
    """ImGui's IM_COL32 packing, without importing ImGui."""
    return (int(a) << 24) | (int(b) << 16) | (int(g) << 8) | int(r)


def filter_indices(entries, mode: str, *, recent_gens: int = 200,
                   novel_pct: int = 25, goal: str = "",
                   source: str = "") -> np.ndarray:
    """-> indices of the entries to draw, ascending.

    Returns ALL indices for an unknown mode rather than none: a stale
    persisted setting must not present as an empty map.
    """
    n = len(entries)
    everything = np.arange(n, dtype=np.int64)
    if n == 0 or mode not in FILTER_MODES or mode == "all":
        return everything

    if mode == "recent":
        gens = np.array([e.gen for e in entries], dtype=np.int64)
        return np.flatnonzero(gens >= gens.max() - max(0, int(recent_gens)))

    if mode == "novel":
        nov = np.array([e.novelty for e in entries], dtype=np.float32)
        keep = max(1, int(round(n * min(100, max(1, int(novel_pct))) / 100.0)))
        # Sorted so the result stays in archive order, which keeps the index
        # mapping monotonic and the hover lookup honest.
        return np.sort(np.argsort(nov)[::-1][:keep]).astype(np.int64)

    if mode == "kept":
        # The results: what an expedition converged on, what beat a goal
        # record, and what the user picked by hand.
        return np.array([i for i, e in enumerate(entries)
                         if e.pinned or e.source in ("summit", "record")],
                        dtype=np.int64)

    if mode == "goal":
        return np.array([i for i, e in enumerate(entries) if e.goal == goal],
                        dtype=np.int64)

    return np.array([i for i, e in enumerate(entries) if e.source == source],
                    dtype=np.int64)


def present_values(entries, field: str) -> list[str]:
    """The distinct non-empty values of `field`, for a combo to offer."""
    return sorted({getattr(e, field, "") for e in entries} - {""})


def scalar_values(entries, mode: str) -> np.ndarray | None:
    """The per-entry number a colour ramp should encode, or None for `source`."""
    if mode == "novelty":
        return np.array([e.novelty for e in entries], dtype=np.float32)
    if mode == "liveness":
        return np.array([e.liveness for e in entries], dtype=np.float32)
    return None


def normalise(values: np.ndarray, lo_pct: float = 2.0,
              hi_pct: float = 98.0) -> np.ndarray:
    """-> (n,) in [0, 1], on ROBUST bounds.

    Percentiles rather than min/max because one outlier - a generation-0 entry
    still stamped with the no-reference novelty of 1.0 - would otherwise flatten
    every real entry into the bottom of the ramp.
    """
    v = np.asarray(values, dtype=np.float32)
    if not len(v):
        return v
    lo, hi = np.percentile(v, [lo_pct, hi_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.full(len(v), 0.5, dtype=np.float32)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


# Dark blue -> cyan -> green -> yellow -> white. Monotone in lightness so it
# still reads as an ordering in greyscale, and it starts well above the
# canvas's (20, 20, 24) background so the low end is visible rather than lost.
_RAMP = np.array([
    (40, 50, 120), (30, 130, 180), (60, 190, 140),
    (220, 210, 90), (255, 255, 235),
], dtype=np.float32)


def ramp_colors(t: np.ndarray, alpha=230) -> np.ndarray:
    """(n,) in [0, 1] -> (n,) int64 of packed RGBA, vectorised.

    `alpha` may be a scalar or a per-entry array.
    """
    t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)
    if not len(t):
        return np.zeros(0, dtype=np.int64)
    pos = t * (len(_RAMP) - 1)
    i = np.clip(pos.astype(np.int32), 0, len(_RAMP) - 2)
    f = (pos - i)[:, None]
    rgb = _RAMP[i] * (1.0 - f) + _RAMP[i + 1] * f
    c = rgb.astype(np.int64)
    a = np.asarray(alpha, dtype=np.int64) & 255
    return ((a << 24) | (c[:, 2] << 16) | (c[:, 1] << 8) | c[:, 0])


def with_alpha(colors: np.ndarray, alpha) -> np.ndarray:
    """Replace the alpha byte of packed colours; `alpha` scalar or per-entry."""
    c = np.asarray(colors, dtype=np.int64) & 0x00FFFFFF
    a = np.asarray(alpha, dtype=np.int64) & 255
    return c | (a << 24)


def bin_points(xs: np.ndarray, ys: np.ndarray, origin, size,
               cell_px: float = 9.0):
    """Assign screen positions to grid cells. -> (flat cell index of each point
    that lands on the canvas, mask of those points, nx, ny, cell).

    In SCREEN space, not projection space, so the resolution follows the zoom:
    a fixed grid in unit space turns back into one blob as soon as you zoom in.
    """
    ox, oy = float(origin[0]), float(origin[1])
    w, h = float(size[0]), float(size[1])
    cell = max(2.0, float(cell_px))
    nx = max(1, int(w // cell))
    ny = max(1, int(h // cell))
    if not len(xs):
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool),
                nx, ny, cell)

    gx = np.floor((np.asarray(xs, dtype=np.float32) - ox) / cell).astype(np.int64)
    gy = np.floor((np.asarray(ys, dtype=np.float32) - oy) / cell).astype(np.int64)
    on = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
    return gy[on] * nx + gx[on], on, nx, ny, cell


def density_grid(xs: np.ndarray, ys: np.ndarray, origin, size,
                 cell_px: float = 9.0):
    """Bin screen positions and count them. -> (counts (ny, nx), nx, ny, cell)."""
    flat, _on, nx, ny, cell = bin_points(xs, ys, origin, size, cell_px)
    counts = np.bincount(flat, minlength=nx * ny).astype(np.int32)
    return counts.reshape(ny, nx), nx, ny, cell


def cell_means(flat: np.ndarray, values: np.ndarray, ncells: int):
    """Mean of `values` per cell. -> (means, counts); means is 0 where empty."""
    counts = np.bincount(flat, minlength=ncells)
    totals = np.bincount(flat, weights=np.asarray(values, dtype=np.float64),
                         minlength=ncells)
    means = np.zeros(ncells, dtype=np.float32)
    hit = counts > 0
    means[hit] = (totals[hit] / counts[hit]).astype(np.float32)
    return means, counts.astype(np.int32)


def cell_majority(flat: np.ndarray, codes: np.ndarray, ncells: int):
    """Most common code per cell. -> (codes, counts); code is 0 where empty.

    For colouring by a category - the winner is the regime that owns the cell,
    not a blend, because averaging two packed colours is not a colour.
    """
    counts = np.bincount(flat, minlength=ncells)
    best = np.zeros(ncells, dtype=np.int64)
    best_n = np.zeros(ncells, dtype=np.int64)
    codes = np.asarray(codes, dtype=np.int64)
    for code in np.unique(codes):
        n = np.bincount(flat[codes == code], minlength=ncells)
        take = n > best_n
        best[take] = code
        best_n[take] = n[take]
    return best, counts.astype(np.int32)


def density_intensity(counts: np.ndarray) -> np.ndarray:
    """Counts -> [0, 1], on a LOG scale.

    Archive density spans orders of magnitude, and a linear ramp renders the
    whole sparse frontier as empty.
    """
    c = np.asarray(counts, dtype=np.float32)
    top = float(c.max()) if c.size else 0.0
    if top <= 0.0:
        return np.zeros_like(c)
    return np.log1p(c) / np.log1p(top)


# A one-entry cell must stay visible, so the floor is well above transparent.
DENSITY_ALPHA_MIN = 90
DENSITY_ALPHA_MAX = 245


def density_alpha(counts: np.ndarray) -> np.ndarray:
    """Counts -> alpha bytes. This is how a heatmap shows COUNT once colour is
    carrying something else."""
    t = density_intensity(counts)
    a = DENSITY_ALPHA_MIN + (DENSITY_ALPHA_MAX - DENSITY_ALPHA_MIN) * t
    return a.astype(np.int64)


def cell_argmax(flat: np.ndarray, values: np.ndarray, ncells: int):
    """Index of the highest-`values` point per cell. -> (winners, counts).

    A winner of -1 marks an empty cell, because 0 is a real row. This is what
    picks the one entry whose thumbnail stands for a cell of the atlas, so the
    picture count follows the VIEWPORT and never the archive.
    """
    counts = np.bincount(flat, minlength=ncells).astype(np.int32)
    win = np.full(ncells, -1, dtype=np.int64)
    if len(flat):
        # Descending, so the LAST write per cell is that cell's maximum.
        order = np.argsort(values, kind="stable")
        win[flat[order]] = order
    return win, counts


def covered_by(flat: np.ndarray, on: np.ndarray, drawn) -> np.ndarray:
    """-> mask over the ORIGINAL points, True where a thumbnail covers them.

    Widened back through `on`, because bin_points drops the off-canvas points
    and the dots are drawn from the full array. Without this the scatter draws
    underneath every picture in the atlas.
    """
    out = np.zeros(len(on), dtype=bool)
    if not len(flat) or not drawn:
        return out
    hit = np.isin(flat, np.fromiter(drawn, dtype=np.int64, count=len(drawn)))
    out[np.flatnonzero(on)] = hit
    return out


def quantised_cell(span_px: float, cell_px: float) -> float:
    """Unit-space cell size whose on-screen size is in [cell_px, 2*cell_px).

    A POWER OF TWO, so the assignment is invariant under pan and under zoom
    within a level. Binning in SCREEN space instead churns: a one-pixel pan
    moves every boundary, so each cell's winner changes and the atlas
    reshuffles under the pointer.
    """
    if span_px <= 0.0 or cell_px <= 0.0:
        return 1.0
    raw = float(cell_px) / float(span_px)
    if raw >= 1.0:
        return 1.0
    level = int(np.floor(np.log2(1.0 / raw)))
    return float(2.0 ** -level)


def atlas_cell(span_x_px: float, span_y_px: float, cell_px: float):
    """-> the unit-space cell (x, y) the atlas bins into.

    ONE level for both axes, so a zoom octave reshuffles the atlas once
    rather than twice, and the cell is SQUARE ON SCREEN rather than in unit
    space - a cell quantised separately per axis is drawn as a rectangle,
    which stretches the thumbnail inside it.
    """
    if span_x_px <= 0.0 or cell_px <= 0.0:
        return 1.0, 1.0
    # The ZOOM is quantised, not the cell. Quantising the cell snaps the size
    # slider to a handful of values, and it is the zoom - not the slider -
    # that must not disturb the binning: moving the slider is a deliberate
    # act and may rebin, where a pan or a zoom may not.
    level = float(2.0 ** np.round(np.log2(float(span_x_px))))
    cx = min(1.0, float(cell_px) / level)
    if span_y_px <= 0.0:
        return cx, cx
    # ROUNDED: the caller scales both spans by the zoom, so the ratio is
    # constant to within a bit or two - and an unrounded ratio is a fresh
    # cache key every frame.
    aspect = round(float(span_x_px) / float(span_y_px), 6)
    return cx, cx * aspect


def atlas_winners(unit: np.ndarray, values: np.ndarray, cell_u,
                  budget: int | None = None):
    """-> (cell origin x, cell origin y, winning row) per occupied cell.

    In UNIT space, so panning cannot change any of it and the result can be
    cached until the zoom LEVEL moves.

    `budget` caps how many cells are returned, keeping the highest-`values`
    ones: a map asking for more pictures than the cache can hold evicts its
    own cells and re-decodes them for as long as it is on screen.
    """
    n = len(unit)
    if not n:
        z = np.zeros(0, dtype=np.float32)
        return z, z, np.zeros(0, dtype=np.int64)
    cx, cy = float(cell_u[0]), float(cell_u[1])
    gx = np.floor(np.asarray(unit[:, 0], dtype=np.float64) / cx).astype(np.int64)
    gy = np.floor(np.asarray(unit[:, 1], dtype=np.float64) / cy).astype(np.int64)
    # unit is in [0, 1], so both are non-negative and this packs without a
    # negative-modulo trap.
    key = gy * (np.int64(1) << np.int64(32)) + gx
    uniq, inv = np.unique(key, return_inverse=True)
    win, _counts = cell_argmax(inv.astype(np.int64),
                               np.asarray(values, dtype=np.float32), len(uniq))
    keep = win >= 0
    win = win[keep]
    if budget is not None and len(win) > int(budget):
        top = np.argpartition(np.asarray(values, dtype=np.float32)[win],
                              -int(budget))[-int(budget):]
        win = win[np.sort(top)]
    return (gx[win].astype(np.float32) * cx,
            gy[win].astype(np.float32) * cy,
            win)
