"""What the archive map draws: which entries, what colour, and density bins.

Split out of the window so the rules are testable without an ImGui context.
Everything here is pure over plain entry data.

The map is a 2-D PCA of 512-d descriptors, and MEASURED on the real archives it
holds 46% of the variance at every size - the projection does not degrade as the
archive grows. What degrades is CROWDING: kNN(10) preservation falls 27.0% at
500 entries to 3.1% at 13049, while the true distance between an entry and the
ten the map puts nearest it stays flat at ~0.038. That is nearly double
`min_separation` (0.02), so at full size two touching dots are on average
further apart than the distance at which the archive calls two entries
different. Hence: filtering is not cosmetic, it is what makes adjacency mean
something again, and DENSITY MUST NOT BE READ OFF DOT OVERLAP - `novelty` is
the real local-density measure and it is computed in the full 512-d space.
"""
from __future__ import annotations

import numpy as np

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


def ramp_colors(t: np.ndarray, alpha: int = 230) -> np.ndarray:
    """(n,) in [0, 1] -> (n,) int64 of packed RGBA, vectorised."""
    t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)
    if not len(t):
        return np.zeros(0, dtype=np.int64)
    pos = t * (len(_RAMP) - 1)
    i = np.clip(pos.astype(np.int32), 0, len(_RAMP) - 2)
    f = (pos - i)[:, None]
    rgb = _RAMP[i] * (1.0 - f) + _RAMP[i + 1] * f
    c = rgb.astype(np.int64)
    return ((int(alpha) << 24) | (c[:, 2] << 16) | (c[:, 1] << 8) | c[:, 0])


def density_grid(xs: np.ndarray, ys: np.ndarray, origin, size,
                 cell_px: float = 9.0):
    """Bin screen positions into a grid. -> (counts (ny, nx), nx, ny, cell).

    In SCREEN space, not projection space, so the resolution follows the zoom:
    binning once in unit space would give a fixed grid that turns back into a
    single blob the moment you zoom in, which is the problem this is for.
    """
    ox, oy = float(origin[0]), float(origin[1])
    w, h = float(size[0]), float(size[1])
    cell = max(2.0, float(cell_px))
    nx = max(1, int(w // cell))
    ny = max(1, int(h // cell))
    if not len(xs):
        return np.zeros((ny, nx), dtype=np.int32), nx, ny, cell

    gx = np.floor((np.asarray(xs, dtype=np.float32) - ox) / cell).astype(np.int64)
    gy = np.floor((np.asarray(ys, dtype=np.float32) - oy) / cell).astype(np.int64)
    on = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
    counts = np.zeros(ny * nx, dtype=np.int32)
    if on.any():
        flat = gy[on] * nx + gx[on]
        np.add.at(counts, flat, 1)
    return counts.reshape(ny, nx), nx, ny, cell


def density_intensity(counts: np.ndarray) -> np.ndarray:
    """Counts -> [0, 1], on a LOG scale.

    Archive density spans orders of magnitude - the middle of the cone holds
    hundreds of entries per cell while the frontier holds one - and a linear
    ramp renders the whole frontier as empty, which is the half worth seeing.
    """
    c = np.asarray(counts, dtype=np.float32)
    top = float(c.max()) if c.size else 0.0
    if top <= 0.0:
        return np.zeros_like(c)
    return np.log1p(c) / np.log1p(top)
