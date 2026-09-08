"""Where a tournament tile's seams are, in texels.

The single Python copy of `tile_lo_texel()`, which appears verbatim in
entity_update.glsl, canvas.frag and brush.vert. Anything on the CPU that needs
to know where a tile starts - the capture crop, tests - uses this, so there is
one definition of a seam rather than four.

A tile owns a WHOLE NUMBER of texels rather than an equal share of the world,
since no canvas size divides evenly for every grid setting - an evenly divided
seam would run through the middle of a texel. See CLAUDE.md.

Integer arithmetic, because a seam is decided by the last bit: e.g. 647*4/8 is
exactly 323.5, and the GPU's float form of this can round to the wrong tile.
"""
from __future__ import annotations


def lo_texel(k: int, grid: int, res: int) -> int:
    """First texel of tile `k` along one axis: the smallest t whose centre
    (t + 0.5) is at or past the seam k/grid."""
    k, grid, res = int(k), int(grid), int(res)
    if k <= 0:
        return 0
    if k >= grid:
        return res
    b = 2 * grid
    return -(-(2 * k * res - grid) // b)          # ceil division, exact


def tile_uv_box(tx: int, ty: int, grid: int, res: tuple[int, int]):
    """-> ((u0, v0), (u1, v1)) for tile (tx, ty), on texel edges.

    v increases upward, matching the GL texture coordinate: tile 0 is
    bottom-left, as index_home_tile() has it.
    """
    w, h = int(res[0]), int(res[1])
    lo = (lo_texel(tx, grid, w) / w, lo_texel(ty, grid, h) / h)
    hi = (lo_texel(tx + 1, grid, w) / w, lo_texel(ty + 1, grid, h) / h)
    return lo, hi
