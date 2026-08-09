"""Where a tournament tile's seams are, in texels.

The single Python copy of `tile_lo_texel()`, which appears verbatim in
entity_update.glsl, canvas.frag and brush.vert. Anything on the CPU that needs
to know where a tile starts - the capture crop, tests - uses this, so there is
one definition of a seam rather than four.

A tile owns a WHOLE NUMBER of texels rather than an equal share of the world.
The canvas is 647 texels across at the default world_size of 0.40 and the grid
slider goes 2..8, so no canvas size makes every setting divide; an evenly
divided seam runs through the middle of a texel. Measured 2026-08-09 at grid 8,
the middle column of the diffusion retained 25.7% of its own trail and 39 of 64
tiles lit a tile they could not legally reach.

Integer arithmetic, because a seam is decided by the last bit: 647*4/8 is
exactly 323.5, and the GPU evaluated the float form of this as tile 3 where the
true answer is 4.
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
    bottom-left, as tournament_home_tile() has it.
    """
    w, h = int(res[0]), int(res[1])
    lo = (lo_texel(tx, grid, w) / w, lo_texel(ty, grid, h) / h)
    hi = (lo_texel(tx + 1, grid, w) / w, lo_texel(ty + 1, grid, h) / h)
    return lo, hi
