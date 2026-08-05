"""Offscreen capture of the tournament grid as exact CLIP-sized crops.

The framebuffer is grid*224 square regardless of grid size, so CLIP always
receives exact 224x224 crops with no resampling, padding or cropping. This is
only correct because Auto mode forces a 1:1 canvas - tiles inherit the canvas
aspect ratio, and a 16:9 tile could not be squared without loss.
"""
from __future__ import annotations

import numpy as np

TILE_PX = 224


def crop_bounds(tile: int, grid: int, tile_px: int = TILE_PX):
    """(row0, row1, col0, col1) into a TOP-DOWN image.

    tournament_home_tile() numbers tile 0 as bottom-left, so ty is measured from
    the bottom and the row is inverted here. Keeping the inversion in exactly
    one place is what stops it being applied twice or cancelled out.
    """
    tx = tile % grid
    ty = tile // grid
    row0 = (grid - 1 - ty) * tile_px
    col0 = tx * tile_px
    return row0, row0 + tile_px, col0, col0 + tile_px


def split_grid(img: np.ndarray, grid: int, tile_px: int = TILE_PX) -> np.ndarray:
    """Top-down (G*px, G*px, 3) -> (G*G, px, px, 3) in shader tile order."""
    n = grid * grid
    out = np.empty((n, tile_px, tile_px, 3), dtype=np.uint8)
    for tile in range(n):
        r0, r1, c0, c1 = crop_bounds(tile, grid, tile_px)
        out[tile] = img[r0:r1, c0:c1]
    return out


class TileCapture:
    """Owns the offscreen framebuffer. The only GL-dependent unit."""

    def __init__(self, ctx, grid: int, tile_px: int = TILE_PX):
        self._ctx = ctx
        self._tile_px = tile_px
        self._tex = None
        self._fbo = None
        self.grid = 0
        self.resize(grid)

    @property
    def size(self) -> int:
        return self.grid * self._tile_px

    @property
    def fbo(self):
        return self._fbo

    def resize(self, grid: int) -> None:
        if grid == self.grid and self._fbo is not None:
            return
        self.release()
        self.grid = grid
        size = grid * self._tile_px
        # dtype='f1' (GL_RGBA8), NOT 'f4'. frame_assembly.frag can emit values
        # above 1.0 after exposure; a float attachment would store those
        # unclamped, so the readback would contain values the display never
        # shows. An 8-bit unsigned-normalized attachment makes the GPU clamp
        # exactly as it does for the visible framebuffer.
        self._tex = self._ctx.texture((size, size), 4, dtype="f1")
        self._fbo = self._ctx.framebuffer(color_attachments=[self._tex])

    def capture(self, render_fn) -> np.ndarray:
        """render_fn(fbo) draws the assembled grid. Returns uint8
        (grid*grid, tile_px, tile_px, 3)."""
        size = self.size
        self._fbo.use()
        self._ctx.viewport = (0, 0, size, size)
        self._ctx.clear(0.0, 0.0, 0.0, 1.0)
        if render_fn is not None:
            render_fn(self._fbo)
        # alignment=1 is mandatory. The default GL_PACK_ALIGNMENT of 4 pads each
        # row to a 4-byte boundary; it happens to be a no-op at these widths and
        # would silently corrupt the image if tile_px or the channel count
        # changed.
        buf = self._fbo.read(components=3, alignment=1)
        expected = size * size * 3
        if len(buf) != expected:
            raise RuntimeError(f"readback was {len(buf)} bytes, expected {expected}")
        # fbo.read() returns rows bottom-up; flip once here to top-down.
        img = np.frombuffer(buf, dtype=np.uint8).reshape(size, size, 3)[::-1]
        return split_grid(np.ascontiguousarray(img), self.grid, self._tile_px)

    def release(self) -> None:
        for obj in (self._fbo, self._tex):
            if obj is not None:
                obj.release()
        self._fbo = None
        self._tex = None
