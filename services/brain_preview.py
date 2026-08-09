"""Renders the Brain Inspector atlas: one tile per unit, plus the whole brain.

Assembles the SAME brain GLSL the compute shader uses and binds the SAME
parameter buffer, so a tile shows what the particles compute rather than a
second implementation that can drift. That is only possible because the
brain_*() functions are pure - they read brain_params, base, x and BRAIN_SHAPE
and nothing else.
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

from utilities.gl_helpers import read_shader, shader_prepend, tryset

# Order matters and is the same as sim.py's: shader_prepend inserts after
# #version, so the LAST prepend lands FIRST in the compiled source.
_BRAINS = ("mlp", "lenia", "gabor", "fourier")

CHANNELS = ("force axial", "force lateral", "strafe axial", "strafe lateral",
            "|force|", "|strafe|")
# The 4D input is (L.axial, L.lateral, R.axial, R.lateral) in the particle's own
# frame. Sweeping the two AXIAL taps is the pair that drives steering, so it is
# the default.
AXES = (
    ("L.axial / R.axial", (0, 2)),
    ("L.axial / L.lateral", (0, 1)),
    ("R.axial / R.lateral", (2, 3)),
    ("L.lateral / R.lateral", (1, 3)),
)


class BrainPreview:
    def __init__(self, ctx: moderngl.Context, tile: int = 96):
        self.ctx = ctx
        self.tile = int(tile)
        self._tex: moderngl.Texture | None = None
        self._fbo: moderngl.Framebuffer | None = None
        self._grid = 0

        # Same order sim.py uses. shader_prepend inserts after #version, so the
        # LAST prepend lands FIRST: dispatch needs the modalities declared, the
        # modalities need the header, the header needs hash() from fourier4_4.
        src = read_shader("shaders/brain_preview.frag")
        src = shader_prepend(src, read_shader("shaders/brains/_dispatch.glsl"))
        for name in _BRAINS:
            src = shader_prepend(src, read_shader(f"shaders/brains/{name}.glsl"))
        src = shader_prepend(src, read_shader("shaders/brains/_header.glsl"))
        src = shader_prepend(src, read_shader("shaders/fourier4_4.glsl"))

        self.program = ctx.program(
            vertex_shader=read_shader("shaders/canvas.vert"),
            fragment_shader=src,
        )
        self.vao = ctx.vertex_array(self.program, [])

    # ---- geometry ------------------------------------------------------

    @staticmethod
    def unit_count(layout) -> int:
        """Units in this brain: centres, filters, bumps or hidden units."""
        return int(layout.shape[0]) if layout.shape else 0

    def _ensure_target(self, tiles: int) -> None:
        grid = max(1, math.ceil(math.sqrt(tiles)))
        if self._tex is not None and grid == self._grid:
            return
        if self._fbo is not None:
            self._fbo.release()
        if self._tex is not None:
            self._tex.release()
        side = grid * self.tile
        self._tex = self.ctx.texture((side, side), 4, dtype="f1")
        self._tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self._fbo = self.ctx.framebuffer(color_attachments=[self._tex])
        self._grid = grid

    # ---- drawing -------------------------------------------------------

    def render(self, layout, brain_buffer, *, axes=(0, 2), channel: int = 0,
               value_range: float = 2.0, gain: float = 1.0,
               include_total: bool = True) -> moderngl.Texture:
        """-> the atlas texture. Tile 0 is the whole brain when include_total.

        Re-rendered on demand rather than cached: at 96 px a full 48-unit brain
        is ~450k fragments of pure arithmetic, far below one simulation step,
        and caching would need invalidating on every parameter write.
        """
        from services.brains import get

        n = self.unit_count(layout)
        tiles = n + (1 if include_total else 0)
        self._ensure_target(tiles)
        assert self._fbo is not None and self._tex is not None

        brain_buffer.bind_to_storage_buffer(4)
        p = self.program
        tryset(p, "BRAIN_MODALITY", get(layout.modality).modality_id)
        tryset(p, "BRAIN_LEN", int(layout.length))
        shape = (tuple(layout.shape) + (0, 0, 0, 0))[:4]
        tryset(p, "BRAIN_SHAPE", tuple(int(v) for v in shape))
        tryset(p, "PREVIEW_AXES", (int(axes[0]), int(axes[1])))
        tryset(p, "PREVIEW_CHANNEL", int(channel))
        tryset(p, "PREVIEW_RANGE", float(value_range))
        tryset(p, "PREVIEW_GAIN", float(gain))

        self._fbo.use()
        self._fbo.clear(0.05, 0.05, 0.06, 1.0)
        units = ([-1] if include_total else []) + list(range(n))
        for slot, unit in enumerate(units):
            gx = slot % self._grid
            gy = slot // self._grid
            # One tile per draw, because PREVIEW_UNIT is a uniform. n is at most
            # 48, so this is dozens of tiny draws, not thousands.
            self._fbo.viewport = (gx * self.tile, gy * self.tile,
                                  self.tile, self.tile)
            tryset(p, "PREVIEW_UNIT", int(unit))
            self.vao.render(moderngl.TRIANGLE_FAN, vertices=4)
        self._fbo.viewport = (0, 0, self._tex.width, self._tex.height)
        # Hand the default framebuffer back, so a caller that draws next does
        # not land in the atlas. None in a standalone context, which is what the
        # GPU tests run under.
        screen = getattr(self.ctx, "screen", None)
        if screen is not None:
            screen.use()
        return self._tex

    def uv_for(self, slot: int) -> tuple[tuple[float, float], tuple[float, float]]:
        """(uv0, uv1) of one tile in the atlas, for imgui.image_button.

        v is flipped: the framebuffer's origin is bottom-left and ImGui's is
        top-left, so handing it the raw range draws every tile upside down.
        """
        g = max(1, self._grid)
        gx, gy = slot % g, slot // g
        u0, u1 = gx / g, (gx + 1) / g
        v0, v1 = (gy + 1) / g, gy / g
        return (u0, v0), (u1, v1)

    @property
    def grid(self) -> int:
        return self._grid

    def release(self) -> None:
        for obj in (self._fbo, self._tex, self.vao, self.program):
            try:
                if obj is not None:
                    obj.release()
            except Exception:
                pass
        self._fbo = self._tex = None
