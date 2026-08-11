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
            "|force|", "|strafe|", "random projection")
# Index of the random-projection entry, which the shader handles as a dot with
# PREVIEW_OUT rather than by picking a component.
CHANNEL_RANDOM = len(CHANNELS) - 1
# The 4D input is (L.axial, L.lateral, R.axial, R.lateral) in the particle's own
# frame. Sweeping the two AXIAL taps is the pair that drives steering, so it is
# the default.
#
# `None` is a RANDOM PROJECTION: a plane spanned by two random orthonormal
# directions instead of two coordinate axes. Standard practice for looking at a
# high-dimensional function - it is how neural-net loss landscapes are drawn
# (Li et al. 2018) - and it is the honest default view, because an axis-aligned
# slice only shows the behaviour along one arbitrary frame. A brain's units
# point wherever the search put them, so the coordinate axes have no special
# claim, and structure lying diagonally is invisible in all four of them.
AXES = (
    ("L.axial / R.axial", (0, 2)),
    ("L.axial / L.lateral", (0, 1)),
    ("R.axial / R.lateral", (2, 3)),
    ("L.lateral / R.lateral", (1, 3)),
    ("random projection", None),
)


def basis_for(axes, seed: int = 0):
    """-> (u, v), two ORTHONORMAL directions in the 4D input space.

    Orthonormal, not merely random: a plane spanned by two arbitrary Gaussian
    vectors is skewed and unequally scaled, so the Input Range slider would mean
    a different distance along each one and the picture would be sheared - and
    two nearly-parallel draws would collapse it to almost a line.

    Deterministic in `seed`, because this is recomputed every frame and a plane
    that redrew itself each time would strobe rather than show anything.
    """
    if axes is not None:
        e = np.eye(4, dtype=np.float32)
        return e[axes[0]], e[axes[1]]
    # QR of a 4x2 Gaussian is the standard uniform draw from the Stiefel
    # manifold - every 2-plane equally likely, and both directions unit length.
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    q, r = np.linalg.qr(rng.standard_normal((4, 2)))
    # numpy's QR does not fix the sign of R's diagonal, so without this the
    # plane flips about an axis for some seeds - harmless but confusing.
    q = q * np.sign(np.diag(r))
    return q[:, 0].astype(np.float32), q[:, 1].astype(np.float32)


def output_direction(seed: int = 0) -> np.ndarray:
    """-> a random UNIT direction in the 4D output space.

    Unit length so the Contrast slider means the same thing here as it does for
    a single channel. Offset from the input plane's seed so that one Reseed
    press moves both without the two being locked to the same draw.
    """
    rng = np.random.default_rng((int(seed) ^ 0x9E3779B9) & 0xFFFFFFFF)
    v = rng.standard_normal(4)
    return (v / np.linalg.norm(v)).astype(np.float32)


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
               value_range: float = 2.0, gain: float = 1.0, seed: int = 0,
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
        tryset(p, "PREVIEW_OUT", tuple(float(c) for c in output_direction(seed)))
        tryset(p, "BRAIN_MODALITY", get(layout.modality).modality_id)
        tryset(p, "BRAIN_LEN", int(layout.length))
        shape = (tuple(layout.shape) + (0, 0, 0, 0))[:4]
        tryset(p, "BRAIN_SHAPE", tuple(int(v) for v in shape))
        u, v = basis_for(axes, seed)
        tryset(p, "PREVIEW_U", tuple(float(c) for c in u))
        tryset(p, "PREVIEW_V", tuple(float(c) for c in v))
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
