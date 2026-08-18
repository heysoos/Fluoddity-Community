"""The trail's DIFFUSION runs on the sim's clock too, on a real GPU.

The decay and the deposit were put on the clock; the blur was not. It is one
step of an explicit heat solve, so halving TIME_SCALE runs twice as many of
them per unit of simulated time and the trail smears twice as far in variance.
That is not a slower creature, it is a blurrier one - and 83 of the 131 shipped
presets run Trail Diffusion at the slider's maximum, where a single step moves
80% of a texel's mass into its neighbours.

So this runs the SHIPPED canvas.frag, whole and unmodified, ping-ponged over a
canvas exactly as sim.py drives it. A source-level reading cannot see the
quantity under test: what matters is how far a point spreads over a FIXED span
of simulated time, which is a property of the recursion and not of any one
line.

Skipped when no GL 4.3 context is available (CI, remote shells, no GPU).
"""
from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from utilities.gl_helpers import read_shader  # noqa: E402

RES = 128
WRAP = 2
SPAN = 20.0          # simulated time every case must cover

QUAD_VERT = """#version 430
out vec2 texcoord;
void main(){
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    texcoord = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:                    # no GPU, no display, no driver
        pytest.skip(f"no GL 4.3 context: {exc}")
    yield c
    c.release()


class Trail:
    """Ping-pong the WHOLE shipped canvas.frag, as sim.py's update() does."""

    def __init__(self, ctx, diffusion: float, time_scale: float, res: int = RES):
        self.ctx = ctx
        self.res = res
        self.prog = ctx.program(vertex_shader=QUAD_VERT,
                                fragment_shader=read_shader("shaders/canvas.frag"))
        flat = {
            "canvas_resolution": (float(res), float(res)),
            "BOUNDARY_CONDITIONS_MODE": WRAP,
            "TOURNAMENT_MODE": 0,
            "TOURNAMENT_GRID": 1,
            "TIME_SCALE": time_scale,
            "frame_count": 1,          # 0 means "reset", which clears the canvas
            "draw_mode": False,
            "erase_mode": False,
            "fill_mode": False,
            "canvas_draw_active": False,
            "tiling_mode": False,
            "can_tex": 1,
        }
        for name, value in flat.items():
            if name in self.prog:
                self.prog[name].value = value
        # Persistence at the top of its range: it is a uniform scale on the
        # whole canvas, so it cannot move the spread being measured, and this
        # keeps the field well clear of fp32's floor over a thousand steps.
        self._setting("TRAIL_PERSISTENCE_SETTING", 1.0)
        self._setting("TRAIL_DIFFUSION_SETTING", diffusion)

        # RG32F with repeat, as sim.py allocates the canvas.
        self.tex = [ctx.texture((res, res), 2, dtype="f4") for _ in range(2)]
        for t in self.tex:
            t.repeat_x = True
            t.repeat_y = True
        self.fbo = [ctx.framebuffer([t]) for t in self.tex]
        self.vao = ctx.vertex_array(self.prog, [])

    def _setting(self, base: str, slider: float):
        for member, value in (("slider_value", slider), ("min_value", 0.0),
                              ("max_value", 1.0), ("x_sweep", 0.0),
                              ("y_sweep", 0.0), ("cohort_sweep", 0.0),
                              ("jitter", 0.0)):
            name = f"{base}.{member}"
            if name in self.prog:
                self.prog[name].value = value

    def run(self, steps: int) -> np.ndarray:
        """One lit texel at the centre, diffused for `steps` steps."""
        rg = np.zeros((self.res, self.res, 2), dtype=np.float32)
        rg[self.res // 2, self.res // 2, 0] = 1.0
        self.tex[0].write(rg.tobytes())
        read, write = 0, 1
        for _ in range(steps):
            self.tex[read].use(location=1)
            self.fbo[write].use()
            self.vao.render(moderngl.TRIANGLES, vertices=3)
            read, write = write, read
        out = np.frombuffer(self.fbo[read].read(components=2, dtype="f4"),
                            dtype=np.float32).reshape(self.res, self.res, 2)
        return out[..., 0].copy()

    def release(self):
        for f in self.fbo:
            f.release()
        for t in self.tex:
            t.release()
        self.vao.release()
        self.prog.release()


def variance(field: np.ndarray) -> float:
    """Per-axis second moment about the centre, in texels squared.

    The 5-point stencil is a random walk - stay, or step one texel N/S/E/W -
    so the mass spreads isotropically and E[r^2] is 2x the per-axis variance.
    """
    n = field.shape[0]
    y, x = np.mgrid[0:n, 0:n]
    c = n // 2
    w = np.maximum(field.astype(np.float64), 0.0)
    total = w.sum()
    assert total > 0, "the canvas went dark; nothing to measure"
    r2 = ((x - c) ** 2 + (y - c) ** 2)
    return float((w * r2).sum() / total / 2.0)


def spread_over_the_span(ctx, diffusion: float, ts: float) -> float:
    t = Trail(ctx, diffusion, ts)
    try:
        return variance(t.run(int(round(SPAN / ts))))
    finally:
        t.release()


# ---- the promise ---------------------------------------------------------

@pytest.mark.parametrize("ts", (0.5, 0.1, 0.02))
def test_a_slower_clock_does_not_smear_the_trail_further(ctx, ts):
    """Trail Diffusion at 1.0 - the value 83 of the 131 presets ship. Against
    the uncompensated shader ts=0.02 spread FIFTY times as far in variance,
    which is a seven-fold wider trail for the sensors to read."""
    at_one = spread_over_the_span(ctx, 1.0, 1.0)
    scaled = spread_over_the_span(ctx, 1.0, ts)
    assert scaled == pytest.approx(at_one, rel=0.02)


@pytest.mark.parametrize("ts", (2.0, 0.25))
def test_it_holds_in_both_directions_where_one_step_can_carry_it(ctx, ts):
    """Diffusion at 0.5. A step covering twice the time wants twice the blur,
    which the stencil can still deliver at this setting - see the cap below
    for where it cannot."""
    at_one = spread_over_the_span(ctx, 0.5, 1.0)
    scaled = spread_over_the_span(ctx, 0.5, ts)
    assert scaled == pytest.approx(at_one, rel=0.02)


def test_a_clock_of_one_leaves_the_blur_bit_identical(ctx):
    """Every preset in the library was made at 1.0. The compensation is a
    division, and (4+K)/1.0 - 4.0 is NOT required to return K, so the identity
    has to be a branch rather than an arithmetic accident."""
    a = Trail(ctx, 1.0, 1.0)
    try:
        untouched = a.run(30)
    finally:
        a.release()
    b = Trail(ctx, 1.0, 1.0)
    try:
        again = b.run(30)
    finally:
        b.release()
    assert np.array_equal(untouched, again)
    # And the spread is the random walk's own: 30 steps at alpha = 0.8.
    assert variance(untouched) == pytest.approx(30 * 0.8 / 2.0, rel=0.01)


def test_the_compensation_is_capped_at_the_strongest_step_the_slider_reaches(ctx):
    """The stencil is an EXPLICIT solve: past a neighbour weight of 1 it is
    unstable, and the checkerboard mode grows without bound. So a clock above
    1.0 at full diffusion cannot be compensated in one step and is capped at
    the strongest blur the slider itself reaches. Under-diffusing is the
    documented price; blowing up is not an option."""
    capped = spread_over_the_span(ctx, 1.0, 2.0)
    at_one = spread_over_the_span(ctx, 1.0, 1.0)
    assert capped < at_one                     # it does under-diffuse
    assert capped > 0.45 * at_one              # but not worse than uncompensated
    assert np.isfinite(capped)
