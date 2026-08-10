"""The canvas border under a NON-WRAP boundary, on a real GPU.

The world reflects but the senses wrapped. `get_can()` fract()-ed uv only in
wrap mode; under bounce or reset uv left [0,1] and fell through to the sampler,
which has repeat_x/repeat_y set (sim.py:132) and duly returned the opposite edge
of the world. A particle at the left wall steered on what was at the right wall.
The trail diffusion in canvas.frag leaked the same way through the same sampler.

Measured before the fix, border/interior luminance over physics_configs/Core:
wrap presets 1.0-2.1x, bounce presets 2.5-10.7x. See
docs/superpowers/specs/2026-08-09-bounce-boundary-sensing.md.

CLAMPING IS NOT THE FIX and these tests enforce that. Clamping the sampler was
measured to roughly DOUBLE the artifact (LavaLamp 10.58x -> 16.06x) because both
sensors then land on the same texel within sample_dist of the wall, so the
steering differential is identically zero in a band round the whole border -
the same dead band CLAUDE.md documents for tournament tiles. The fix is a MIRROR
(zero-flux/Neumann), which keeps the two sensors distinct.

Runs the shipped shader text: the helpers are read from disk and only main() is
supplied. Skipped when no GL 4.3 context is available.
"""
from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from utilities.gl_helpers import read_shader  # noqa: E402

RES = 128
BOUNCE, RESET, WRAP = 0, 1, 2
HALF_TEXEL = 0.5 / RES


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:
        pytest.skip(f"no GL 4.3 context: {exc}")
    yield c
    c.release()


# ---- the sensor: entity_update.glsl get_can() ---------------------------

PROBE_SRC = """#version 430
layout(local_size_x = 64) in;
uniform sampler2D canvas;
uniform int BOUNDARY_CONDITIONS_MODE;
int get_particle_boundary_conditions(){ return BOUNDARY_CONDITIONS_MODE; }
layout(std430, binding = 0) buffer In  { vec2 pts[]; };
layout(std430, binding = 1) buffer Out { vec4 res[]; };
"""

PROBE_MAIN = """
void main(){
    uint i = gl_GlobalInvocationID.x;
    if(i >= pts.length()) return;
    res[i] = get_can(pts[i]);
}
"""


def _sense_helpers() -> str:
    """The real sense_uv + get_can, lifted from entity_update.glsl."""
    src = read_shader("shaders/entity_update.glsl")
    start = src.index("vec2 sense_uv(")
    return src[start:src.index("vec4 get_field(", start)]


@pytest.fixture(scope="module")
def sense(ctx):
    """-> f(points, mode) = the canvas value each world point senses.

    The canvas is a horizontal ramp, value == u. So the returned red channel
    IS the u the shader sampled, which makes wrap and mirror trivially
    distinguishable: a point just past the left wall reads ~0 if mirrored and
    ~1 if wrapped.
    """
    prog = ctx.compute_shader(PROBE_SRC + _sense_helpers() + PROBE_MAIN)
    ramp = np.zeros((RES, RES, 4), dtype=np.float32)
    ramp[..., 0] = (np.arange(RES, dtype=np.float32) + 0.5) / RES   # texel centres
    ramp[..., 1] = ((np.arange(RES, dtype=np.float32) + 0.5) / RES)[:, None]
    tex = ctx.texture((RES, RES), 4, dtype="f4")
    tex.write(ramp.tobytes())
    tex.repeat_x = True                        # exactly as sim.py sets them
    tex.repeat_y = True
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.use(location=3)
    prog["canvas"].value = 3

    def run(points, mode):
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        bin_ = ctx.buffer(pts.tobytes())
        bout = ctx.buffer(reserve=len(pts) * 16)
        bin_.bind_to_storage_buffer(0)
        bout.bind_to_storage_buffer(1)
        prog["BOUNDARY_CONDITIONS_MODE"].value = int(mode)
        prog.run(group_x=(len(pts) + 63) // 64)
        out = np.frombuffer(bout.read(), dtype=np.float32).reshape(-1, 4)
        for b in (bin_, bout):
            b.release()
        return out[:, :2].copy()

    yield run
    tex.release()
    prog.release()


def _world_x(u: float) -> float:
    """World x whose uv is u. Square canvas, so half_extent is (1,1)."""
    return 2.0 * u - 1.0


@pytest.mark.parametrize("mode", [BOUNCE, RESET])
@pytest.mark.parametrize("depth", [0.02, 0.05, 0.13])
def test_a_wall_mirrors_the_sensor_instead_of_wrapping(sense, mode, depth):
    """The defect itself. A sensor `depth` past the left wall must read the
    canvas `depth` INSIDE that wall, not `depth` inside the opposite one."""
    got = sense([[_world_x(-depth), 0.0]], mode)[0][0]
    assert got == pytest.approx(depth, abs=2.0 / RES), (
        f"read u={got:.4f}, wanted the mirror {depth:.4f}")
    assert abs(got - (1.0 - depth)) > 0.1, "it wrapped to the far wall"


@pytest.mark.parametrize("mode", [BOUNCE, RESET])
def test_the_right_wall_mirrors_too(sense, mode):
    got = sense([[_world_x(1.0 + 0.05), 0.0]], mode)[0][0]
    assert got == pytest.approx(0.95, abs=2.0 / RES)


@pytest.mark.parametrize("mode", [BOUNCE, RESET])
def test_both_axes_mirror(sense, mode):
    """A corner is where the two folds meet; getting one axis right and the
    other wrong is exactly the kind of bug that only shows in two corners."""
    out = sense([[_world_x(-0.05), _world_x(-0.05)]], mode)[0]
    assert out[0] == pytest.approx(0.05, abs=2.0 / RES)
    assert out[1] == pytest.approx(0.05, abs=2.0 / RES)


def test_wrap_still_wraps(sense):
    """The other half of the statement, and the regression guard: three of the
    Core presets are wrap and must be untouched."""
    got = sense([[_world_x(-0.05), 0.0]], WRAP)[0][0]
    assert got == pytest.approx(0.95, abs=2.0 / RES)


@pytest.mark.parametrize("mode", [BOUNCE, RESET, WRAP])
def test_the_interior_is_untouched(sense, mode):
    """99% of the canvas must sample exactly as before, in every mode."""
    us = np.linspace(0.1, 0.9, 33)
    pts = [[_world_x(u), 0.0] for u in us]
    got = sense(pts, mode)[:, 0]
    assert np.allclose(got, us, atol=2.0 / RES)


@pytest.mark.parametrize("mode", [BOUNCE, RESET])
def test_two_sensors_past_the_wall_stay_distinct(sense, mode):
    """What rules out the clamp.

    A particle heading into the wall has BOTH sensors past it. Clamping sends
    both to the same texel, the steering differential is identically zero, and
    the result is a dead band of width sample_dist round the entire border -
    measured to be worse than the wrap it replaced. A mirror keeps them apart,
    and keeps them the same distance apart as they started.
    """
    a, b = -0.02, -0.09
    got = sense([[_world_x(a), 0.0], [_world_x(b), 0.0]], mode)[:, 0]
    assert abs(got[0] - got[1]) == pytest.approx(0.07, abs=2.0 / RES), (
        f"the two sensors collapsed to {got} - this is the clamp, not a mirror")


@pytest.mark.parametrize("mode", [BOUNCE, RESET])
def test_the_sample_at_the_wall_never_blends_the_far_edge(sense, mode):
    """A mirror alone is not enough. texture() is bilinear and the sampler
    repeats, so a coordinate nearer the seam than half a texel still blends in
    the texel from the OPPOSITE edge - the leak, at reduced weight. The fold
    must land no closer than the first texel CENTRE.
    """
    got = sense([[_world_x(0.0), 0.0], [_world_x(1.0), 0.0]], mode)
    assert got[0][0] == pytest.approx(HALF_TEXEL, abs=1e-3), (
        "the left wall sample is a blend with the right edge")
    assert got[1][0] == pytest.approx(1.0 - HALF_TEXEL, abs=1e-3)


@pytest.mark.parametrize("mode", [BOUNCE, RESET, WRAP])
def test_no_sensor_anywhere_escapes_the_canvas(sense, mode):
    """Swept far outside, including several worlds away - fract() and the fold
    both have to keep every reading on the canvas."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-6.0, 6.0, size=(512, 2)).astype(np.float32)
    got = sense(pts, mode)
    assert np.all(got >= -1e-4) and np.all(got <= 1.0 + 1e-4)


# ---- the trail: canvas.frag getBlur() -----------------------------------

QUAD_VERT = """#version 430
out vec2 texcoord;
void main(){
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    texcoord = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

BLUR_MAIN = """
uniform float K_TEST;
void main(){ can_out = getBlur(texcoord, can_tex, K_TEST); }
"""


def _diffuse(ctx, field, mode, steps):
    """The real getBlur, ping-ponged, with TOURNAMENT_MODE off."""
    src = read_shader("shaders/canvas.frag")
    prog = ctx.program(vertex_shader=QUAD_VERT,
                       fragment_shader=src[:src.index("void main()")] + BLUR_MAIN)
    for name, value in (("BOUNDARY_CONDITIONS_MODE", mode),
                        ("TOURNAMENT_MODE", 0), ("TOURNAMENT_GRID", 4),
                        ("canvas_resolution", (RES, RES)), ("K_TEST", 1.0)):
        if name in prog:
            prog[name].value = value
    rgba = np.zeros((RES, RES, 4), dtype=np.float32)
    rgba[..., 0] = field
    tex = [ctx.texture((RES, RES), 4, dtype="f4") for _ in range(2)]
    for t in tex:
        t.repeat_x = True
        t.repeat_y = True
    fbo = [ctx.framebuffer([t]) for t in tex]
    tex[0].write(rgba.tobytes())
    vao = ctx.vertex_array(prog, [])
    read, write = 0, 1
    for _ in range(steps):
        tex[read].use(location=1)
        if "can_tex" in prog:
            prog["can_tex"].value = 1
        fbo[write].use()
        vao.render(moderngl.TRIANGLES, vertices=3)
        read, write = write, read
    out = np.frombuffer(fbo[read].read(components=4, dtype="f4"),
                        dtype=np.float32).reshape(RES, RES, 4)[..., 0].copy()
    for f in fbo:
        f.release()
    for t in tex:
        t.release()
    vao.release()
    prog.release()
    return out


@pytest.mark.parametrize("mode", [BOUNCE, RESET])
def test_the_trail_does_not_diffuse_across_the_canvas_border(ctx, mode):
    """The diffusion half. A blob on the left edge reached the right edge
    through the same repeating sampler."""
    field = np.zeros((RES, RES), dtype=np.float32)
    field[:, 0:2] = 1.0
    out = _diffuse(ctx, field, mode, steps=60)
    assert float(out[:, -1].max()) < 1e-6, "the trail crossed the world border"


def test_wrap_still_carries_the_trail_across(ctx):
    field = np.zeros((RES, RES), dtype=np.float32)
    field[:, 0:2] = 1.0
    out = _diffuse(ctx, field, WRAP, steps=60)
    assert float(out[:, -1].mean()) > 1e-3


@pytest.mark.parametrize("mode", [BOUNCE, RESET, WRAP])
def test_the_trail_is_conserved(ctx, mode):
    """A torus and a zero-flux wall both conserve mass exactly. Mass leaving
    means the canvas is losing trail over the edge; mass arriving means the
    kernel is double-counting a mirrored tap."""
    rng = np.random.default_rng(1)
    field = rng.random((RES, RES)).astype(np.float32)
    out = _diffuse(ctx, field, mode, steps=120)
    assert float(out.sum()) == pytest.approx(float(field.sum()), rel=1e-4)
