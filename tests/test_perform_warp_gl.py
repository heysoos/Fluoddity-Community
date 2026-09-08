"""The corner-pin warp, run through the real shader on a real GPU.

Unlike tests/test_perform_view_gl.py these assertions are EXACT: the source is
a synthetic texture rather than the particle pass, so there is no additive
splat race and a known point moves to a known place.

The source encodes its own coordinates - R = u, G = v - so every rendered
pixel says which source texel it came from. That turns "is the matrix
transposed, flipped or rotated" into one comparison against the inverse map,
which no amount of reading the shader can settle.

`fbo.read()` returns rows from the BOTTOM, so row index IS display v. No flip
here; the flip belongs to the ImGui proxy widget.
"""
from __future__ import annotations

import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from services.corner_pin import (  # noqa: E402
    homography, inverse_homography, map_point,
)
from utilities.gl_helpers import read_shader  # noqa: E402

N = 64

# Asymmetric on every axis: a mirror-symmetric quad passes under a v flip.
SKEW = ((0.14, 0.93), (0.88, 0.78), (0.95, 0.16), (0.07, 0.09))
FULL = ((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0))


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=330)
    except Exception as exc:
        pytest.skip(f"no GL context: {exc}")
    yield c
    c.release()


@pytest.fixture(scope="module")
def rig(ctx):
    """The real perform shaders, a coordinate-encoding source, an f4 target."""
    prog = ctx.program(
        vertex_shader=read_shader('shaders/perform.vert'),
        fragment_shader=read_shader('shaders/perform.frag'),
    )
    quad = np.array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1],
                    dtype='f4')
    vao = ctx.vertex_array(prog, [(ctx.buffer(quad.tobytes()),
                                   '2f', 'in_position')])

    u = (np.arange(N) + 0.5) / N
    src = np.zeros((N, N, 4), dtype='f4')
    src[:, :, 0] = u[None, :]           # R = u
    src[:, :, 1] = u[:, None]           # G = v, row 0 at the bottom
    src[:, :, 2] = 1.0                  # B marks "this came from the source"
    src[:, :, 3] = 1.0
    tex = ctx.texture((N, N), 4, src.tobytes(), dtype='f4')
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

    target = ctx.texture((N, N), 4, dtype='f4')
    fbo = ctx.framebuffer(color_attachments=[target])
    yield prog, vao, tex, fbo
    for obj in (fbo, target, tex, vao, prog):
        obj.release()


def render(rig, corners, guides=False, held=-1):
    prog, vao, tex, fbo = rig
    inv = inverse_homography(corners)
    assert inv is not None, "the test asked for a degenerate quad"

    fbo.use()
    fbo.ctx.viewport = (0, 0, N, N)
    fbo.clear(0.0, 0.0, 0.0, 1.0)
    tex.use(location=0)
    prog['view_tex'].value = 0
    # GLSL reads a mat3 column-major; the solve is row-major.
    prog['inv_h'].write(np.ascontiguousarray(inv.T, dtype='f4').tobytes())
    prog['fb_size'].value = (float(N), float(N))
    prog['show_guides'].value = 1 if guides else 0
    prog['held_corner'].value = held
    prog['corners'].write(
        np.ascontiguousarray(corners, dtype='f4').tobytes())
    vao.render()
    raw = np.frombuffer(fbo.read(components=4, dtype='f4'), dtype='f4')
    return raw.reshape(N, N, 4)


def inside(img):
    """Mask of pixels the source actually reached."""
    return img[:, :, 2] > 0.5


def test_an_identity_quad_reproduces_the_source(rig):
    img = render(rig, FULL)
    u = (np.arange(N) + 0.5) / N
    assert img[:, :, 0] == pytest.approx(np.tile(u, (N, 1)), abs=1e-5)
    assert img[:, :, 1] == pytest.approx(np.tile(u[:, None], (1, N)), abs=1e-5)
    assert inside(img).all()


def test_every_pixel_comes_from_where_the_inverse_map_says(rig):
    """The whole contract in one assertion: transpose, flip and rotation."""
    img = render(rig, SKEW)
    inv = inverse_homography(SKEW)
    mask = inside(img)
    assert mask.sum() > 0.3 * N * N, "the warp lost most of the picture"

    rows, cols = np.nonzero(mask)
    for r, c in zip(rows[::7], cols[::7]):
        disp = ((c + 0.5) / N, (r + 0.5) / N)
        want = map_point(inv, disp)
        got = (img[r, c, 0], img[r, c, 1])
        # One texel of slack: the source is sampled NEAREST.
        assert got == pytest.approx(want, abs=1.5 / N), f"at row {r} col {c}"


def test_outside_the_quad_is_black(rig):
    img = render(rig, SKEW)
    # The bottom-left display corner is outside this quad.
    assert not inside(img)[0, 0]
    assert img[0, 0, :3] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)


def test_the_source_corners_land_on_the_quad_corners(rig):
    """The orientation pin. A v flip moves TL to the bottom and fails here."""
    img = render(rig, SKEW)
    h = homography(SKEW)
    src_corners = ((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0))

    for (su, sv), dst in zip(src_corners, SKEW):
        # Step a little inside the quad so the sample is unambiguous.
        inner = map_point(h, (0.5 + 0.45 * (su - 0.5),
                              0.5 + 0.45 * (sv - 0.5)))
        c = int(inner[0] * N)
        r = int(inner[1] * N)
        assert inside(img)[r, c], f"quad corner {dst} is not covered"
        # The nearer source corner in the picture is the one we aimed at.
        got = (img[r, c, 0], img[r, c, 1])
        assert abs(got[0] - su) < 0.5 and abs(got[1] - sv) < 0.5


def test_guides_are_absent_until_calibration_is_on(rig):
    plain = render(rig, SKEW, guides=False)
    lit = render(rig, SKEW, guides=True, held=0)
    assert not np.allclose(plain, lit), "the guides drew nothing"


def test_the_guides_leave_the_warp_alone_when_off(rig):
    a = render(rig, SKEW, guides=False, held=2)
    b = render(rig, SKEW, guides=False, held=-1)
    assert np.array_equal(a, b), "held_corner drew with the guides off"
