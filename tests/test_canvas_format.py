"""The canvas is RG32F, and the trail is a velocity field.

Halving it from RGBA32F is worth 5-16% of the sim step (see CLAUDE.md), and it
is safe only because nothing reads a third channel: the brain senses .xy and
the canvas view draws atan(y,x) and length(xy).

Going back to four components would not fail anywhere - it would just quietly
cost the frame time again - so the format is asserted rather than assumed.

There is no brush texture any more. The deposit goes straight into the canvas
framebuffer, which is why the blend weight moved out of the alpha channel and
into the fragment.
"""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_both_canvas_buffers_are_two_component():
    src = (ROOT / "sim.py").read_text(encoding="utf-8")
    assert src.count("self.ctx.texture(canvas_shape, 2, dtype='f4')") == 2, (
        "both canvas buffers must be RG32F, and nothing else is allocated here"
    )
    assert "canvas_shape, 4" not in src


def test_the_brush_squares_the_kernel_itself():
    """The blend used to be SRC_ALPHA, ONE with alpha = kernel_func, so the
    kernel was applied TWICE - once in the output and once by the blend. The
    blend is ONE, ONE now, so the second application has to be explicit or
    every deposit is softer and broader than it was."""
    src = (ROOT / "shaders" / "brush.frag").read_text(encoding="utf-8")
    assert "kernel_func * kernel_func" in src


def test_the_brush_carries_the_deposits_share_of_the_average():
    """canvas.frag decays by trail_persistence and no longer adds anything, so
    the (1 - p) half of that one moving average has to be applied here."""
    src = (ROOT / "shaders" / "brush.frag").read_text(encoding="utf-8")
    assert "(1.0 - p)" in src
    assert "TRAIL_PERSISTENCE_SETTING" in src, (
        "computed per fragment, not taken as a whole-canvas uniform: sweeps and "
        "jitter make it vary across the canvas"
    )


# ---- the driver assumption the whole change rests on ----------------------

def test_rg_blending_matches_rgba_bit_for_bit():
    """SRC_ALPHA, ONE into a target with no alpha channel. The source factor
    comes from the fragment, so this must be exact - but a format with no
    destination alpha is precisely where a driver could differ."""
    moderngl = pytest.importorskip("moderngl")
    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception as exc:                       # no GPU / no driver
        pytest.skip(f"no standalone GL context: {exc}")

    prog = ctx.program(
        vertex_shader="""
        #version 430
        out vec2 uv;
        void main(){
            vec2 o[4] = vec2[4](vec2(-1,-1), vec2(1,-1), vec2(1,1), vec2(-1,1));
            vec2 u[4] = vec2[4](vec2(0,0), vec2(1,0), vec2(1,1), vec2(0,1));
            int i = gl_InstanceID;
            vec2 c = vec2(sin(float(i) * 2.3), cos(float(i) * 1.7)) * 0.4;
            gl_Position = vec4(c + o[gl_VertexID] * 0.35, 0, 1);
            uv = u[gl_VertexID];
        }""",
        fragment_shader="""
        #version 430
        in vec2 uv;
        out vec4 frag;
        void main(){
            float k = exp(-dot(uv - 0.5, uv - 0.5) / 0.05);
            frag = vec4((uv - 0.5) * 0.7, 0.0, 1.0) * k;   // brush.frag's shape
        }""")
    vao = ctx.vertex_array(prog, [])
    res = 128
    out = {}
    for comps in (4, 2):
        tex = ctx.texture((res, res), comps, dtype="f4")
        fbo = ctx.framebuffer([tex])
        fbo.use()
        fbo.clear()
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        ctx.blend_equation = moderngl.FUNC_ADD
        # Overlapping, so most texels are blended many times over. Instanced
        # draws have a defined primitive order, so this is exactly comparable.
        vao.render(moderngl.TRIANGLE_FAN, vertices=4, instances=64)
        ctx.disable(moderngl.BLEND)
        ctx.finish()
        out[comps] = np.frombuffer(tex.read(), np.float32).reshape(
            res, res, comps)
        tex.release()
        fbo.release()

    assert (np.abs(out[4]) > 0).any(), "nothing was drawn - the test is vacuous"
    assert np.array_equal(out[4][:, :, :2], out[2])
    ctx.release()
