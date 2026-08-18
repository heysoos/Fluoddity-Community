"""The canvas and the brush are RG32F, and the trail is a velocity field.

Halving them from RGBA32F is worth 5-16% of the sim step (see CLAUDE.md), and
it is safe only because nothing reads a third channel: the brain senses .xy,
the canvas view draws atan(y,x) and length(xy), and the alpha the blend needs
comes from the FRAGMENT rather than from the target.

Going back to four components would not fail anywhere - it would just quietly
cost the frame time again - so the format is asserted rather than assumed.
"""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_the_canvas_and_brush_are_two_component():
    src = (ROOT / "sim.py").read_text(encoding="utf-8")
    assert src.count("self.ctx.texture(canvas_shape, 2, dtype='f4')") == 3, (
        "both canvas buffers and the brush must be RG32F"
    )
    assert "canvas_shape, 4" not in src


def test_the_brush_still_writes_the_blend_weight():
    """The blend is SRC_ALPHA, ONE. The target has no alpha, but the FACTOR is
    the fragment's - zero it and every deposit multiplies to nothing."""
    src = (ROOT / "shaders" / "brush.frag").read_text(encoding="utf-8")
    assert "brush_out = vec4(vel, 0, 1) * kernel_func;" in src


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
