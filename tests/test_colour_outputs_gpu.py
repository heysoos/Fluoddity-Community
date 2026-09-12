"""The two colour outputs, evaluated by the shipped brain GLSL.

A widened legacy genome's hue is twice its own axial output and its saturation
output is zero; the colour floats reach colour and nothing else, even under
mutation. Needs a real GL context, so it skips where there is none.
"""
import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from services.brains import (COLOUR_GAIN, colour_weight_index, get,  # noqa: E402
                             widen_colour, with_audio_inputs,
                             with_colour_outputs)

HEAD = """#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer InBuf  { vec4 xs[]; };
layout(std430, binding = 1) buffer OutBuf { vec4 ys[]; };
layout(std430, binding = 3) buffer ColBuf { vec2 cs[]; };
uniform float MUT;
"""
MAIN = """
void main(){
    uint i = gl_GlobalInvocationID.x;
    if(i >= xs.length()) return;
    g_brain_mut = MUT;
    g_brain_cohort = 3.0;
    ys[i] = eval_brain(0u, xs[i]);
    cs[i] = g_brain_colour;
}
"""
CASES = [("fourier", {}), ("gabor", {}), ("lenia", {}), ("mlp", {}),
         ("mlp", {"layers": [[8, 0], [6, 1]]})]
_MOD_ID = {"fourier": 0, "gabor": 1, "lenia": 2, "mlp": 3}


@pytest.fixture(scope="module")
def gl():
    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception as exc:                       # no GPU / no driver
        pytest.skip(f"no standalone GL context: {exc}")
    from pathlib import Path

    sh = Path(__file__).resolve().parent.parent / "shaders"
    src = (HEAD
           + (sh / "fourier4_4.glsl").read_text()
           + (sh / "brains" / "_header.glsl").read_text()
           + "".join((sh / "brains" / f"{m}.glsl").read_text()
                     for m in ("fourier", "gabor", "lenia", "mlp"))
           + (sh / "brains" / "_dispatch.glsl").read_text()
           + MAIN)
    prog = ctx.compute_shader(src)
    scratch = ctx.buffer(reserve=8192)
    scratch.bind_to_storage_buffer(2)
    yield ctx, prog


def evaluate(gl, params, layout, mut=0.0, legacy=False):
    from utilities.gl_helpers import set_brain_layout_uniforms

    ctx, prog = gl
    xs = (np.random.default_rng(3).standard_normal((512, 4)) * 0.6
          ).astype(np.float32)
    bufs = [ctx.buffer(xs.tobytes()), ctx.buffer(reserve=len(xs) * 16),
            ctx.buffer(reserve=len(xs) * 8),
            ctx.buffer(np.asarray(params, np.float32).tobytes())]
    for b, at in zip(bufs, (0, 1, 3, 4)):
        b.bind_to_storage_buffer(at)
    for k, v in (("MUT", mut), ("BRAIN_MODALITY", _MOD_ID[layout.modality]),
                 ("BRAIN_LEN", layout.length), ("LEGACY_COLOR", legacy)):
        if k in prog:
            prog[k].value = v
    set_brain_layout_uniforms(prog, layout)
    prog.run(group_x=len(xs) // 64)
    ctx.finish()
    y = np.frombuffer(bufs[1].read(), np.float32).reshape(-1, 4).copy()
    c = np.frombuffer(bufs[2].read(), np.float32).reshape(-1, 2).copy()
    for b in bufs:
        b.release()
    return y, c


def _layout(name, settings, audio=0):
    return with_audio_inputs(get(name).layout_from_settings(settings), audio)


@pytest.mark.parametrize("name,settings", CASES)
@pytest.mark.parametrize("audio", [0, 2])
def test_a_widened_legacy_genome_reads_hue_off_its_axial_output(
        gl, name, settings, audio):
    legacy = with_colour_outputs(_layout(name, settings, audio), 0)
    p = get(name).random(np.random.default_rng(7), legacy)
    layout, w = widen_colour(p, legacy)
    y, c = evaluate(gl, w, layout)
    np.testing.assert_allclose(c[:, 0], COLOUR_GAIN * y[:, 0],
                               rtol=1e-6, atol=1e-6)
    assert not c[:, 1].any()


@pytest.mark.parametrize("name,settings", CASES)
@pytest.mark.parametrize("mut", [0.0, 0.3])
def test_colour_floats_move_colour_and_nothing_else(gl, name, settings, mut):
    layout = _layout(name, settings)
    rng = np.random.default_rng(9)
    p = get(name).random(rng, layout)
    q = p.copy()
    idx = colour_weight_index(layout)
    q[idx] = rng.normal(0.0, 1.0, idx.size)
    y_p, c_p = evaluate(gl, p, layout, mut)
    y_q, c_q = evaluate(gl, q, layout, mut)
    np.testing.assert_array_equal(y_p, y_q)
    assert not np.allclose(c_p, c_q)


@pytest.mark.parametrize("name,settings", CASES[:3])
def test_a_brain_with_no_motion_still_has_colour(gl, name, settings):
    layout = _layout(name, settings)
    p = get(name).random(np.random.default_rng(4), layout)
    lo, hi = get(name).AMPLITUDE_SLICE
    stride = get(name).unit_floats(layout)
    for start in range(0, layout.length, stride):
        p[start + lo:start + hi] = 0.0
    y, c = evaluate(gl, p, layout)
    assert not y.any()
    assert np.abs(c).max() > 1e-3


@pytest.mark.parametrize("name,settings", CASES)
@pytest.mark.parametrize("mut", [0.0, 0.3])
def test_legacy_colouring_skips_the_colour_outputs_and_keeps_the_motion(
        gl, name, settings, mut):
    """The switch back to the old colouring is also the switch that pays for
    itself: every colour weight is left unread, and the four motion outputs
    must not move by a bit."""
    layout = _layout(name, settings)
    p = get(name).random(np.random.default_rng(11), layout)
    y_on, c_on = evaluate(gl, p, layout, mut)
    y_off, c_off = evaluate(gl, p, layout, mut, legacy=True)
    np.testing.assert_array_equal(y_on, y_off)
    assert np.abs(c_on).max() > 1e-6, "the genome must have colour to skip"
    assert not c_off.any()
