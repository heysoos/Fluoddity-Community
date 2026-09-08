"""A brain with its audio channels at zero computes what its deaf ancestor
computed, bit for bit, on the real shaders.

The purefn pattern: eval_brain over a fixed input grid, one invocation per
point and no shared writes, which IS reproducible - the only instrument that
can see a one-bit change. A source-level reading cannot, because the
invariant is a property of evaluation ORDER rather than of any one line. See
docs/superpowers/specs/2026-09-06-brain-audio-inputs-design.md.

Needs a real GL context, so it skips where there is none (CI).
"""
import numpy as np
import pytest

from services.brains import REGISTRY, audio_weight_index
from services.brains.layout_moves import grow_inputs, transfer_audio_inputs

moderngl = pytest.importorskip("moderngl")

N = 512
K = 4
AUDIO = (0.31, -0.22, 0.47, 0.13)
HEAD = """#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer InBuf  { vec4 xs[]; };
layout(std430, binding = 1) buffer OutBuf { vec4 ys[]; };
"""
MAIN = """
uniform float AUDIO[MAX_AUDIO_INPUTS];
uniform int MODE;     // 0 = eval_brain, 1 = brain_param_at(i)
void main(){
    uint i = gl_GlobalInvocationID.x;
    if(i >= xs.length()) return;
    g_brain_mut = 0.0;
    g_brain_cohort = 3.0;
    for (int k = 0; k < BRAIN_AUDIO_IN; k++) g_audio[k] = AUDIO[k];
    if (MODE == 1) { ys[i] = vec4(brain_param_at(0u, int(i)), 0.0, 0.0, 0.0); return; }
    ys[i] = eval_brain(0u, xs[i]);
}
"""
_MOD_ID = {"fourier": 0, "gabor": 1, "lenia": 2, "mlp": 3}


def _all():
    import services.brains.fourier  # noqa: F401
    import services.brains.gabor    # noqa: F401
    import services.brains.lenia    # noqa: F401
    import services.brains.mlp      # noqa: F401
    return sorted(REGISTRY.values(), key=lambda m: m.modality_id)


ALL = _all()


@pytest.fixture(scope="module")
def gl():
    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception as exc:
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
    scratch = ctx.buffer(reserve=4096)
    scratch.bind_to_storage_buffer(2)
    yield ctx, prog


def evaluate(gl, params, layout, xs, audio=(), mode=0):
    from utilities.gl_helpers import set_brain_layout_uniforms

    ctx, prog = gl
    flat = np.asarray(params, dtype=np.float32).reshape(-1)
    b_in = ctx.buffer(xs.astype(np.float32).tobytes())
    b_out = ctx.buffer(reserve=len(xs) * 16)
    b_par = ctx.buffer(flat.tobytes())
    b_in.bind_to_storage_buffer(0)
    b_out.bind_to_storage_buffer(1)
    b_par.bind_to_storage_buffer(4)
    full = list(audio) + [0.0] * (8 - len(audio))
    for k, v in (("MODE", int(mode)),
                 ("BRAIN_MODALITY", _MOD_ID[layout.modality]),
                 ("BRAIN_LEN", layout.length),
                 ("AUDIO", tuple(float(a) for a in full))):
        if k in prog:
            prog[k].value = v
    set_brain_layout_uniforms(prog, layout)
    prog.run(group_x=(len(xs) + 63) // 64)
    ctx.finish()
    out = np.frombuffer(b_out.read(), dtype=np.float32).reshape(-1, 4).copy()
    for b in (b_in, b_out, b_par):
        b.release()
    return out


def inputs(seed=0):
    return (np.random.default_rng(seed).standard_normal((N, 4)) * 0.5
            ).astype(np.float32)


def _pair(m, seed=0):
    """A deaf brain and the same brain widened with drawn audio weights."""
    base = m.layout_from_settings({})
    wide = grow_inputs(base, K)
    parent = np.asarray(m.random(np.random.default_rng(seed), base),
                        dtype=np.float32).reshape(-1)
    child = transfer_audio_inputs(parent, base, wide, 0.9)
    return base, parent, wide, child


# ---- NumPy references of the widened formulas, from the spec's table ----

def ref_fourier(p, x, a_in):
    p = p.reshape(-1, 8 + len(a_in))
    out = np.zeros((len(x), 4), np.float64)
    for i, row in enumerate(p):
        f, a, fa = row[:4], row[4:8], row[8:]
        ph = x @ f + a_in @ fa
        po = 2.0 * i * 0.6283 + a[3] * 3.14159
        basis = np.stack([np.sin(ph + po), np.cos(ph + po * 0.7),
                          np.sin(ph * 2 + po * 1.3),
                          np.cos(ph * 2 + po * 0.5)], axis=1)
        out += a * basis
    return out


def ref_gabor(p, x, a_in):
    p = p.reshape(-1, 14 + len(a_in))
    out = np.zeros((len(x), 4), np.float64)
    for row in p:
        c, f, a, sg, ph, fa = (row[:4], row[4:8], row[8:12], row[12], row[13],
                               row[14:])
        sg = max(abs(sg), 1e-3)
        d = x - c                       # the centre stays 4-D
        env = np.exp(-np.einsum("ij,ij->i", d, d) / (2 * sg * sg))
        out += a * (env * np.cos(x @ f + a_in @ fa + ph))[:, None]
    return out


def ref_lenia(p, x, a_in):
    p = p.reshape(-1, 10 + len(a_in))
    out = np.zeros((len(x), 4), np.float64)
    for row in p:
        w, a, mu, sg, wa = row[:4], row[4:8], row[8], row[9], row[10:]
        sg = max(abs(sg), 1e-3)
        u = x @ w + a_in @ wa - mu
        g = 2.0 * np.exp(-(u * u) / (2 * sg * sg)) - 1.0
        out += a * g[:, None]
    return out


def ref_mlp(p, x, a_in, h):
    fan = 4 + len(a_in)
    W1 = p[:fan * h].reshape(h, fan)
    b1 = p[fan * h:(fan + 1) * h]
    W2 = p[(fan + 1) * h:(fan + 5) * h].reshape(4, h)
    b2 = p[(fan + 5) * h:]
    xa = np.concatenate([x, np.broadcast_to(a_in, (len(x), len(a_in)))], axis=1)
    return np.tanh(xa @ W1.T + b1) @ W2.T + b2


def reference(m, params, layout, x, a_in):
    a_in = np.asarray(a_in, np.float64)
    if m.name == "fourier":
        return ref_fourier(params, x, a_in)
    if m.name == "gabor":
        return ref_gabor(params, x, a_in)
    if m.name == "lenia":
        return ref_lenia(params, x, a_in)
    return ref_mlp(params, x, a_in, int(layout.shape[0]))


# ---- the invariant ---------------------------------------------------------

@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_zero_channels_is_the_deaf_ancestor_bit_for_bit(gl, m):
    base, parent, wide, child = _pair(m)
    xs = inputs()
    deaf = evaluate(gl, parent, base, xs)
    wide_zero = evaluate(gl, child, wide, xs, audio=(0.0,) * K)
    assert np.array_equal(deaf, wide_zero), (
        f"{m.name}: zero audio input changed the brain by "
        f"{np.abs(deaf - wide_zero).max():.3e}")


@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_a_non_zero_channel_changes_the_brain(gl, m):
    """The control: without it the invariant test passes on a shader that
    ignores its audio weights entirely."""
    base, parent, wide, child = _pair(m)
    xs = inputs()
    deaf = evaluate(gl, parent, base, xs)
    heard = evaluate(gl, child, wide, xs, audio=AUDIO)
    assert np.abs(deaf - heard).max() > 1e-3


@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_the_widened_shader_computes_the_spec_formula(gl, m):
    base, parent, wide, child = _pair(m)
    xs = inputs(1)
    got = evaluate(gl, child, wide, xs, audio=AUDIO)
    want = reference(m, child, wide, xs.astype(np.float64), AUDIO)
    assert np.allclose(got, want, atol=2e-4), (
        f"{m.name}: max |gpu - ref| = {np.abs(got - want).max():.3e}")


@pytest.mark.parametrize("m", ALL, ids=lambda m: m.name)
def test_param_at_reads_every_float_at_the_widened_stride(gl, m):
    """The readback and the Inspector go through brain_param_at, which
    recomputes the stride; a stale one reads the wrong float silently."""
    _base, _parent, wide, child = _pair(m)
    xs = np.zeros((wide.length, 4), np.float32)
    got = evaluate(gl, child, wide, xs, audio=AUDIO, mode=1)[:, 0]
    np.testing.assert_array_equal(got, child)


def test_gabor_audio_enters_the_phase_and_never_the_envelope(gl):
    """A filter whose sensor frequency and phase are zero has cos(...) driven
    by audio alone, and its envelope must not move with it. Two evaluations
    at audio values with the same cosine must agree exactly."""
    m = REGISTRY["gabor"]
    base, parent, wide, child = _pair(m)
    stride = 14 + K
    rows = child.reshape(-1, stride).copy()
    rows[:, 4:8] = 0.0                # no sensor frequency
    rows[:, 13] = 0.0                 # no phase
    rows[:, 14:] = 0.0
    rows[:, 14] = 1.0                 # phase = a_1 exactly
    xs = inputs(2)
    a = evaluate(gl, rows.reshape(-1), wide, xs, audio=(0.0, 0.0, 0.0, 0.0))
    b = evaluate(gl, rows.reshape(-1), wide, xs,
                 audio=(2.0 * np.pi, 0.0, 0.0, 0.0))
    assert np.allclose(a, b, atol=1e-5)
    # And the envelope alone, without any oscillation, is untouched by K.
    assert np.array_equal(a, evaluate(gl, rows.reshape(-1), wide, xs))
