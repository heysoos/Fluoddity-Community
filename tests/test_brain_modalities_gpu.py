"""Each modality's GLSL must compute what its Python docstring says it does.

This is the only test that crosses the language boundary. A modality is two
hand-written halves of one layout - decode() in Python decides what each float
MEANS, brain_<name>() in GLSL decides where each float is READ - and the Python
round-trip test only ever checks Python against itself. Swap two fields and
everything else still passes while the brain computes something else entirely.

Needs a real GL context, so it skips where there is none (CI).
"""
import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

N = 512
HEAD = """#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer InBuf  { vec4 xs[]; };
layout(std430, binding = 1) buffer OutBuf { vec4 ys[]; };
"""
MAIN = """
void main(){
    uint i = gl_GlobalInvocationID.x;
    if(i >= xs.length()) return;
    g_brain_mut = MUT;
    g_brain_cohort = 3.0;
    ys[i] = eval_brain(0u, xs[i]);
}
"""


@pytest.fixture(scope="module")
def gl():
    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception as exc:                       # no GPU / no driver
        pytest.skip(f"no standalone GL context: {exc}")
    from pathlib import Path

    sh = Path(__file__).resolve().parent.parent / "shaders"
    # Verbatim shader text - no edits. The readback buffer declared in
    # _header.glsl sits at binding 2 and this harness uses 0, 1 and 4, so
    # nothing collides and brain_write() compiles as shipped.
    src = (HEAD
           + (sh / "fourier4_4.glsl").read_text()
           + (sh / "brains" / "_header.glsl").read_text()
           + "".join((sh / "brains" / f"{m}.glsl").read_text()
                     for m in ("fourier", "gabor", "lenia", "mlp"))
           + (sh / "brains" / "_dispatch.glsl").read_text()
           + "uniform float MUT;\n" + MAIN)
    prog = ctx.compute_shader(src)
    scratch = ctx.buffer(reserve=4096)
    scratch.bind_to_storage_buffer(2)
    yield ctx, prog


def evaluate(gl, params, layout, xs, mut=0.0):
    ctx, prog = gl
    flat = np.asarray(params, dtype=np.float32).reshape(-1)
    b_in = ctx.buffer(xs.astype(np.float32).tobytes())
    b_out = ctx.buffer(reserve=len(xs) * 16)
    b_par = ctx.buffer(flat.tobytes())
    b_in.bind_to_storage_buffer(0)
    b_out.bind_to_storage_buffer(1)
    b_par.bind_to_storage_buffer(4)
    shape = tuple(layout.shape) + (0, 0, 0, 0)
    for k, v in (("MUT", mut),
                 ("BRAIN_MODALITY", _MOD_ID[layout.modality]),
                 ("BRAIN_LEN", layout.length),
                 ("BRAIN_SHAPE", shape[:4])):
        if k in prog:
            prog[k].value = v
    prog.run(group_x=(len(xs) + 63) // 64)
    ctx.finish()
    out = np.frombuffer(b_out.read(), dtype=np.float32).reshape(-1, 4).copy()
    for b in (b_in, b_out, b_par):
        b.release()
    return out


_MOD_ID = {"fourier": 0, "gabor": 1, "lenia": 2, "mlp": 3}


def inputs(seed=0):
    return (np.random.default_rng(seed).standard_normal((N, 4)) * 0.5
            ).astype(np.float32)


# ---- NumPy references, written from each modality's docstring ----

def ref_fourier(p, x):
    p = p.reshape(-1, 8)
    out = np.zeros((len(x), 4), np.float64)
    for i, row in enumerate(p):
        f, a = row[:4], row[4:]
        ph = x @ f
        po = 2.0 * i * 0.6283 + a[3] * 3.14159
        basis = np.stack([np.sin(ph + po), np.cos(ph + po * 0.7),
                          np.sin(ph * 2 + po * 1.3),
                          np.cos(ph * 2 + po * 0.5)], axis=1)
        out += a * basis
    return out


def ref_gabor(p, x):
    p = p.reshape(-1, 14)
    out = np.zeros((len(x), 4), np.float64)
    for row in p:
        c, f, a, sg, ph = row[:4], row[4:8], row[8:12], row[12], row[13]
        sg = max(abs(sg), 1e-3)
        d = x - c
        env = np.exp(-np.einsum("ij,ij->i", d, d) / (2 * sg * sg))
        out += a * (env * np.cos(x @ f + ph))[:, None]
    return out


def ref_lenia(p, x):
    p = p.reshape(-1, 10)
    out = np.zeros((len(x), 4), np.float64)
    for row in p:
        w, a, mu, sg = row[:4], row[4:8], row[8], row[9]
        sg = max(abs(sg), 1e-3)
        u = x @ w - mu
        g = 2.0 * np.exp(-(u * u) / (2 * sg * sg)) - 1.0
        out += a * g[:, None]
    return out


def ref_mlp(p, x, h, act):
    W1 = p[:4 * h].reshape(h, 4)
    b1 = p[4 * h:5 * h]
    W2 = p[5 * h:9 * h].reshape(4, h)
    b2 = p[9 * h:]
    pre = x @ W1.T + b1
    if act == 0:
        a = np.tanh(pre)
    elif act == 1:
        a = np.sin(pre)
    else:
        a = 0.5 * pre * (1 + np.tanh(0.7978845608 *
                                     (pre + 0.044715 * pre ** 3)))
    return a @ W2.T + b2


def check(got, want, tag):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    scale = max(np.abs(want).mean(), 1e-6)
    err = np.abs(got - want).mean() / scale
    corr = np.corrcoef(got.ravel(), want.ravel())[0, 1]
    assert err < 1e-4 and corr > 0.9999, (
        f"{tag}: GLSL disagrees with the Python layout - "
        f"mean err {err:.3%} of signal, correlation {corr:.6f}"
    )


@pytest.mark.parametrize("name", ["fourier", "gabor", "lenia", "mlp"])
def test_glsl_matches_the_python_layout(gl, name):
    from services.brains import REGISTRY

    m = REGISTRY[name]
    layout = m.layout_from_settings({})
    p = m.decode(np.random.default_rng(5).normal(
        0, 0.6, layout.length).astype(np.float32), layout).reshape(-1)
    x = inputs()
    got = evaluate(gl, p, layout, x)
    if name == "fourier":
        want = ref_fourier(p, x)
    elif name == "gabor":
        want = ref_gabor(p, x)
    elif name == "lenia":
        want = ref_lenia(p, x)
    else:
        want = ref_mlp(p, x, layout.shape[0], layout.shape[1])
    check(got, want, name)


@pytest.mark.parametrize("name,act", [("mlp", 0), ("mlp", 1), ("mlp", 2)])
def test_every_mlp_activation_matches(gl, name, act):
    from services.brains import REGISTRY

    m = REGISTRY[name]
    layout = m.layout_from_settings({"hidden": 8, "activation": act})
    p = m.decode(np.random.default_rng(6).normal(
        0, 0.6, layout.length).astype(np.float32), layout).reshape(-1)
    x = inputs(1)
    check(evaluate(gl, p, layout, x), ref_mlp(p, x, 8, act), f"mlp act={act}")


@pytest.mark.parametrize("name", ["fourier", "gabor", "lenia", "mlp"])
def test_output_stays_finite_under_heavy_mutation(gl, name):
    """Mutation scales widths, so a modality that lets one reach zero divides by
    it and poisons the canvas with inf for every particle in the cohort."""
    from services.brains import REGISTRY

    m = REGISTRY[name]
    layout = m.layout_from_settings({})
    p = m.decode(np.random.default_rng(8).normal(
        0, 0.6, layout.length).astype(np.float32), layout).reshape(-1)
    x = inputs(2)
    for mut in (0.25, 1.0, 4.0):
        out = evaluate(gl, p, layout, x, mut=mut)
        assert np.all(np.isfinite(out)), f"{name} went non-finite at mut={mut}"


@pytest.mark.parametrize("name", ["gabor", "lenia", "mlp"])
def test_mutation_actually_changes_the_output(gl, name):
    from services.brains import REGISTRY

    m = REGISTRY[name]
    layout = m.layout_from_settings({})
    p = m.decode(np.random.default_rng(9).normal(
        0, 0.6, layout.length).astype(np.float32), layout).reshape(-1)
    x = inputs(3)
    base = evaluate(gl, p, layout, x, mut=0.0)
    mutated = evaluate(gl, p, layout, x, mut=0.3)
    assert not np.allclose(base, mutated, atol=1e-6), (
        f"{name} ignores Mutation Scale"
    )
