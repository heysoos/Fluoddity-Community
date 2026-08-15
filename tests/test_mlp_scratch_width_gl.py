"""A per-layout scratch width must not change what the brain COMPUTES.

MAX_MLP_WIDTH sizes local arrays, so it is compiled in rather than uniform -
which makes the entity-update program per layout. Two things can go wrong and
neither shows up anywhere else: a shader narrower than its stack reads past the
end of an array (silently, in GLSL), and a program cache can hand a layout the
program built for a different one. The first is checked by evaluating the same
brain under its own bucket and under the widest, the second against sim.py's
cache directly.

Needs a real GL context, so it skips where there is none (CI).
"""
import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from services.brains import REGISTRY, default_layout, layout_defines  # noqa: E402
from services.brains.mlp import MAX_WIDTH, scratch_width  # noqa: E402

M = REGISTRY["mlp"]
N = 256

HEAD = """#version 430
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer InBuf  { vec4 xs[]; };
layout(std430, binding = 1) buffer OutBuf { vec4 ys[]; };
"""
MAIN = """
void main(){
    uint i = gl_GlobalInvocationID.x;
    if(i >= xs.length()) return;
    g_brain_mut = 0.0;
    g_brain_cohort = 3.0;
    ys[i] = eval_brain(0u, xs[i]);
}
"""

# One shallow, one that needs a wider bucket than the default 8, one at the
# widest a two-layer stack can reach, and one deep enough to exercise the
# ping-pong across several layers.
STACKS = [
    [[16, 0]],
    [[8, 0], [8, 0]],
    [[16, 1], [16, 0]],
    [[24, 0], [24, 2]],
    [[6, 0], [6, 1], [6, 2], [6, 0]],
]
ids = ["-".join(f"{w}a{a}" for w, a in s) for s in STACKS]


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:                       # no GPU / no driver
        pytest.skip(f"no standalone GL context: {exc}")
    yield c


def _program(ctx, width):
    from pathlib import Path

    sh = Path(__file__).resolve().parent.parent / "shaders"
    src = (HEAD
           + f"#define MAX_MLP_WIDTH {int(width)}\n"
           + (sh / "fourier4_4.glsl").read_text()
           + (sh / "brains" / "_header.glsl").read_text()
           + "".join((sh / "brains" / f"{m}.glsl").read_text()
                     for m in ("fourier", "gabor", "lenia", "mlp"))
           + (sh / "brains" / "_dispatch.glsl").read_text()
           + MAIN)
    prog = ctx.compute_shader(src)
    ctx.buffer(reserve=4096).bind_to_storage_buffer(2)
    return prog


def _evaluate(ctx, prog, params, layout, xs):
    from utilities.gl_helpers import set_brain_layout_uniforms

    flat = np.asarray(params, dtype=np.float32).reshape(-1)
    b_in = ctx.buffer(xs.astype(np.float32).tobytes())
    b_out = ctx.buffer(reserve=len(xs) * 16)
    b_par = ctx.buffer(flat.tobytes())
    b_in.bind_to_storage_buffer(0)
    b_out.bind_to_storage_buffer(1)
    b_par.bind_to_storage_buffer(4)
    for k, v in (("BRAIN_MODALITY", 3), ("BRAIN_LEN", layout.length)):
        if k in prog:
            prog[k].value = v
    set_brain_layout_uniforms(prog, layout)
    prog.run(group_x=(len(xs) + 63) // 64)
    ctx.finish()
    out = np.frombuffer(b_out.read(), dtype=np.float32).reshape(-1, 4).copy()
    for b in (b_in, b_out, b_par):
        b.release()
    return out


@pytest.mark.parametrize("layers", STACKS, ids=ids)
def test_the_bucket_computes_what_the_widest_shader_does(ctx, layers):
    layout = M.layout_from_settings({"layers": layers})
    bucket = scratch_width(layout.shape)
    params = M.random(np.random.default_rng(11), layout)
    xs = (np.random.default_rng(2).standard_normal((N, 4)) * 0.5
          ).astype(np.float32)

    narrow = _evaluate(ctx, _program(ctx, bucket), params, layout, xs)
    widest = _evaluate(ctx, _program(ctx, MAX_WIDTH), params, layout, xs)

    assert np.isfinite(narrow).all()
    # Identical arithmetic in a different-sized array, so this is exact.
    assert np.array_equal(narrow, widest), (
        f"bucket {bucket} does not agree with {MAX_WIDTH}")
    assert np.abs(narrow).max() > 1e-6, "a silent zero brain proves nothing"


# ---- sim.py's cache -------------------------------------------------------

@pytest.fixture(scope="module")
def sim(ctx):
    from sim import Sim

    return Sim(ctx, world_size=1.0, canvas_aspect_ratio="1:1",
               particle_density=0.05)


def test_layouts_needing_the_same_bucket_share_one_program(sim):
    """Compiling per layout rather than per bucket would rebuild the shader on
    every hover, which is what the borrow exists to avoid."""
    a = M.layout_from_settings({"layers": [[8, 0], [8, 0]]})
    b = M.layout_from_settings({"layers": [[5, 1], [7, 2], [8, 0]]})
    assert scratch_width(a.shape) == scratch_width(b.shape)

    sim.realloc_brain_buffers(a)
    first = sim.entity_update_program
    sim.realloc_brain_buffers(b)
    assert sim.entity_update_program is first


def test_a_wider_stack_gets_a_different_program(sim):
    narrow = M.layout_from_settings({"layers": [[8, 0], [8, 0]]})
    wide = M.layout_from_settings({"layers": [[24, 0], [24, 0]]})
    assert scratch_width(narrow.shape) < scratch_width(wide.shape)

    sim.realloc_brain_buffers(narrow)
    a = sim.entity_update_program
    sim.realloc_brain_buffers(wide)
    assert sim.entity_update_program is not a
    # ...and coming back is a cache hit, not a rebuild.
    sim.realloc_brain_buffers(narrow)
    assert sim.entity_update_program is a


def test_every_layout_needing_no_scratch_shares_one_program(sim):
    """Fourier, Gabor, Lenia and every depth-1 MLP. Without this the cache
    holds one identical program per layout the user ever touches - and a
    Fourier brain, which pays for mlp.glsl's arrays without running a line of
    it, would compile at whatever default covers the widest stack."""
    flat = [default_layout(),
            REGISTRY["gabor"].layout_from_settings({}),
            REGISTRY["lenia"].layout_from_settings({}),
            M.layout_from_settings({"layers": [[MAX_WIDTH, 0]]})]
    assert len({tuple(sorted(layout_defines(x).items())) for x in flat}) == 1

    sim.realloc_brain_buffers(flat[0])
    first = sim.entity_update_program
    for layout in flat[1:]:
        sim.realloc_brain_buffers(layout)
        assert sim.entity_update_program is first


def test_a_deep_stack_actually_drives_particles(sim, ctx):
    """The width is compiled in, so a stack wider than the old cap is the one
    case no other GPU test covers - and a brain that returns zero for every
    input compiles and runs perfectly well."""
    from state import SimState

    layout = M.layout_from_settings({"layers": [[24, 0], [24, 0]]})
    sim.realloc_brain_buffers(layout)
    st = SimState()
    st.MUTATION_SCALE = 0.0
    sim.apply_state(st)
    sim.apply_rule(M.random(np.random.default_rng(7), layout))
    sim.reset_seed = 0.0
    sim.reset()

    def positions():
        return np.frombuffer(sim.entities.read(), dtype=np.float32
                             ).reshape(-1, 12)[: sim.entity_count, 0:2].copy()

    before = positions()
    for _ in range(60):
        sim.apply_state(st)
        sim.update(ctx)
    ctx.finish()
    after = positions()
    assert np.isfinite(after).all()
    assert np.abs(after - before).max() > 1e-4, "the deep stack drives nothing"
