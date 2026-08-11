"""The Fourier mutation must keep the ORIGINAL mutate_rule() semantics.

These assert structural invariants rather than shader source text, because the
defect they guard was invisible to both: the CPU suite passed, the shader
compiled, click-to-adopt round-tripped, and the sim is too noisy run-to-run for
a trajectory diff to resolve anything (the same tree does not reproduce itself
- particles splat additively into a shared texture). Measured against the
pre-rewrite shader over a 4096-point input grid, a per-component frequency
jitter deviated by 56% of signal while looking entirely plausible.

Needs a real GL context, so it skips where there is none (CI).
"""
import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

N_CENTERS = 10
FLOATS = N_CENTERS * 8

SRC_HEAD = """#version 430
layout(local_size_x = 1) in;
layout(std430, binding = 1) buffer OutBuf { float ys[]; };
uniform float MUT;
uniform float COHORT;
"""

SRC_MAIN = """
void main(){
    g_brain_mut = MUT;
    g_brain_cohort = COHORT;
    for(int i = 0; i < BRAIN_LEN; i++) ys[i] = fourier_param_at(0u, i);
}
"""


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:                       # no GPU / no driver
        pytest.skip(f"no standalone GL context: {exc}")
    yield c


@pytest.fixture(scope="module")
def prog(ctx):
    from pathlib import Path

    sh = Path(__file__).resolve().parent.parent / "shaders"

    def rd(p):
        return (sh / p).read_text()

    src = (SRC_HEAD + rd("fourier4_4.glsl") + rd("brains/_header.glsl")
           + "".join(rd(f"brains/{m}.glsl")
                     for m in ("fourier", "gabor", "lenia", "mlp"))
           + rd("brains/_dispatch.glsl") + SRC_MAIN)
    return ctx.compute_shader(src)


def mutated(ctx, prog, params, mut, cohort=3.0):
    """Run fourier_param_at over the whole brain and return it as (n, 8)."""
    flat = np.asarray(params, dtype=np.float32).reshape(-1)
    b_par = ctx.buffer(flat.tobytes())
    b_out = ctx.buffer(reserve=FLOATS * 4)
    b_par.bind_to_storage_buffer(4)
    b_out.bind_to_storage_buffer(1)
    for name, val in (("MUT", mut), ("COHORT", cohort),
                      ("BRAIN_MODALITY", 0), ("BRAIN_LEN", FLOATS),
                      ("BRAIN_SHAPE", (N_CENTERS, 0, 0, 0))):
        if name in prog:
            prog[name].value = val
    prog.run()
    ctx.finish()
    out = np.frombuffer(b_out.read(), dtype=np.float32).reshape(N_CENTERS, 8)
    b_par.release()
    b_out.release()
    return out


@pytest.fixture(scope="module")
def genome():
    from services.genome import random_genome

    return np.asarray(random_genome(np.random.default_rng(42)),
                      dtype=np.float32).reshape(N_CENTERS, 8)


def test_zero_mutation_is_the_identity(ctx, prog, genome):
    got = mutated(ctx, prog, genome, 0.0)
    assert np.allclose(got, genome, atol=1e-6)


def test_one_scalar_scales_a_whole_frequency_vector(ctx, prog, genome):
    """The defect this exists for. The original multiplies all four components
    of centers[i].frequency by ONE scalar, so the frequency VECTOR keeps its
    direction and only changes length. Jittering each component independently
    rotates it - a different function, and the one that shipped.
    """
    got = mutated(ctx, prog, genome, 0.25)
    ratio = got[:, :4] / genome[:, :4]
    spread = ratio.max(axis=1) - ratio.min(axis=1)
    assert np.all(spread < 1e-5), (
        "frequency components of a centre were scaled by DIFFERENT factors "
        f"(max spread {spread.max():.3e}); the frequency vector was rotated, "
        "not scaled"
    )


def test_the_frequency_scale_stays_in_the_legacy_band(ctx, prog, genome):
    """`freq *= 1 + amount*0.5*(hash-0.5)` with hash in [0,1] is 1 +/- 0.25*amount."""
    amount = 0.25
    got = mutated(ctx, prog, genome, amount)
    ratio = (got[:, :4] / genome[:, :4]).mean(axis=1)
    assert np.all(np.abs(ratio - 1.0) <= 0.25 * amount + 1e-6), (
        f"frequency scale out of the legacy band: {ratio}"
    )


def test_amplitudes_offset_within_the_requested_amount(ctx, prog, genome):
    amount = 0.25
    got = mutated(ctx, prog, genome, amount)
    delta = got[:, 4:] - genome[:, 4:]
    assert np.abs(delta).max() <= amount + 1e-6, (
        f"amplitude offset exceeded the mutation amount: {np.abs(delta).max()}"
    )
    # Four INDEPENDENT values per centre (the original takes a vec4 from one
    # hash4). All four equal would mean a single scalar was broadcast.
    assert np.all(delta.std(axis=1) > 1e-4), "amplitude jitter is not per-component"


def test_the_mutation_seed_depends_on_rule_CONTENT(ctx, prog, genome):
    """The original hashed centers[4].frequency.xy + centers[7].amplitude.yx +
    centers[1].frequency.zw. Seeding from the cohort alone makes two different
    rules receive the SAME jitter, so a cohort stops being an independent probe
    of the neighbourhood.
    """
    other = genome.copy()
    other[4, 0] += 0.5                       # one of the six hashed floats
    a = mutated(ctx, prog, genome, 0.25) - genome
    b = mutated(ctx, prog, other, 0.25) - other
    assert not np.allclose(a, b, atol=1e-6), (
        "identical jitter for two different rules: the seed ignores content"
    )


def test_mutation_is_deterministic_in_the_cohort(ctx, prog, genome):
    a = mutated(ctx, prog, genome, 0.25, cohort=3.0)
    b = mutated(ctx, prog, genome, 0.25, cohort=3.0)
    c = mutated(ctx, prog, genome, 0.25, cohort=4.0)
    assert np.array_equal(a, b), "same cohort gave a different mutation"
    assert not np.allclose(a, c, atol=1e-6), "different cohorts share a mutation"
