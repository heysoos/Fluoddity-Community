"""With no rule loaded, EVERY modality gives each cohort its own brain.

Fourier used to be the only one that did, and it did it a second way: an
all-zero buffer switched on a GPU-side generator that built FourierCenters, so
it could never have worked for Gabor, Lenia or MLP. Those three got one CPU
brain shared by every cohort - and since MUTATION_SCALE defaults to 0.0 and
num_cohorts to 64, that is 64 cohorts running the identical brain. A monoculture,
against Fourier's 64 independent rules. Same words on screen, different
behaviour underneath, and the Inspector could not describe either honestly.

The generated rules now come from the host for all four, into dedicated cohort
slots. There is no blank state left, so nothing has to ask which of two brains
it is looking at.

Needs a real GL context, so it skips where there is none (CI).
"""
import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

from services.brains import (COHORT_BRAIN_SLOT0, MAX_BRAIN_FLOATS,  # noqa: E402
                             MAX_COHORT_BRAINS, REGISTRY, generated_brains)

MODALITIES = ["fourier", "gabor", "lenia", "mlp"]


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:
        pytest.skip(f"no standalone GL context: {exc}")
    yield c


@pytest.fixture(scope="module")
def sim(ctx):
    from sim import Sim

    return Sim(ctx, world_size=1.0, canvas_aspect_ratio="1:1",
               particle_density=0.05)


def _cohort(sim, i):
    """Cohort i's brain, read straight out of the flat buffer."""
    n = sim.brain_layout.length
    off = (COHORT_BRAIN_SLOT0 + i) * MAX_BRAIN_FLOATS * 4
    return np.frombuffer(sim.multi_load_rule_buffer.read(size=n * 4, offset=off),
                         dtype=np.float32).copy()


def _fresh(sim, name, cohorts=64, seed=0.42):
    from state import SimState

    layout = REGISTRY[name].layout_from_settings({})
    st = SimState()
    st.num_cohorts = cohorts
    st.rule_seed = seed
    sim.apply_state(st)
    sim.realloc_brain_buffers(layout)
    sim.apply_rule(None)
    return layout


@pytest.mark.parametrize("name", MODALITIES)
def test_no_rule_loaded_gives_every_cohort_a_different_brain(sim, name):
    """The behaviour Fourier had and the other three did not."""
    _fresh(sim, name)
    assert sim.brain_per_cohort
    brains = [_cohort(sim, i) for i in range(8)]
    for i in range(1, len(brains)):
        assert not np.allclose(brains[0], brains[i], atol=1e-6), (
            f"{name}: cohort {i} is a copy of cohort 0 - a monoculture")


@pytest.mark.parametrize("name", MODALITIES)
def test_no_cohort_is_silent(sim, name):
    """An all-zero brain outputs zero for every input, so no force reaches the
    particle and the canvas fades. That is what switching to Gabor did before
    any of this."""
    _fresh(sim, name)
    for i in range(0, 64, 7):
        assert _cohort(sim, i).any(), f"{name}: cohort {i} is silent"


@pytest.mark.parametrize("name", MODALITIES)
def test_slot_zero_is_cohort_zero(sim, name):
    """Anything asking for "the current rule" - the Inspector included - reads
    slot 0, so it must hold a real brain rather than the blank marker."""
    _fresh(sim, name)
    assert np.allclose(sim.slot0_params, _cohort(sim, 0), atol=1e-6)


@pytest.mark.parametrize("name", MODALITIES)
def test_the_same_seed_reproduces_the_same_brains(sim, name):
    """Deterministic in rule_seed, as the GPU generator was - so a saved seed
    still reproduces a run."""
    _fresh(sim, name, seed=0.7)
    a = [_cohort(sim, i) for i in range(4)]
    _fresh(sim, name, seed=0.7)
    assert all(np.allclose(x, _cohort(sim, i), atol=1e-6)
               for i, x in enumerate(a))


@pytest.mark.parametrize("name", MODALITIES)
def test_a_new_seed_gives_new_brains(sim, name):
    """The reset key picks a new rule_seed; that has to change what runs."""
    _fresh(sim, name, seed=0.1)
    a = _cohort(sim, 0).copy()
    _fresh(sim, name, seed=0.9)
    assert not np.allclose(a, _cohort(sim, 0), atol=1e-6)


@pytest.mark.parametrize("rule", [
    pytest.param(np.zeros((10, 8), np.float32), id="legacy-10x8"),
    pytest.param(np.zeros(80, np.float32), id="flat-80"),
    pytest.param(None, id="none"),
])
def test_an_all_zero_rule_still_means_no_rule(sim, rule):
    """The codebase's "no brain" marker, and it arrives at the RIGHT width as
    well as the wrong one - the Z key, the undo history and _Default.json all
    send a zeroed (10, 8).

    It used to be harmless because the GPU answered zeros with a generated rule.
    Uploaded verbatim it is a brain that outputs zero for every input, so no
    force reaches any particle. Measured on _Default when this branch was
    missing: the brain's own p90 output fell from 0.431 to 0.034.
    """
    _fresh(sim, "fourier")
    sim.apply_rule(rule)
    assert sim.brain_per_cohort, "an all-zero rule was taken as a real brain"
    assert sim.slot0_params.any(), "the loaded brain is silent"


def test_loading_a_rule_leaves_per_cohort_mode(sim):
    """A real rule is shared by every cohort, with variety coming from the
    mutation - the pre-existing behaviour, unchanged."""
    layout = _fresh(sim, "fourier")
    assert sim.brain_per_cohort
    sim.apply_rule(REGISTRY["fourier"].random(np.random.default_rng(0), layout))
    assert not sim.brain_per_cohort


def test_the_cohort_slots_sit_past_the_config_slots(sim):
    """They share one buffer with the multi-load configs and the tournament
    tiles. An overlap would have a cohort brain overwrite a tile's genome."""
    _fresh(sim, "fourier")
    assert COHORT_BRAIN_SLOT0 >= 64
    need = (COHORT_BRAIN_SLOT0 + MAX_COHORT_BRAINS) * MAX_BRAIN_FLOATS * 4
    assert sim.multi_load_rule_buffer.size >= need


def test_the_cohort_count_is_clamped_to_the_slots(sim):
    """num_cohorts reaches 144, which is exactly MAX_COHORT_BRAINS; anything
    beyond must not write past the buffer."""
    _fresh(sim, "fourier", cohorts=MAX_COHORT_BRAINS)
    assert _cohort(sim, MAX_COHORT_BRAINS - 1).any()


def test_the_python_and_glsl_slot_constants_agree():
    """They are declared twice - once here, once in _header.glsl - and a
    disagreement would silently read another slot's brain."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "shaders" / "brains" / "_header.glsl").read_text()
    assert f"#define COHORT_BRAIN_SLOT0 {COHORT_BRAIN_SLOT0}" in src
    assert f"#define MAX_COHORT_BRAINS {MAX_COHORT_BRAINS}" in src


def test_generated_brains_is_deterministic_and_independent():
    layout = REGISTRY["lenia"].layout_from_settings({})
    a = generated_brains(layout, 0.33, 6)
    b = generated_brains(layout, 0.33, 6)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.allclose(a[0], a[1], atol=1e-6)
    assert all(x.size == layout.length for x in a)
