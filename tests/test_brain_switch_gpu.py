"""Switching modality must produce a sim that actually runs.

Every other test checks one half: the Python layout, the GLSL formula, the
archive path. None of them would notice that a switched-to modality drives no
particles - a brain that returns 0 for every input compiles, decodes, round
trips and archives perfectly well, and shows a frozen canvas.

Needs a real GL context, so it skips where there is none (CI).
"""
import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")

MODALITIES = ["fourier", "gabor", "lenia", "mlp"]


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:                       # no GPU / no driver
        pytest.skip(f"no standalone GL context: {exc}")
    yield c


@pytest.fixture(scope="module")
def sim(ctx):
    from sim import Sim

    return Sim(ctx, world_size=1.0, canvas_aspect_ratio="1:1",
               particle_density=0.05)


def run(sim, ctx, name, steps=60, mut=0.0):
    """Switch to `name`, seed a random brain of that layout, and step."""
    from services.brains import REGISTRY
    from state import SimState

    m = REGISTRY[name]
    layout = m.layout_from_settings({})
    sim.realloc_brain_buffers(layout)

    st = SimState()
    st.MUTATION_SCALE = mut
    sim.apply_state(st)
    sim.apply_rule(m.random(np.random.default_rng(4), layout))
    sim.reset_seed = 0.0
    sim.reset()

    before = np.frombuffer(sim.entities.read(), dtype=np.float32
                           ).reshape(-1, 12)[: sim.entity_count, 0:2].copy()
    for _ in range(steps):
        sim.apply_state(st)
        sim.update(ctx)
    after = np.frombuffer(sim.entities.read(), dtype=np.float32
                          ).reshape(-1, 12)[: sim.entity_count, 0:2].copy()
    return before, after


@pytest.mark.parametrize("name", MODALITIES)
def test_every_modality_drives_the_particles(sim, ctx, name):
    before, after = run(sim, ctx, name)
    assert np.all(np.isfinite(after)), f"{name} produced non-finite positions"
    moved = np.linalg.norm(after - before, axis=1)
    assert float(moved.mean()) > 1e-4, (
        f"{name} left the particles where they started - the brain is silent"
    )


@pytest.mark.parametrize("name", MODALITIES)
def test_every_modality_survives_mutation(sim, ctx, name):
    _before, after = run(sim, ctx, name, mut=0.5)
    assert np.all(np.isfinite(after)), f"{name} went non-finite under mutation"


def test_switching_back_and_forth_keeps_working(sim, ctx):
    """The buffer is reallocated on each switch and rebound to binding 2; a
    stale binding shows up as a frozen or garbage sim on the second switch."""
    for name in ["fourier", "mlp", "lenia", "fourier", "gabor"]:
        before, after = run(sim, ctx, name, steps=30)
        assert np.all(np.isfinite(after)), name
        assert float(np.linalg.norm(after - before, axis=1).mean()) > 1e-4, name


@pytest.mark.parametrize("name", MODALITIES)
def test_a_switch_leaves_a_brain_that_actually_RUNS(sim, ctx, name):
    """The state the switch really produces, not the one the tests wanted.

    _apply_brain_layout drops the old genome, because its floats mean something
    else under a new layout, and what replaces it must RUN. An all-zero brain is
    not an error, it is silence: a Gabor filter with amplitude 0 and envelope
    width 1e-3 returns 0 for every input, so every particle stops and the canvas
    fades to black. Reported as "changed to gabor and particles disappeared".
    """
    from services.brains import REGISTRY
    from state import SimState

    layout = REGISTRY[name].layout_from_settings({})
    sim.realloc_brain_buffers(layout)
    st = SimState()
    st.MUTATION_SCALE = 0.0
    st.rule_seed = 0.5
    sim.apply_state(st)
    sim.apply_rule(None)                    # exactly what the switch does
    sim.reset_seed = 0.0
    sim.reset()

    before = np.frombuffer(sim.entities.read(), dtype=np.float32
                           ).reshape(-1, 12)[: sim.entity_count, 0:2].copy()
    for _ in range(60):
        sim.apply_state(st)
        sim.update(ctx)
    after = np.frombuffer(sim.entities.read(), dtype=np.float32
                          ).reshape(-1, 12)[: sim.entity_count, 0:2].copy()

    assert np.all(np.isfinite(after)), name
    assert float(np.linalg.norm(after - before, axis=1).mean()) > 1e-4, (
        f"{name}: a fresh switch leaves the particles frozen"
    )


@pytest.mark.parametrize("name", MODALITIES)
def test_a_rule_of_the_wrong_width_does_not_crash(sim, ctx, name):
    """Presets, the undo history and the Z key all carry (10, 8) Fourier
    genomes. Handing one to a 168-float layout raised inside pack_brains and
    took the whole app down."""
    from services.brains import REGISTRY

    layout = REGISTRY[name].layout_from_settings({})
    sim.realloc_brain_buffers(layout)
    sim.apply_rule(np.zeros((10, 8), dtype=np.float32))     # must not raise
    sim.apply_rule(np.zeros(512, dtype=np.float32))         # nor this


def test_the_readback_buffer_is_resized_by_the_switch(sim, ctx):
    """It is ONE brain of BRAIN_LEN floats, and must not scale with the
    particle count. Left at the old width, a wider layout reads past the end of
    it on adopt."""
    from services.brains import REGISTRY

    for name in MODALITIES:
        layout = REGISTRY[name].layout_from_settings({})
        sim.realloc_brain_buffers(layout)
        assert sim.get_rule_buffer().size == layout.length * 4, name


def test_adoption_returns_the_right_width_after_a_switch(sim, ctx):
    from services.brains import REGISTRY
    from state import SimState
    from utilities.gl_helpers import readback_rule

    for name in MODALITIES:
        m = REGISTRY[name]
        layout = m.layout_from_settings({})
        sim.realloc_brain_buffers(layout)
        st = SimState()
        st.MUTATION_SCALE = 0.0
        want = np.asarray(m.random(np.random.default_rng(11), layout),
                          dtype=np.float32).reshape(-1)
        sim.apply_rule(want)
        sim.request_rule_buffer_update(0)
        sim.apply_state(st)
        sim.entity_update(ctx)
        got = readback_rule(sim.get_rule_buffer(), layout).reshape(-1)
        assert got.shape == want.shape, name
        assert np.allclose(got, want, atol=1e-5), (
            f"{name}: adopted brain differs from the one that was loaded"
        )
