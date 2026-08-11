"""Click-to-adopt must read back the brain the particle is ACTUALLY running.

Needs a real GL context, so it skips where there is none (CI). It exists
because the whole CPU-side suite passed while adoption copied zeros: the bug
lived entirely in what the compute shader chose to write.
"""
import numpy as np
import pytest

moderngl = pytest.importorskip("moderngl")


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:                      # no GPU / no driver
        pytest.skip(f"no standalone GL context: {exc}")
    yield c


@pytest.fixture(scope="module")
def sim(ctx):
    from sim import Sim

    return Sim(ctx, world_size=1.0, canvas_aspect_ratio="1:1",
               particle_density=0.01)


def _adopt(sim, ctx, entity_id=0):
    from state import SimState
    from utilities.gl_helpers import readback_rule

    st = SimState()
    st.MUTATION_SCALE = 0.0
    sim.request_rule_buffer_update(entity_id)
    sim.apply_state(st)
    sim.entity_update(ctx)
    return readback_rule(sim.get_rule_buffer(), entity_id, sim.brain_layout)


def test_adopting_a_loaded_brain_returns_it(sim, ctx):
    from services.genome import random_genome

    genome = random_genome(np.random.default_rng(0))
    sim.apply_rule(genome)
    assert np.allclose(_adopt(sim, ctx), genome, atol=1e-5)


def test_adopting_a_generated_cohort_rule_does_not_return_zeros(sim, ctx):
    """With no brain loaded the particle runs a generated per-cohort rule. The
    readback must be THAT rule, mutation included.

    Returning zeros re-marks the rule as "none" on apply, which redraws every
    cohort - most sluggish, a few lively. That was the reported symptom.
    """
    sim.apply_rule(None)
    rule = _adopt(sim, ctx)
    assert not np.all(rule == 0.0), (
        "adoption returned zeros instead of the rule the particle was running"
    )
    assert np.all(np.isfinite(rule))
    assert np.abs(rule[:, :4]).max() <= 3.0 + 1e-4, "frequency out of range"
    assert np.abs(rule[:, 4:]).max() <= 1.0 + 1e-4, "amplitude out of range"


def test_adopting_a_generated_rule_is_stable_under_reapply(sim, ctx):
    """Adopt, apply, adopt again -> the same rule. If the first adoption
    returned zeros this oscillates between "no rule" and a fresh draw
    forever."""
    sim.apply_rule(None)
    first = _adopt(sim, ctx)
    sim.apply_rule(first)
    second = _adopt(sim, ctx)
    assert np.allclose(first, second, atol=1e-5)


def test_only_the_adopted_particle_is_written(sim, ctx):
    """The writeback is scoped to the entity being read back.

    Writing all 600k means every particle re-derives its mutation to produce
    bytes nobody reads - 13 ms a click, against ~2 ms before brains carried
    their mutation on read. Adoption itself must still be exact, which the
    tests above cover; this pins the scope.
    """
    import numpy as np
    from services.genome import random_genome

    n = sim.brain_layout.length
    sim.apply_rule(random_genome(np.random.default_rng(3)))
    sim.get_rule_buffer().clear()
    _adopt(sim, ctx, entity_id=7)

    raw = np.frombuffer(sim.get_rule_buffer().read(), dtype=np.float32)
    written = raw.reshape(-1, n).any(axis=1).nonzero()[0]
    assert written.tolist() == [7], (
        f"expected only entity 7 to be written, got {len(written)} entities"
    )


def test_the_brain_buffer_starts_zeroed(ctx):
    """ctx.buffer(reserve=) does not zero memory, and every slot is read as a
    brain whether anything has written it or not. Measured: a bare reserve left
    13 nonzero floats here."""
    from sim import Sim

    s = Sim(ctx, world_size=1.0, canvas_aspect_ratio="1:1", particle_density=0.01)
    raw = np.frombuffer(s.multi_load_rule_buffer.read(), dtype=np.float32)
    assert np.all(raw == 0.0), (
        f"{int(np.count_nonzero(raw))} uninitialised floats in the brain buffer"
    )
