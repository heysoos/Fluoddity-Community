"""A masked mapping moves the cohorts it names and no others.

Source-level tests cannot see this: the wrapper can be present at every call
site and still read the wrong row, or the buffer can go up transposed.

Two things are load-bearing and both come from CLAUDE.md.

Only `entity_update` is stepped, never `update()`. The sim does not reproduce
itself run to run - particles splat additively into a shared texture, which
races - so comparing two full runs is the blind instrument that caveat names.
`entity_update` alone is one invocation per particle with no shared writes, and
against a constant canvas it IS bit-reproducible. Do not "fix" this test by
switching to `update()`.

The parameter under test is GLOBAL_FORCE_MULT, which scales the force directly.
SENSOR_GAIN would be the obvious pick and it is the wrong one: it scales what is
read off the TRAIL, and with no canvas pass the trail is all zeros, so every
setting of it produces identical motion and the test passes on a broken build.
"""
import numpy as np
import pytest

from services import cohort_audio as ca

moderngl = pytest.importorskip("moderngl")
N_COHORTS = 64


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:
        pytest.skip(f"no GL 4.3 context: {exc}")
    yield c
    c.release()


def _identity():
    rows = len(ca.COHORT_AUDIO_PARAMS)
    arr = np.empty((rows, ca.MASK_SLOTS + 1, 2), dtype=np.float32)
    arr[..., 0] = 1.0
    arr[..., 1] = 0.0
    arr[:, ca.MASK_SLOTS, 0] = -1.0e30
    arr[:, ca.MASK_SLOTS, 1] = 1.0e30
    return arr


def _positions(sim):
    """SIZE_OF_ENTITY_STRUCT, never a literal - a stale stride does not fail,
    it reads other fields AS positions. See test_entity_struct.py."""
    from sim import SIZE_OF_ENTITY_STRUCT
    raw = np.frombuffer(sim.entities.read(), dtype=np.float32)
    wide = raw.reshape(-1, SIZE_OF_ENTITY_STRUCT // 4)
    return wide[: sim.entity_count, :2].copy()


def _run(ctx, arr, steps=12):
    """The particles are reset by the shader on the frame_count == 0 step, so
    the first entity_update is the reset and the rest are the measurement."""
    from sim import Sim
    from state import SimState

    state = SimState()
    state.num_cohorts = N_COHORTS
    state.HAZARD_RATE = 0.0          # a respawn is a teleport, not a step
    sim = Sim(ctx, world_size=0.05, canvas_aspect_ratio="1:1")
    sim.apply_state(state)
    # Without this the brain buffer is all zeros, so the force is zero and
    # scaling it is still zero - the sim barely moves and the test passes on a
    # broken build. apply_rule(None) is the "no rule loaded" path that fills
    # the cohort slots, and rule_seed is fixed so two runs draw the same brains.
    sim.apply_rule(None)
    sim.reset()
    sim.set_cohort_audio(arr)
    for step in range(steps + 1):
        sim.frame_count = step
        sim.entity_update(ctx)
    ctx.finish()
    return _positions(sim)


def test_the_unmasked_half_is_left_alone(ctx):
    row = ca.COHORT_AUDIO_PARAMS.index("GLOBAL_FORCE_MULT")

    off = _identity()
    on = _identity()
    on[row, : N_COHORTS // 2, 1] = 8.0
    on[row, ca.MASK_SLOTS] = (0.0, 20.0)

    base = _run(ctx, off)
    moved = _run(ctx, on)

    # Cohorts are contiguous slices of the particle index, so the first half of
    # the buffer IS the first half of the cohorts.
    travel = np.abs(moved - base).sum(axis=1)
    half = travel.shape[0] // 2
    lit = float(travel[:half].mean())
    dark = float(travel[half:].mean())

    assert lit > dark * 4.0, (lit, dark)


def test_an_identity_array_changes_nothing(ctx):
    """Or every rig would perturb the sim merely by being switched on."""
    base = _run(ctx, None)
    same = _run(ctx, _identity())
    assert np.array_equal(base, same)
