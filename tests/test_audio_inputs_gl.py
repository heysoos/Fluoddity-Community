"""The channel buffer reaches the particles through the real sim program.

Only `entity_update` is stepped, never `update()`: against a constant canvas
it is one invocation per particle with no shared writes and IS reproducible,
which is what lets a silent widened brain be compared with its deaf ancestor
bit for bit. See tests/test_cohort_audio_gl.py for why `update()` is the
blind instrument here.
"""
import numpy as np
import pytest

from services import cohort_audio as ca
from services.brains import REGISTRY
from services.brains.layout_moves import grow_inputs, transfer_audio_inputs

moderngl = pytest.importorskip("moderngl")
N_COHORTS = 64
K = 2


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:
        pytest.skip(f"no GL 4.3 context: {exc}")
    yield c
    c.release()


def _positions(sim):
    from sim import SIZE_OF_ENTITY_STRUCT
    raw = np.frombuffer(sim.entities.read(), dtype=np.float32)
    wide = raw.reshape(-1, SIZE_OF_ENTITY_STRUCT // 4)
    return wide[: sim.entity_count, :2].copy()


def _pair():
    base = REGISTRY["fourier"].layout_from_settings({})
    wide = grow_inputs(base, K)
    deaf = np.asarray(REGISTRY["fourier"].random(np.random.default_rng(3), base),
                      np.float32).reshape(-1)
    heard = transfer_audio_inputs(deaf, base, wide, np.random.default_rng(4))
    return base, deaf, wide, heard


def _run(ctx, layout, rule, audio, steps=12):
    from sim import Sim
    from state import SimState

    state = SimState()
    state.num_cohorts = N_COHORTS
    state.HAZARD_RATE = 0.0
    sim = Sim(ctx, world_size=0.05, canvas_aspect_ratio="1:1")
    sim.apply_state(state)
    sim.realloc_brain_buffers(layout)
    sim.apply_rule(rule)
    sim.reset()
    sim.set_audio_inputs(audio)
    for step in range(steps + 1):
        sim.frame_count = step
        sim.entity_update(ctx)
    ctx.finish()
    return _positions(sim)


def test_a_silent_widened_brain_moves_exactly_as_its_deaf_ancestor(ctx):
    base, deaf, wide, heard = _pair()
    ancestor = _run(ctx, base, deaf, None)
    silent = _run(ctx, wide, heard, None)
    assert np.array_equal(ancestor, silent)
    zeros = np.zeros((K, ca.MASK_SLOTS), np.float32)
    assert np.array_equal(ancestor, _run(ctx, wide, heard, zeros))


def test_a_channel_reaches_the_cohorts_it_is_written_for(ctx):
    base, deaf, wide, heard = _pair()
    arr = np.zeros((K, ca.MASK_SLOTS), np.float32)
    arr[0, : N_COHORTS // 2] = 0.6
    quiet = _run(ctx, wide, heard, None)
    loud = _run(ctx, wide, heard, arr)
    travel = np.abs(loud - quiet).sum(axis=1)
    half = travel.shape[0] // 2
    lit, dark = float(travel[:half].mean()), float(travel[half:].mean())
    assert lit > 0.0
    assert dark == 0.0, dark
