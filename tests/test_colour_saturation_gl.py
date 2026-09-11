"""The real entity-update program turns the brain's colour outputs into each
particle's hue and saturation.

The other colour tests evaluate the brain functions alone; only running the sim
reads what entity_update.glsl writes. An MLP's colour BIASES make both outputs
one exact constant for every particle, so the stored values are checkable.
"""
import numpy as np
import pytest

from services.brains import colour_index, get
from state.sim_state import SimState

pytestmark = pytest.mark.gpu

# What a zero saturation output must land on: the constant every particle had
# before the brain chose it.
REST_SAT = 0.8


def _mapped_sat(x):
    return 0.5 + 0.5 * np.tanh(x + np.arctanh(0.6))


def _stored_colour(hue_bias, sat_bias, sensitivity=0.25):
    """-> (hue, sat) of every active particle after a few steps."""
    moderngl = pytest.importorskip("moderngl")
    from sim import SIZE_OF_ENTITY_STRUCT, Sim

    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no GL context: {exc}")

    layout = get("mlp").layout_from_settings({})
    params = get("mlp").random(np.random.default_rng(0), layout)
    hue, sat = colour_index(layout)
    params[hue] = 0.0
    params[sat] = 0.0
    params[hue[-1]] = hue_bias
    params[sat[-1]] = sat_bias

    state = SimState()
    state.hue_sensitivity = sensitivity
    state.color_by_cohort = False
    state.MUTATION_SCALE = 0.0
    state.HAZARD_RATE = 0.0
    sim = Sim(ctx, world_size=0.02, canvas_aspect_ratio="1:1")
    sim.realloc_brain_buffers(layout)
    sim.apply_state(state)
    sim.apply_rule(params)
    sim.reset()
    for _ in range(3):
        sim.apply_state(state)
        sim.update(ctx)
    ctx.finish()
    raw = np.frombuffer(sim.entities.read(), dtype=np.float32).reshape(
        -1, SIZE_OF_ENTITY_STRUCT // 4).copy()
    ctx.release()
    # Inactive slots are parked far off the canvas.
    live = raw[np.abs(raw[:, 0]) < 100.0]
    assert len(live) > 100, "the sim must have particles for this to mean anything"
    return live[:, 5], live[:, 6]


def test_a_zero_colour_output_is_the_old_saturation():
    hue, sat = _stored_colour(0.0, 0.0)
    np.testing.assert_allclose(sat, REST_SAT, atol=1e-5)
    np.testing.assert_allclose(hue, 0.0, atol=1e-6)


@pytest.mark.parametrize("hue_bias,sat_bias", [(0.3, 2.0), (-0.7, -3.0)])
def test_the_outputs_reach_the_particle(hue_bias, sat_bias):
    hue, sat = _stored_colour(hue_bias, sat_bias, sensitivity=0.25)
    np.testing.assert_allclose(hue, 0.25 * hue_bias, atol=1e-5)
    np.testing.assert_allclose(sat, _mapped_sat(sat_bias), atol=1e-5)
