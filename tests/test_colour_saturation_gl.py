"""The real entity-update program turns the brain's colour outputs into each
particle's hue and saturation.

The other colour tests evaluate the brain functions alone; only running the sim
reads what entity_update.glsl writes. An MLP's colour BIASES make both outputs
one exact constant for every particle, so the stored values are checkable.
"""
import numpy as np
import pytest

from services.brains import colour_index, get
from services.brains.mlp import layer_spans
from state.sim_state import SimState

pytestmark = pytest.mark.gpu

# What a zero saturation output must land on: the constant every particle had
# before the brain chose it.
REST_SAT = 0.8


def _mapped_sat(x):
    return 0.5 + 0.5 * np.tanh(x + np.arctanh(0.6))


def _stored_colour(hue_bias, sat_bias, sensitivity=0.25, legacy=False,
                   force_bias=None):
    """-> (hue, sat) of every active particle after a few steps.

    `force_bias` silences every weight and leaves one output BIAS standing, so
    the brain's axial force is that constant for every particle - which is what
    legacy colouring reads its hue off.
    """
    moderngl = pytest.importorskip("moderngl")
    from sim import SIZE_OF_ENTITY_STRUCT, Sim

    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"no GL context: {exc}")

    layout = get("mlp").layout_from_settings({})
    params = get("mlp").random(np.random.default_rng(0), layout)
    if force_bias is not None:
        _hidden, _out_w, out_b, _n = layer_spans(layout.shape,
                                                 layout.audio_inputs)
        params[:] = 0.0
        params[out_b] = force_bias
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
    state.DISABLE_SYMMETRY = False
    state.legacy_color = legacy
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


# Legacy colouring: the hue every particle had before the brains grew colour
# outputs, read off the force terms, at one flat saturation. The mirror call
# doubles a constant brain's axial output, which is why the factor is two.
@pytest.mark.parametrize("hue_bias,sat_bias", [(0.0, 0.0), (0.3, 2.0)])
def test_legacy_colouring_reads_the_force_term_and_ignores_the_outputs(
        hue_bias, sat_bias):
    hue, sat = _stored_colour(hue_bias, sat_bias, sensitivity=0.25,
                              legacy=True, force_bias=0.4)
    np.testing.assert_allclose(hue, 0.25 * 2.0 * 0.4, atol=1e-5)
    np.testing.assert_allclose(sat, REST_SAT, atol=1e-6)


def test_the_outputs_come_back_when_legacy_colouring_is_off():
    """The same brain, the same everything, one checkbox apart."""
    hue, sat = _stored_colour(0.3, 2.0, sensitivity=0.25, force_bias=0.4)
    np.testing.assert_allclose(hue, 0.25 * 0.3, atol=1e-5)
    np.testing.assert_allclose(sat, _mapped_sat(2.0), atol=1e-5)
