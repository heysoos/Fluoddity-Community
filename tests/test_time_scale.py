"""The sim's clock, and the promise that 1.0 changes nothing.

TIME_SCALE says how much simulated time one physics step covers. Every preset
ever saved was made at 1.0, so at 1.0 every arithmetic path it touches has to
be untouched - which is what makes the rest of it safe to reason about.

Compensation is the point. Scaling the position alone gives a particle that
turns as sharply per step while covering less ground, and a trail that takes
in a full step's deposit for half a step's travel: a different creature, drawn
thicker, not a slower one. Two moving averages carry the whole correction -
the momentum filter in entity_update.glsl and the trail in canvas.frag.
"""
import pathlib

import pytest

from state import SimState
from ui.physics_params import PARAM_BY_NAME, PHYSICS_PARAM_NAMES

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENTITY = (ROOT / "shaders" / "entity_update.glsl").read_text(encoding="utf-8")
CANVAS = (ROOT / "shaders" / "canvas.frag").read_text(encoding="utf-8")


def test_the_default_is_one():
    """A preset saved before the clock existed must run exactly as it did."""
    assert SimState().TIME_SCALE == 1.0


def test_both_shaders_declare_it():
    """The particles and their trail must keep ONE clock - the trail is a
    fragment pass and the particles a compute pass, and a creature whose
    halves disagree about the step is not the same creature slower."""
    assert "uniform float TIME_SCALE" in ENTITY
    assert "uniform float TIME_SCALE" in CANVAS


def test_the_python_side_pushes_it_to_both_programs():
    sim_src = (ROOT / "sim.py").read_text(encoding="utf-8")
    assert "tryset(self.entity_update_program, 'TIME_SCALE'" in sim_src
    assert "tryset(self.canvas_update_program, 'TIME_SCALE'" in sim_src


def test_the_momentum_filter_keeps_its_steady_state():
    """`vel = vel*d + f` settles at f/(1-d). Running it at ts means retaining
    d^ts, and the force share has to follow or the particle settles somewhere
    else entirely - which is a different creature, not a slower one."""
    assert "pow(abs(drag), TIME_SCALE)" in ENTITY
    assert "(1.0 - drag_ts)/(1.0 - drag)" in ENTITY


def test_the_trail_is_compensated_by_one_pow():
    """The decay and the deposit are two halves of one moving average - what
    the trail keeps is P and what it takes in is 1-P - so raising P to the
    step's length corrects both at once."""
    assert "pow(trail_persistence, TIME_SCALE)" in CANVAS


def test_the_step_is_scaled_after_the_speed_cap():
    assert "e.pos += step_delta*TIME_SCALE;" in ENTITY


def test_the_blur_runs_on_the_same_clock_as_the_decay():
    """The trail's decay went on the clock and its diffusion did not, so a
    slower sim drew the same creature smeared - the blur is one step of an
    explicit heat solve, and halving the clock ran twice as many of them per
    unit of simulated time. What the spread actually does is measured by
    tests/test_time_scale_diffusion_gl.py, which runs the shader; this only
    pins the wiring."""
    assert "float time_scaled_K(float K)" in CANVAS
    assert "getBlur(texcoord, can_tex,time_scaled_K(TRAIL_DIFFUSION))" in CANVAS
    assert "if (TIME_SCALE == 1.0) { return K; }" in CANVAS


def test_every_pow_is_guarded_so_that_one_is_bit_identical():
    """`pow(x, 1.0)` is NOT required to return x - GLSL does not promise a
    correctly rounded pow - so an unguarded one would perturb every preset in
    the library by a bit or two the moment this shipped. Multiplying by
    exactly 1.0 is safe and needs no guard; raising to it is not.
    """
    assert "TIME_SCALE == 1.0 ? drag :" in ENTITY
    assert "if (TIME_SCALE != 1.0) {" in CANVAS
    canvas_pow = CANVAS.index("pow(trail_persistence, TIME_SCALE)")
    guard = CANVAS.index("if (TIME_SCALE != 1.0) {")
    assert guard < canvas_pow


# ---- the exact arithmetic, in Python, against the shader's own formulae ----

def _momentum(drag, ts, steps, force=1.0):
    """The shader's filter, run in Python."""
    d_ts = abs(drag) ** ts * (1.0 if drag >= 0 else -1.0)
    gain = ts if abs(1.0 - drag) < 1e-6 else (1.0 - d_ts) / (1.0 - drag)
    v = 0.0
    for _ in range(steps):
        v = v * d_ts + force * gain
    return v


def _trail(persistence, ts, steps, deposit=1.0):
    p_ts = persistence ** ts
    c = 0.0
    for _ in range(steps):
        c = c * p_ts + (1.0 - p_ts) * deposit
    return c


@pytest.mark.parametrize("drag", (0.0, 0.25, 0.504, 0.9, 0.99))
@pytest.mark.parametrize("ts", (0.25, 0.5, 2.0))
def test_half_the_clock_reaches_the_same_velocity_in_twice_the_steps(drag, ts):
    at_one = _momentum(drag, 1.0, 400)
    scaled = _momentum(drag, ts, int(400 / ts))
    assert scaled == pytest.approx(at_one, rel=1e-6)


@pytest.mark.parametrize("p", (0.5, 0.938, 0.99))
@pytest.mark.parametrize("ts", (0.25, 0.5, 2.0))
def test_the_trail_settles_the_same_however_fast_the_clock_runs(p, ts):
    """Not just the endpoint: two steps at half time land exactly where one
    step at full time did, which is what stops a slower sim drawing thicker."""
    assert _trail(p, ts, 2, 1.0) == pytest.approx(_trail(p, 2 * ts, 1, 1.0),
                                                  rel=1e-9)


@pytest.mark.parametrize("drag", (0.0, 0.504, 0.99))
def test_a_clock_of_one_is_the_identity(drag):
    d_ts = abs(drag) ** 1.0
    gain = 1.0 if abs(1.0 - drag) < 1e-6 else (1.0 - d_ts) / (1.0 - drag)
    assert d_ts == pytest.approx(drag)
    assert gain == pytest.approx(1.0)


# ---- how it is declared --------------------------------------------------

def test_it_is_in_the_one_registry_and_not_a_second_list():
    """A second list of modulatable parameters is what
    tests/test_audio_mapping.py forbids, so the clock is a table entry like
    everything else - it just declares that it is a plain uniform."""
    p = PARAM_BY_NAME["TIME_SCALE"]
    assert p.plain_uniform is True
    assert p.is_log_scaled is True


def test_it_seeds_no_sweep_or_jitter_entry():
    """One float for the whole canvas: a sweep would offer a control the
    shader never reads, which is the recurring defect in this codebase."""
    assert "TIME_SCALE" not in PHYSICS_PARAM_NAMES
    assert "TIME_SCALE" not in SimState().x_sweeps
    assert "TIME_SCALE" not in SimState().jitters


def test_audio_may_drive_it():
    from services.audio_mapping import physics_targets

    assert "TIME_SCALE" in {t.key for t in physics_targets(SimState())}


def test_it_is_undoable():
    from state import sim_state

    assert "TIME_SCALE" in sim_state.UNDOABLE_FIELDS


def test_a_preset_does_not_carry_it():
    """The clock is how fast you are watching, not what you are watching, so
    loading someone else's creature must not reset your tempo."""
    from services.config_saver import PhysicsConfig

    assert "time_scale" not in PhysicsConfig.__dataclass_fields__
