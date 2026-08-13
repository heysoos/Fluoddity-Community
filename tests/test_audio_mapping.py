"""The modulation maths, and which parameters may be modulated at all."""
import pytest

from services.audio_mapping import (MODES, Mapping, TargetDef, brain_targets,
                                    deaf_targets, modulate, physics_targets)
from services.audio_shapers import ShaperParams, ShaperState
from state.sim_state import SimState

DT = 1 / 60


def one(target="SENSOR_GAIN", **kw):
    kw.setdefault("signal", "bass")
    kw.setdefault("shaper", ShaperParams())
    return Mapping(target=target, **kw)


def gain_target(lo=0.0, hi=10.0, hard_lo=None, hard_hi=None):
    return TargetDef("SENSOR_GAIN", "Sensor Gain", "physics", lo, hi,
                     hard_lo, hard_hi)


def run(mappings, targets, signals, bases, **kw):
    states = {id(m): ShaperState() for m in mappings}
    return modulate(bases, targets, mappings, signals, states,
                    kw.get("strengths", {}), kw.get("global_strength", 1.0),
                    DT, kw.get("deaf", set()))


# --- targets -----------------------------------------------------------------

def test_modes_are_the_three_boids_had():
    assert MODES == ("add", "subtract", "multiply")


def test_physics_targets_come_from_the_registry_not_a_second_list():
    from ui.physics_params import PHYSICS_PARAMS
    keys = {t.key for t in physics_targets(SimState())}
    registry = {p.name for p in PHYSICS_PARAMS if not p.off_at_max}
    assert keys == registry


def test_a_parameter_that_switches_off_at_max_is_never_a_target():
    """V Max's top of track means Off; modulating it would make the readout lie."""
    assert "V_MAX" not in {t.key for t in physics_targets(SimState())}


def test_a_target_carries_the_users_current_slider_range():
    st = SimState()
    st.slider_ranges["Sensor Gain"] = [1.0, 4.0, 0.0, 10.0]
    t = next(t for t in physics_targets(st) if t.key == "SENSOR_GAIN")
    assert (t.lo, t.hi) == (1.0, 4.0)


def test_brain_targets_are_the_float_settings_only():
    from services import brains
    m = brains.get("fourier")
    layout = m.layout_from_settings({})
    keys = [t.key for t in brain_targets(m, layout)]
    assert keys == ["freq_scale", "low_freq_bias"]


def test_structural_brain_settings_are_never_targets():
    from services import brains
    for name in ("fourier", "gabor", "lenia", "mlp"):
        m = brains.get(name)
        layout = m.layout_from_settings({})
        schema = {s.key: s for s in m.settings_schema()}
        for t in brain_targets(m, layout):
            assert schema[t.key].kind not in brains.STRUCTURAL_KINDS


def test_a_modality_with_no_float_settings_offers_no_rows():
    """Structural settings change the parameter count, so modulating one would
    reshape the brain buffer every frame. A modality declaring only those has
    nothing to offer - stubbed, because every shipped modality now has scales.
    """
    from services.brains import Setting

    class _Structural:
        def settings_schema(self):
            return [Setting("layers", "Layers", "layers", 1, 48, 16)]

    assert brain_targets(_Structural(), None) == []


def test_every_shipped_modality_offers_at_least_one_row():
    """A modality with no continuous scale cannot be audio-modulated at all,
    which reads as the panel being broken rather than as a property of that
    brain."""
    from services import brains
    for name, m in brains.REGISTRY.items():
        rows = brain_targets(m, m.layout_from_settings({}))
        assert rows, f"{name} offers nothing to modulate"


# --- deafness ----------------------------------------------------------------

def test_a_swept_parameter_is_reported_deaf():
    st = SimState()
    st.x_sweeps["SENSOR_GAIN"] = 1.0
    assert "SENSOR_GAIN" in deaf_targets(st)


def test_every_sweep_axis_counts():
    for axis in ("x_sweeps", "y_sweeps", "cohort_sweeps"):
        st = SimState()
        getattr(st, axis)["DRAG"] = -1.0
        assert "DRAG" in deaf_targets(st)


def test_nothing_is_deaf_by_default():
    assert deaf_targets(SimState()) == set()


def test_a_deaf_target_is_left_out_so_the_caller_keeps_the_base():
    """Absent, not present-at-base: the caller only copies what moved, and a
    key it never writes keeps whatever the user's slider says."""
    out = run([one()], [gain_target()], {"bass": 1.0}, {"SENSOR_GAIN": 0.116},
              deaf={"SENSOR_GAIN"})
    assert "SENSOR_GAIN" not in out


# --- the maths ---------------------------------------------------------------

def test_add_moves_by_depth_times_the_range():
    out = run([one(mode="add", depth=0.5)], [gain_target(0.0, 10.0)],
              {"bass": 1.0}, {"SENSOR_GAIN": 1.0})
    assert out["SENSOR_GAIN"] == pytest.approx(6.0)


def test_subtract_moves_the_other_way():
    out = run([one(mode="subtract", depth=0.2)], [gain_target(0.0, 10.0)],
              {"bass": 1.0}, {"SENSOR_GAIN": 5.0})
    assert out["SENSOR_GAIN"] == pytest.approx(3.0)


def test_multiply_scales_by_one_plus_depth():
    out = run([one(mode="multiply", depth=0.5)], [gain_target(0.0, 10.0)],
              {"bass": 1.0}, {"SENSOR_GAIN": 4.0})
    assert out["SENSOR_GAIN"] == pytest.approx(6.0)


def test_additive_resolves_before_multiplicative():
    """add then multiply: (1 + 1) * 1.5 = 3, not 1 + 1*1.5 = 2.5."""
    ms = [one(mode="add", depth=0.1), one(signal="mid", mode="multiply", depth=0.5)]
    out = run(ms, [gain_target(0.0, 10.0)], {"bass": 1.0, "mid": 1.0},
              {"SENSOR_GAIN": 1.0})
    assert out["SENSOR_GAIN"] == pytest.approx(3.0)


def test_gain_scales_the_signal_before_depth():
    out = run([one(mode="add", depth=0.5, gain=2.0)], [gain_target(0.0, 10.0)],
              {"bass": 0.25}, {"SENSOR_GAIN": 0.0})
    assert out["SENSOR_GAIN"] == pytest.approx(2.5)


def test_a_signal_amplified_past_one_is_clamped_before_use():
    a = run([one(mode="add", depth=1.0, gain=4.0)], [gain_target(0.0, 10.0)],
            {"bass": 1.0}, {"SENSOR_GAIN": 0.0})
    b = run([one(mode="add", depth=1.0, gain=1.0)], [gain_target(0.0, 10.0)],
            {"bass": 1.0}, {"SENSOR_GAIN": 0.0})
    assert a["SENSOR_GAIN"] == pytest.approx(b["SENSOR_GAIN"])


def test_strength_scales_the_whole_delta_not_one_term():
    ms = [one(mode="add", depth=0.2), one(signal="mid", mode="add", depth=0.2)]
    full = run(ms, [gain_target(0.0, 10.0)], {"bass": 1.0, "mid": 1.0},
               {"SENSOR_GAIN": 0.0})
    half = run(ms, [gain_target(0.0, 10.0)], {"bass": 1.0, "mid": 1.0},
               {"SENSOR_GAIN": 0.0}, global_strength=0.5)
    assert half["SENSOR_GAIN"] == pytest.approx(full["SENSOR_GAIN"] / 2)


def test_per_target_and_global_strength_multiply():
    out = run([one(mode="add", depth=1.0)], [gain_target(0.0, 10.0)],
              {"bass": 1.0}, {"SENSOR_GAIN": 0.0},
              strengths={"SENSOR_GAIN": 0.5}, global_strength=0.5)
    assert out["SENSOR_GAIN"] == pytest.approx(2.5)


def test_zero_strength_returns_the_base_exactly():
    out = run([one(mode="add", depth=1.0)], [gain_target(0.0, 10.0)],
              {"bass": 1.0}, {"SENSOR_GAIN": 0.3}, global_strength=0.0)
    assert out["SENSOR_GAIN"] == pytest.approx(0.3)


def test_a_disabled_mapping_contributes_nothing():
    out = run([one(mode="add", depth=1.0, enabled=False)],
              [gain_target(0.0, 10.0)], {"bass": 1.0}, {"SENSOR_GAIN": 0.3})
    assert "SENSOR_GAIN" not in out


def test_an_unknown_signal_name_contributes_nothing():
    out = run([one(signal="nope", mode="add", depth=1.0)],
              [gain_target(0.0, 10.0)], {"bass": 1.0}, {"SENSOR_GAIN": 0.3})
    assert "SENSOR_GAIN" not in out


def test_a_mapping_for_an_unknown_target_is_ignored():
    """It names no known target, so nothing is modulated and nothing invented."""
    out = run([one(target="NOT_A_PARAM", mode="add", depth=1.0)],
              [gain_target()], {"bass": 1.0}, {"SENSOR_GAIN": 0.3})
    assert out == {}


# --- clamping ----------------------------------------------------------------

def test_the_slider_range_clamps_when_there_is_no_hard_limit():
    out = run([one(mode="add", depth=1.0)], [gain_target(0.0, 10.0)],
              {"bass": 1.0}, {"SENSOR_GAIN": 8.0})
    assert out["SENSOR_GAIN"] == pytest.approx(10.0)


def test_a_hard_limit_wins_over_the_slider_range():
    t = gain_target(0.0, 10.0, hard_lo=0.0, hard_hi=2.0)
    out = run([one(mode="add", depth=1.0)], [t], {"bass": 1.0},
              {"SENSOR_GAIN": 1.0})
    assert out["SENSOR_GAIN"] == pytest.approx(2.0)


def test_a_bipolar_target_clamps_at_its_lower_rail():
    t = TargetDef("DRAG", "Drag", "physics", -1.0, 1.0, -1.0, 1.0)
    out = run([one(target="DRAG", mode="subtract", depth=1.0)], [t],
              {"bass": 1.0}, {"DRAG": 0.5})
    assert out["DRAG"] == pytest.approx(-1.0)


# --- the rule the whole design rests on --------------------------------------

def test_the_bases_dict_is_never_mutated():
    bases = {"SENSOR_GAIN": 0.116}
    run([one(mode="add", depth=1.0)], [gain_target()], {"bass": 1.0}, bases)
    assert bases == {"SENSOR_GAIN": 0.116}


def test_an_unbound_target_is_absent_from_the_result():
    """Only modulated targets are returned, so the caller copies nothing else."""
    targets = [gain_target(), TargetDef("DRAG", "Drag", "physics", -1, 1, None, None)]
    out = run([one()], targets, {"bass": 0.5}, {"SENSOR_GAIN": 1.0, "DRAG": 0.5})
    assert set(out) == {"SENSOR_GAIN"}
