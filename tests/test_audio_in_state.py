"""What a rig remembers, and what it must refuse to remember."""
import pytest

from services.audio_mapping import Mapping
from services.audio_shapers import ShaperParams
from state.audio_in_state import (PERSISTED_FIELDS, AudioInState, apply_dict,
                                  to_dict)


def test_it_hangs_off_ui_state():
    from state import UIState
    assert isinstance(UIState().audio, AudioInState)


def test_a_fresh_rig_is_empty_and_off():
    st = AudioInState()
    assert st.mappings == [] and st.enabled is False


def test_enabled_is_never_persisted():
    """Launching the app must not start capturing audio on its own."""
    assert "enabled" not in PERSISTED_FIELDS


def test_one_shot_commands_are_never_persisted():
    for name in ("request_start", "request_stop", "open_target"):
        assert name not in PERSISTED_FIELDS


def test_a_round_trip_preserves_a_mapping():
    st = AudioInState()
    st.mappings.append(Mapping(signal="bass", target="SENSOR_GAIN",
                               mode="multiply", depth=0.25, gain=1.5,
                               shaper=ShaperParams(kind="lfo", rate_max=9.0)))
    st.global_strength = 1.5
    st.strengths["SENSOR_GAIN"] = 0.5

    fresh = AudioInState()
    apply_dict(fresh, to_dict(st))

    m = fresh.mappings[0]
    assert (m.signal, m.target, m.mode) == ("bass", "SENSOR_GAIN", "multiply")
    assert m.depth == pytest.approx(0.25) and m.gain == pytest.approx(1.5)
    assert m.shaper.kind == "lfo" and m.shaper.rate_max == pytest.approx(9.0)
    assert fresh.global_strength == pytest.approx(1.5)
    assert fresh.strengths["SENSOR_GAIN"] == pytest.approx(0.5)


def test_brain_mappings_round_trip_per_modality():
    st = AudioInState()
    st.brain_mappings["lenia"] = [Mapping(signal="hi", target="mu_scale")]
    fresh = AudioInState()
    apply_dict(fresh, to_dict(st))
    assert fresh.brain_mappings["lenia"][0].target == "mu_scale"
    assert "gabor" not in fresh.brain_mappings


def test_an_empty_dict_keeps_what_is_already_there():
    st = AudioInState()
    st.global_strength = 1.7
    apply_dict(st, {})
    assert st.global_strength == pytest.approx(1.7)


def test_a_rig_written_by_a_newer_build_still_opens():
    st = AudioInState()
    apply_dict(st, {"global_strength": 1.2, "some_future_field": 99})
    assert st.global_strength == pytest.approx(1.2)


def test_a_malformed_mapping_is_dropped_rather_than_crashing():
    st = AudioInState()
    apply_dict(st, {"mappings": [
        {"signal": "bass", "target": "DRAG", "mode": "add"},
        {"signal": "bass"},                       # no target
        "not a dict",
        {"signal": "bass", "target": "DRAG", "mode": "sideways"},
    ]})
    assert len(st.mappings) == 1
    assert st.mappings[0].target == "DRAG"


def test_strengths_are_clamped_to_the_slider_range():
    st = AudioInState()
    apply_dict(st, {"global_strength": 99.0, "strengths": {"DRAG": -4.0}})
    assert st.global_strength == pytest.approx(2.0)
    assert st.strengths["DRAG"] == pytest.approx(0.0)
