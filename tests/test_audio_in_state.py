"""What a rig remembers, and what it must refuse to remember."""
import subprocess
import sys
from pathlib import Path

import pytest

from services.audio_mapping import Mapping
from services.audio_shapers import ShaperParams
from state.audio_in_state import (PERSISTED_FIELDS, AudioInState, apply_dict,
                                  to_dict)


def test_it_hangs_off_ui_state():
    from state import UIState
    assert isinstance(UIState().audio, AudioInState)


def test_importing_state_first_does_not_deadlock_on_services():
    """A fresh interpreter, importing `state` before anything else.

    Every other test in the suite imports `services` first, which hides a
    module-scope `services` import here: services/__init__ reaches ui, which
    imports back from state while state/__init__ is still part-built.
    """
    root = Path(__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, "-c", "import state; print(state.UIState().audio)"],
                       cwd=str(root), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


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
                               shaper=ShaperParams(kind="phase", rate=3.0)))
    st.global_strength = 1.5
    st.strengths["SENSOR_GAIN"] = 0.5

    fresh = AudioInState()
    apply_dict(fresh, to_dict(st))

    m = fresh.mappings[0]
    assert (m.signal, m.target, m.mode) == ("bass", "SENSOR_GAIN", "multiply")
    assert m.depth == pytest.approx(0.25) and m.gain == pytest.approx(1.5)
    assert m.shaper.kind == "phase" and m.shaper.rate == pytest.approx(3.0)
    assert fresh.global_strength == pytest.approx(1.5)
    assert fresh.strengths["SENSOR_GAIN"] == pytest.approx(0.5)


def test_a_rig_saved_with_an_lfo_row_loads_as_a_phase_shaper():
    """The free-running oscillator is gone; a stored row must not silently
    become 'none', which is a mapping that does nothing."""
    fresh = AudioInState()
    apply_dict(fresh, {"mappings": [{
        "signal": "mid", "target": "SENSOR_GAIN", "mode": "add", "depth": 0.4,
        "gain": 1.0, "enabled": True,
        "shaper": {"kind": "lfo", "rate_min": 0.5, "rate_max": 12.0,
                   "wave": "triangle"},
    }]})
    m = fresh.mappings[0]
    assert m.shaper.kind == "phase"
    assert m.shaper.wave == "triangle"
    assert m.depth == pytest.approx(0.4)
    # rate_max was already the Hz a full-scale band asks for, so it carries
    # straight over; rate_min was the floor that made it run on silence.
    assert m.shaper.rate == pytest.approx(12.0)


def test_auto_gain_is_off_until_it_is_asked_for():
    """It amplifies a room's noise floor to near full scale on every band."""
    assert AudioInState().auto_gain is False


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


def test_the_band_measures_and_the_release_survive_a_save():
    st = AudioInState()
    st.bands = {"hi": {"measure": "peak", "floor": -40.0, "ceiling": -8.0}}
    st.release_seconds = 0.0

    back = AudioInState()
    apply_dict(back, to_dict(st))
    assert back.bands["hi"] == {"measure": "peak", "floor": -40.0,
                                "ceiling": -8.0}
    assert back.release_seconds == pytest.approx(0.0)


def test_a_rig_saved_before_the_measures_existed_opens_on_the_defaults():
    from services.audio_analysis import effective_bands

    st = AudioInState()
    apply_dict(st, {"global_strength": 1.0})
    assert st.bands == {}
    assert effective_bands(st.bands)["bass"]["measure"] == "power"


def test_the_release_is_clamped_to_the_slider_range():
    st = AudioInState()
    apply_dict(st, {"release_seconds": -3.0})
    assert st.release_seconds == pytest.approx(0.0)
    apply_dict(st, {"release_seconds": 99.0})
    assert st.release_seconds == pytest.approx(2.0)
