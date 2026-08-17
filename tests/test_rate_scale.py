"""One knob that slides a whole rig onto another tempo.

`rate_scale` multiplies every phase shaper's rate at once. It is FREQUENCY
only: an envelope's attack, a gate's hold and a smoother's release are
durations, and a tempo control that stretched those would be a different
control wearing the same label.
"""
import numpy as np
import pytest

from services.audio_mapping import Mapping, TargetDef, modulate
from services.audio_shapers import SHAPER_KINDS, ShaperParams, ShaperState
from state import UIState
from state.audio_in_state import apply_dict, to_dict

DT = 1.0 / 60.0


def test_the_default_multiplies_nothing():
    assert UIState().audio.rate_scale == 1.0


def test_it_multiplies_the_phase_advance():
    def travelled(scale):
        s = ShaperState()
        p = ShaperParams(kind="phase", rate=1.0)
        for _ in range(30):
            s.apply(1.0, DT, p, True, scale)
        return s._phase

    assert travelled(2.0) == pytest.approx(2 * travelled(1.0), abs=1e-9)
    assert travelled(0.5) == pytest.approx(travelled(1.0) / 2, abs=1e-9)


def test_doubling_the_scale_is_doubling_every_rate():
    """The whole promise: one knob does what editing every row would."""
    fast = ShaperState()
    slow = ShaperState()
    for _ in range(45):
        fast.apply(0.7, DT, ShaperParams(kind="phase", rate=6.0), True, 1.0)
        slow.apply(0.7, DT, ShaperParams(kind="phase", rate=3.0), True, 2.0)
    assert fast._phase == pytest.approx(slow._phase, abs=1e-9)


@pytest.mark.parametrize("kind", [k for k in SHAPER_KINDS if k != "phase"])
def test_no_other_shaper_notices_it(kind):
    """Durations are not frequencies. Scaling dt would have reached all of
    these, which is why the scale is applied to the rate instead."""
    def run(scale):
        s = ShaperState()
        p = ShaperParams(kind=kind)
        return [s.apply(v, DT, p, True, scale)
                for v in ([0.9] * 20 + [0.0] * 60)]

    assert run(4.0) == run(1.0)


def test_it_cannot_make_a_silent_band_move():
    """The rule every shaper obeys, and a rate multiplier is exactly the kind
    of thing that would break it."""
    s = ShaperState()
    p = ShaperParams(kind="phase", rate=2.0)
    for _ in range(30):
        s.apply(0.9, DT, p, True, 8.0)
    settled = [s.apply(0.0, DT, p, True, 8.0) for _ in range(200)]
    assert max(settled) - min(settled) < 1e-12


def test_it_reaches_the_shaper_through_modulate():
    """The traces the panel draws ARE the shaper outputs recorded here, so a
    scale that stopped at modulate() would move the sim and not the drawing."""
    target = TargetDef("SENSOR_GAIN", "Sensor Gain", "physics",
                       0.0, 5.0, None, None)
    m = Mapping(signal="bass", target="SENSOR_GAIN", mode="add", depth=0.5,
                shaper=ShaperParams(kind="phase", rate=1.0))

    def phase_after(scale):
        states = {}
        for _ in range(30):
            modulate({"SENSOR_GAIN": 1.0}, [target], [m], {"bass": 1.0},
                     states, {}, 1.0, DT, set(), rate_scale=scale)
        return states[m.uid]._phase

    assert phase_after(2.0) == pytest.approx(2 * phase_after(1.0), abs=1e-9)


def test_a_shaped_trace_moves_with_it():
    """`shaped` is what the drawer plots."""
    target = TargetDef("SENSOR_GAIN", "Sensor Gain", "physics",
                       0.0, 5.0, None, None)
    m = Mapping(signal="bass", target="SENSOR_GAIN", mode="add", depth=0.5,
                shaper=ShaperParams(kind="phase", rate=1.0, wave="ramp"))

    def trace(scale):
        states, out = {}, []
        for _ in range(30):
            shaped = {}
            modulate({"SENSOR_GAIN": 1.0}, [target], [m], {"bass": 1.0},
                     states, {}, 1.0, DT, set(), shaped, rate_scale=scale)
            out.append(shaped[m.uid])
        return out

    assert trace(3.0) != trace(1.0)
    assert max(trace(3.0)) > max(trace(1.0))


def test_it_survives_a_save_and_a_load():
    st = UIState().audio
    st.rate_scale = 2.75
    back = UIState().audio
    apply_dict(back, to_dict(st))
    assert back.rate_scale == pytest.approx(2.75)


def test_a_rig_saved_before_it_existed_loads_at_one():
    st = UIState().audio
    st.rate_scale = 4.0
    apply_dict(st, {"mappings": []})
    assert st.rate_scale == 4.0, "a missing key keeps what is on screen"

    fresh = UIState().audio
    apply_dict(fresh, {"mappings": []})
    assert fresh.rate_scale == 1.0


def test_a_nonsense_value_is_clamped_rather_than_trusted():
    st = UIState().audio
    apply_dict(st, {"rate_scale": 10_000.0})
    assert st.rate_scale == 8.0
    apply_dict(st, {"rate_scale": -3.0})
    assert st.rate_scale == 0.05
