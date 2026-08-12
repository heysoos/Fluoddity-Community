"""The seam: what the sim receives, and what the user's sliders keep."""
import numpy as np
import pytest

from services.audio_mapping import Mapping
from services.audio_runtime import AudioRuntime
from state import UIState


class FakeSnapshot:
    def __init__(self, **signals):
        self.signals = signals
        self.mel = np.zeros(40, dtype=np.float32)
        self.seq = 1


def rig(**kw):
    st = UIState()
    st.audio.enabled = True
    st.audio.mappings.append(Mapping(signal="bass", target="SENSOR_GAIN",
                                     mode="add", depth=0.5))
    st.sim.SENSOR_GAIN = 1.0
    for k, v in kw.items():
        setattr(st.audio, k, v)
    return st


def runtime_with(signals):
    rt = AudioRuntime()
    rt.capture._snapshot = FakeSnapshot(**signals)
    return rt


def test_disabled_returns_the_state_object_untouched():
    rt, st = AudioRuntime(), rig()
    st.audio.enabled = False
    out, _brain = rt.update(st, 1 / 60, None, None)
    assert out is st.sim


def test_with_no_snapshot_the_state_is_returned_untouched():
    rt, st = AudioRuntime(), rig()
    out, _brain = rt.update(st, 1 / 60, None, None)
    assert out is st.sim


def test_the_sim_receives_a_modulated_copy():
    rt, st = runtime_with({"bass": 1.0}), rig()
    out, _brain = rt.update(st, 1 / 60, None, None)
    assert out is not st.sim
    assert out.SENSOR_GAIN > 1.0


def test_the_users_slider_value_is_never_written():
    rt, st = runtime_with({"bass": 1.0}), rig()
    rt.update(st, 1 / 60, None, None)
    assert st.sim.SENSOR_GAIN == pytest.approx(1.0)


def test_the_copy_shares_nothing_the_caller_will_mutate():
    """A shallow copy shares the sweep dicts; writing one must not reach back."""
    rt, st = runtime_with({"bass": 1.0}), rig()
    out, _ = rt.update(st, 1 / 60, None, None)
    assert out.slider_ranges is st.sim.slider_ranges


def test_auto_mode_suppresses_modulation():
    """CLIP ranks tiles against each other; audio would move the physics
    mid-comparison."""
    rt, st = runtime_with({"bass": 1.0}), rig()
    st.auto_tournament.enabled = True
    out, _brain = rt.update(st, 1 / 60, None, None)
    assert out is st.sim


def test_explore_mode_suppresses_modulation():
    rt, st = runtime_with({"bass": 1.0}), rig()
    st.archive.enabled = True
    out, _brain = rt.update(st, 1 / 60, None, None)
    assert out is st.sim


def test_a_swept_target_is_left_alone():
    rt, st = runtime_with({"bass": 1.0}), rig()
    st.sim.x_sweeps["SENSOR_GAIN"] = 1.0
    out, _brain = rt.update(st, 1 / 60, None, None)
    assert out.SENSOR_GAIN == pytest.approx(1.0)


def test_a_brain_mapping_produces_a_modulated_brain():
    from services import brains
    m = brains.get("fourier")
    layout = m.layout_from_settings({})
    params = m.random(np.random.default_rng(0), layout)

    rt, st = runtime_with({"bass": 1.0}), rig()
    st.brain.modality = "fourier"
    st.audio.brain_mappings["fourier"] = [
        Mapping(signal="bass", target="freq_scale", mode="add", depth=0.5)
    ]
    _sim, brain = rt.update(st, 1 / 60, layout, params)
    assert brain is not None
    assert not np.allclose(brain, params)


def test_no_brain_mapping_means_no_brain_write():
    from services import brains
    m = brains.get("fourier")
    layout = m.layout_from_settings({})
    params = m.random(np.random.default_rng(0), layout)
    rt, st = runtime_with({"bass": 1.0}), rig()
    _sim, brain = rt.update(st, 1 / 60, layout, params)
    assert brain is None


def test_a_new_base_brain_is_re_encoded_rather_than_stacked():
    from services import brains
    m = brains.get("fourier")
    layout = m.layout_from_settings({})
    first = m.random(np.random.default_rng(1), layout)
    second = m.random(np.random.default_rng(2), layout)

    rt, st = runtime_with({"bass": 0.0}), rig()
    st.brain.modality = "fourier"
    st.audio.brain_mappings["fourier"] = [
        Mapping(signal="bass", target="freq_scale", mode="add", depth=0.5)
    ]
    rt.update(st, 1 / 60, layout, first)
    _sim, brain = rt.update(st, 1 / 60, layout, second)
    assert np.allclose(brain, second, atol=1e-2)


def test_close_is_safe_to_call_twice():
    rt = AudioRuntime()
    rt.close()
    rt.close()
