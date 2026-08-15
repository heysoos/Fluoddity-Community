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


# --- shaper state does not outlive the mapping that made it ------------------

def test_a_deleted_mappings_shaper_state_is_dropped():
    """Nothing prunes on delete, so the table has to be cut back to the rig -
    an orphan left behind is the envelope the next row would inherit."""
    rt, st = runtime_with({"bass": 1.0}), rig()
    rt.update(st, 1 / 60, None, None)
    gone = st.audio.mappings[0].uid
    assert gone in rt._states

    st.audio.mappings.clear()
    st.audio.mappings.append(Mapping(signal="bass", target="SENSOR_GAIN",
                                     mode="add", depth=0.5))
    rt.update(st, 1 / 60, None, None)
    assert gone not in rt._states
    assert set(rt._states) == {st.audio.mappings[0].uid}


def test_editing_the_rig_all_session_does_not_grow_the_state_table():
    rt, st = runtime_with({"bass": 1.0}), rig()
    for _ in range(50):
        st.audio.mappings[:] = [Mapping(signal="bass", target="SENSOR_GAIN",
                                        mode="add", depth=0.5)]
        rt.update(st, 1 / 60, None, None)
    assert len(rt._states) == 1


def test_an_inactive_modality_keeps_its_shaper_state():
    """The whole rig is live, not just the modality on screen: switching brains
    and back must not restart the shapers that were waiting there."""
    from services import brains
    m = brains.get("fourier")
    layout = m.layout_from_settings({})
    params = m.random(np.random.default_rng(0), layout)

    rt, st = runtime_with({"bass": 1.0}), rig()
    st.brain.modality = "fourier"
    st.audio.brain_mappings["fourier"] = [
        Mapping(signal="bass", target="freq_scale", mode="add", depth=0.5)
    ]
    rt.update(st, 1 / 60, layout, params)
    uid = st.audio.brain_mappings["fourier"][0].uid
    assert uid in rt._states

    st.brain.modality = "mlp"
    rt.update(st, 1 / 60, layout, params)
    assert uid in rt._states


# --- status while a loopback endpoint is idle --------------------------------

def test_an_active_capture_with_no_audio_yet_reads_waiting():
    """WASAPI loopback delivers nothing while the endpoint is silent, so a
    correctly-running capture can sit with no snapshot indefinitely. Reporting
    plain 'active' there makes a working panel look broken."""
    rt, st = AudioRuntime(), rig()
    rt.capture.status = "active"
    rt.update(st, 1 / 60, None, None)
    assert st.audio.status == "waiting"


def test_once_blocks_arrive_the_status_is_active():
    rt, st = runtime_with({"bass": 0.5}), rig()
    rt.capture.status = "active"
    rt.update(st, 1 / 60, None, None)
    assert st.audio.status == "active"


def test_idle_and_error_are_reported_unchanged():
    for status in ("idle", "error"):
        rt, st = AudioRuntime(), rig()
        rt.capture.status = status
        rt.update(st, 1 / 60, None, None)
        assert st.audio.status == status


# --- the in-track overlay ----------------------------------------------------

def test_an_unmodulated_run_offers_no_overlays():
    """An unbound slider must stay pixel-identical, so it gets no entry."""
    rt, st = AudioRuntime(), rig()
    st.audio.enabled = False
    sim_out, _ = rt.update(st, 1 / 60, None, None)
    assert rt.overlays(st, sim_out) == {}


def test_a_bound_slider_gets_base_live_reach_and_a_colour():
    rt, st = runtime_with({"bass": 0.5}), rig()
    sim_out, _ = rt.update(st, 1 / 60, None, None)
    ov = rt.overlays(st, sim_out)["SENSOR_GAIN"]
    assert set(ov) == {"lo", "hi", "base", "live", "reach", "color"}
    assert ov["base"] == pytest.approx(1.0)
    assert ov["live"] == pytest.approx(sim_out.SENSOR_GAIN)


def test_reach_is_where_a_full_scale_signal_would_land_not_the_live_value():
    rt, st = runtime_with({"bass": 0.25}), rig()
    sim_out, _ = rt.update(st, 1 / 60, None, None)
    ov = rt.overlays(st, sim_out)["SENSOR_GAIN"]
    assert ov["reach"] > ov["live"]


def test_only_bound_targets_appear():
    rt, st = runtime_with({"bass": 1.0}), rig()
    sim_out, _ = rt.update(st, 1 / 60, None, None)
    assert set(rt.overlays(st, sim_out)) == {"SENSOR_GAIN"}


def test_a_swept_target_draws_no_overlay():
    rt, st = runtime_with({"bass": 1.0}), rig()
    st.sim.x_sweeps["SENSOR_GAIN"] = 1.0
    sim_out, _ = rt.update(st, 1 / 60, None, None)
    assert "SENSOR_GAIN" not in rt.overlays(st, sim_out)


# --- the analyser follows the panel while it plays ---------------------------

class _SpyCapture:
    """Records what the runtime pushes at it each frame."""

    def __init__(self):
        self.status = "active"
        self.last_error = ""
        self.pushed = []
        self._snap = FakeSnapshot(bass=1.0)

    def snapshot(self):
        return self._snap

    def set_auto_gain(self, on):
        pass

    def set_bands(self, bands):
        self.pushed.append(("bands", dict(bands or {})))

    def set_release(self, seconds):
        self.pushed.append(("release", seconds))


def test_the_measures_and_the_release_are_pushed_every_frame():
    """Otherwise the only way to change one is a Stop and a Start, with the
    music still playing."""
    rt, st = AudioRuntime(), rig()
    rt.capture = _SpyCapture()
    st.audio.bands = {"hi": {"measure": "peak", "floor": -40.0,
                             "ceiling": -8.0}}
    st.audio.release_seconds = 0.0

    rt.update(st, 1 / 60, None, None)
    rt.update(st, 1 / 60, None, None)

    assert rt.capture.pushed.count(("release", 0.0)) == 2
    bands = [v for k, v in rt.capture.pushed if k == "bands"]
    assert len(bands) == 2 and bands[0]["hi"]["measure"] == "peak"


def test_probing_full_scale_does_not_disturb_the_live_shaper_state():
    """overlays() drives the mappings at signal 1.0; a shared shaper state
    would make the next frame's envelope think a peak had just arrived."""
    from services.audio_shapers import ShaperParams

    rt, st = runtime_with({"bass": 0.0}), rig()
    st.audio.mappings[0].shaper = ShaperParams(kind="smooth", attack=0.5,
                                               release=0.5)
    first, _ = rt.update(st, 1 / 60, None, None)
    rt.overlays(st, first)
    second, _ = rt.update(st, 1 / 60, None, None)
    assert second.SENSOR_GAIN == pytest.approx(first.SENSOR_GAIN, abs=1e-6)
