"""A brain's audio channel is a mapping TARGET with a base of zero.

The rows, shapers, masks and mutes are the rig's own; what is new is the
target and the per-cohort value it produces. See
docs/superpowers/specs/2026-09-06-brain-audio-inputs-design.md.
"""
import numpy as np
import pytest

from services import cohort_audio as ca
from services.audio_mapping import (Channel, Mapping, channel_key,
                                    channel_targets)
from services.audio_runtime import AudioRuntime
from services.audio_shapers import ShaperParams
from services.brains import REGISTRY
from services.brains.layout_moves import grow_inputs
from state import UIState
from state.audio_in_state import AudioInState, apply_dict, to_dict

K = 2


def _layout(k=K):
    base = REGISTRY["fourier"].layout_from_settings({})
    return grow_inputs(base, k) if k else base


def _values(mappings, channels=None, signals=None, n=8, layout=None,
            strengths=None, global_strength=1.0, deaf=(), states=None):
    layout = layout or _layout()
    targets = channel_targets(layout, channels or [])
    return ca.channel_values(
        mappings, targets, signals or {"bass": 1.0, "hi": 0.0},
        states if states is not None else {}, strengths or {},
        global_strength, 1 / 60, set(deaf), n)


# ---- targets --------------------------------------------------------------

def test_a_channel_is_a_target_per_input():
    chans = [Channel(name="kick", range=0.35)]
    targets = channel_targets(_layout(), chans)
    assert [t.key for t in targets] == [channel_key(0), channel_key(1)]
    assert [t.label for t in targets] == ["kick", "A2"]
    assert all(t.group == "channel" for t in targets)
    kick = targets[0]
    assert (kick.lo, kick.hi) == (-0.35, 0.35)
    assert (kick.hard_lo, kick.hard_hi) == (-0.35, 0.35)


def test_a_deaf_layout_offers_no_channels():
    assert channel_targets(_layout(0), [Channel()]) == []


def test_extra_channels_beyond_k_are_kept_waiting():
    chans = [Channel(name="a"), Channel(name="b"), Channel(name="c")]
    assert len(channel_targets(_layout(), chans)) == K


# ---- values ---------------------------------------------------------------

def test_a_full_scale_add_row_reads_as_the_range_on_every_cohort():
    m = Mapping(signal="bass", target=channel_key(0), mode="add", depth=1.0)
    arr = _values([m], [Channel(range=0.35)])
    assert arr.shape == (K, ca.MASK_SLOTS) and arr.dtype == np.float32
    assert np.allclose(arr[0, :8], 0.35)
    assert np.all(arr[1] == 0.0)


def test_silence_is_exactly_zero():
    m = Mapping(signal="bass", target=channel_key(0), mode="add", depth=1.0)
    arr = _values([m], signals={"bass": 0.0})
    assert np.all(arr == 0.0)


def test_a_subtract_row_goes_negative():
    m = Mapping(signal="bass", target=channel_key(0), mode="subtract",
                depth=0.5)
    arr = _values([m], [Channel(range=0.4)])
    assert np.allclose(arr[0, :8], -0.2)


def test_a_multiply_row_alone_multiplies_zero():
    m = Mapping(signal="bass", target=channel_key(0), mode="multiply",
                depth=1.0)
    assert np.all(_values([m]) == 0.0)


def test_the_sum_is_clamped_to_the_range():
    rows = [Mapping(signal="bass", target=channel_key(0), mode="add",
                    depth=1.0) for _ in range(3)]
    arr = _values(rows, [Channel(range=0.3)])
    assert np.allclose(arr[0, :8], 0.3)


def test_strength_scales_the_value():
    m = Mapping(signal="bass", target=channel_key(0), mode="add", depth=1.0)
    arr = _values([m], [Channel(range=1.0)],
                  strengths={channel_key(0): 0.5}, global_strength=0.5)
    assert np.allclose(arr[0, :8], 0.25)


def test_a_masked_row_reaches_only_its_cohorts():
    m = Mapping(signal="bass", target=channel_key(0), mode="add", depth=1.0)
    m.cohorts[:] = False
    ca.paint(m.cohorts, 0, 4, True)          # first of four cells
    arr = _values([m], [Channel(range=1.0)], n=4)
    assert np.allclose(arr[0, :4], [1.0, 0.0, 0.0, 0.0])


def test_a_muted_or_disabled_row_contributes_nothing():
    on = Mapping(signal="bass", target=channel_key(0), mode="add", depth=1.0)
    off = Mapping(signal="bass", target=channel_key(1), mode="add", depth=1.0,
                  enabled=False)
    assert _values([off]) is None
    assert _values([on], deaf={channel_key(0)}) is None


def test_no_live_row_means_none():
    assert _values([]) is None
    assert _values([Mapping(signal="bass", target="SENSOR_GAIN")]) is None


def test_shaper_state_is_keyed_by_uid_and_carried():
    m = Mapping(signal="bass", target=channel_key(0), mode="add", depth=1.0,
                shaper=ShaperParams(kind="smooth", attack=1.0, release=1.0))
    states = {}
    first = _values([m], [Channel(range=1.0)], states=states)[0, 0]
    second = _values([m], [Channel(range=1.0)], states=states)[0, 0]
    assert m.uid in states
    assert 0.0 < first < second < 1.0


# ---- persistence ----------------------------------------------------------

def test_channels_round_trip_through_the_rig():
    st = AudioInState()
    st.channels = [Channel(name="kick", range=0.35), Channel()]
    st.channel_mappings.append(Mapping(signal="hi", target=channel_key(1),
                                       mode="subtract", depth=0.3))
    st.feed = False
    d = to_dict(st)
    fresh = AudioInState()
    apply_dict(fresh, d)
    assert [(c.name, c.range) for c in fresh.channels] == [("kick", 0.35),
                                                          ("", 0.25)]
    assert fresh.channel_mappings[0].target == channel_key(1)
    assert fresh.channel_mappings[0].mode == "subtract"
    assert fresh.feed is False


def test_a_rig_written_before_channels_existed_opens_untouched():
    st = AudioInState()
    apply_dict(st, {"mappings": []})
    assert st.channels == [] and st.channel_mappings == [] and st.feed is True


def test_a_malformed_channel_keeps_its_defaults_and_a_bad_row_is_dropped():
    st = AudioInState()
    apply_dict(st, {"channels": [{"name": 3, "range": "x"}, "junk"],
                    "channel_mappings": [{"signal": "bass"}, 5]})
    assert len(st.channels) == 2
    assert st.channels[0].name == "" and st.channels[0].range == 0.25
    assert st.channel_mappings == []


def test_channel_uids_are_fresh_per_load():
    st = AudioInState()
    st.channels = [Channel(name="a")]
    a = to_dict(st)
    x, y = AudioInState(), AudioInState()
    apply_dict(x, a)
    apply_dict(y, a)
    assert x.channels[0].uid != y.channels[0].uid


# ---- the runtime ----------------------------------------------------------

class FakeSnapshot:
    def __init__(self, **signals):
        self.signals = signals
        self.mel = np.zeros(40, dtype=np.float32)
        self.seq = 1


def _rig():
    st = UIState()
    st.audio.enabled = True
    st.audio.channels = [Channel(name="kick", range=0.5)]
    st.audio.channel_mappings.append(
        Mapping(signal="bass", target=channel_key(0), mode="add", depth=1.0))
    return st


def _runtime(**signals):
    rt = AudioRuntime()
    rt.capture._snapshot = FakeSnapshot(**signals)
    return rt


def test_the_runtime_hands_the_sim_the_channel_array():
    rt, st = _runtime(bass=1.0), _rig()
    rt.update(st, 1 / 60, _layout(), np.zeros(_layout().length, np.float32))
    assert rt.audio_inputs is not None
    assert rt.audio_inputs.shape == (K, ca.MASK_SLOTS)
    assert np.allclose(rt.audio_inputs[0, :st.sim.num_cohorts], 0.5)


def test_feed_off_is_silence():
    rt, st = _runtime(bass=1.0), _rig()
    st.audio.feed = False
    rt.update(st, 1 / 60, _layout(), None)
    assert rt.audio_inputs is None


def test_a_deaf_brain_gets_nothing():
    rt, st = _runtime(bass=1.0), _rig()
    rt.update(st, 1 / 60, _layout(0), None)
    assert rt.audio_inputs is None


def test_capture_off_is_silence():
    rt, st = _runtime(bass=1.0), _rig()
    st.audio.enabled = False
    rt.update(st, 1 / 60, _layout(), None)
    assert rt.audio_inputs is None


def test_a_search_hears_nothing():
    rt, st = _runtime(bass=1.0), _rig()
    st.archive.enabled = True
    rt.update(st, 1 / 60, _layout(), None)
    assert rt.audio_inputs is None


def test_the_array_is_cleared_when_the_rig_goes_quiet():
    rt, st = _runtime(bass=1.0), _rig()
    rt.update(st, 1 / 60, _layout(), None)
    assert rt.audio_inputs is not None
    st.audio.modulate = False
    rt.update(st, 1 / 60, _layout(), None)
    assert rt.audio_inputs is None


def test_channel_shaper_states_survive_pruning():
    rt, st = _runtime(bass=1.0), _rig()
    st.audio.channel_mappings[0].shaper = ShaperParams(kind="smooth")
    rt.update(st, 1 / 60, _layout(), None)
    assert st.audio.channel_mappings[0].uid in rt._states


# ---- the seed travels with the config -------------------------------------

def test_the_audio_seed_is_saved_with_the_config_and_defaults_when_absent():
    from services.config_saver import ConfigSaver, PhysicsConfig
    from state.sim_state import SimState

    st = SimState()
    st.audio_seed = 0.777
    cfg = ConfigSaver().create_config(st, None)
    back = PhysicsConfig.from_dict(cfg.to_dict())
    assert back.audio_seed == pytest.approx(0.777)
    fresh = SimState()
    ConfigSaver().apply_config(back, fresh)
    assert fresh.audio_seed == pytest.approx(0.777)

    old = PhysicsConfig.from_dict({"simulation": {}, "physics": {}})
    assert old.audio_seed == SimState().audio_seed
