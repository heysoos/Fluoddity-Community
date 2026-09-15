"""Knobs nudge the real slider; rings show the value the sim ran under.

Driven against a fake port, so nothing here needs a device or the MIDI
packages. The Twister's channel numbers come from services.midi_control.
"""
from dataclasses import replace

import pytest

from services import midi_control as mc
from services.midi_runtime import GESTURE_SECONDS, MidiRuntime
from state.preferences_state import PreferencesState
from state.perform_state import PerformState
from state.set_list import KnobBinding, SetList
from state.sim_state import SimState

DEVICE = "Midi Fighter Twister"


class FakePort:
    def __init__(self, name, fail_send=False):
        self.name = name
        self.queued = []
        self.sent = []
        self.closed = False
        self.fail_send = fail_send
        self.can_send = True

    def poll(self):
        out, self.queued = self.queued, []
        return out

    def send_cc(self, channel, control, value):
        if self.fail_send:
            raise OSError("unplugged")
        self.sent.append((channel, control, value))

    def close(self):
        self.closed = True


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


class State:
    def __init__(self):
        from state.audio_in_state import AudioInState
        self.sim = SimState()
        self.audio = AudioInState()
        self.perform = PerformState()
        self.preferences = PreferencesState()
        self.undo_tag = ""


@pytest.fixture
def rig():
    ports = []
    devices = [DEVICE]

    def open_port(name):
        p = FakePort(name)
        ports.append(p)
        return p

    clock = Clock()
    rt = MidiRuntime(open_port=open_port, clock=clock,
                     list_devices=lambda: list(devices))
    st = State()
    st.preferences.midi_device = DEVICE
    st.perform.set_list = SetList(knobs=[
        KnobBinding(cc=0, param="SENSOR_GAIN", step=0.1),
        KnobBinding(cc=1, param="DRAG", step=0.01),
        KnobBinding(cc=2, param="audio:global_strength", step=0.1),
        KnobBinding(cc=3, param="audio:rate_scale", step=0.25)])
    return rt, st, ports, devices, clock


def _turn(port, cc, detents):
    port.queued.append((mc.ENCODER_CHANNEL, cc, 64 + detents))


def _range(st, param):
    from services.midi_runtime import slider_range
    return slider_range(st.sim, param)


def test_a_detent_moves_the_slider_by_a_fraction_of_its_range(rig):
    rt, st, ports, _, _ = rig
    rt.update(st)
    lo, hi = _range(st, "SENSOR_GAIN")
    before = st.sim.SENSOR_GAIN = (lo + hi) / 2
    _turn(ports[0], 0, +1)
    rt.update(st)
    assert st.sim.SENSOR_GAIN == pytest.approx(before + 0.1 * (hi - lo))
    _turn(ports[0], 0, -2)
    rt.update(st)
    assert st.sim.SENSOR_GAIN == pytest.approx(before - 0.1 * (hi - lo))


def test_a_turn_is_clamped_to_the_slider_range(rig):
    rt, st, ports, _, _ = rig
    rt.update(st)
    lo, hi = _range(st, "SENSOR_GAIN")
    _turn(ports[0], 0, +50)
    rt.update(st)
    assert st.sim.SENSOR_GAIN == pytest.approx(hi)
    _turn(ports[0], 0, -50)
    rt.update(st)
    assert st.sim.SENSOR_GAIN == pytest.approx(lo)


def test_an_unbound_encoder_and_a_value_of_64_do_nothing(rig):
    rt, st, ports, _, _ = rig
    rt.update(st)
    snap = replace(st.sim)
    ports[0].queued.append((mc.ENCODER_CHANNEL, 9, 65))
    _turn(ports[0], 0, 0)
    rt.update(st)
    assert st.sim == snap
    assert not rt.gesture_active()


def test_a_push_returns_the_slider_to_where_the_preset_loaded_it(rig):
    rt, st, ports, _, _ = rig
    st.sim.SENSOR_GAIN = 0.3
    st.perform.recentre_knobs = True
    rt.update(st)
    assert st.perform.recentre_knobs is False
    _turn(ports[0], 0, +3)
    rt.update(st)
    assert st.sim.SENSOR_GAIN != pytest.approx(0.3)
    ports[0].queued.append((mc.SWITCH_CHANNEL, 0, mc.SWITCH_PRESSED))
    rt.update(st)
    assert st.sim.SENSOR_GAIN == pytest.approx(0.3)


def test_a_push_release_is_not_a_press(rig):
    rt, st, ports, _, _ = rig
    st.perform.recentre_knobs = True
    rt.update(st)
    _turn(ports[0], 0, +3)
    rt.update(st)
    moved = st.sim.SENSOR_GAIN
    ports[0].queued.append((mc.SWITCH_CHANNEL, 0, 0))
    rt.update(st)
    assert st.sim.SENSOR_GAIN == moved


def test_a_twist_is_one_undo_gesture_named_after_the_slider(rig):
    rt, st, ports, _, clock = rig
    rt.update(st)
    _turn(ports[0], 0, +1)
    rt.update(st)
    assert rt.gesture_active()
    assert st.undo_tag == "Sensor Gain"
    clock.t += GESTURE_SECONDS / 2
    assert rt.gesture_active()
    clock.t += GESTURE_SECONDS
    assert not rt.gesture_active()


def test_the_device_is_opened_when_it_appears_and_dropped_when_it_goes(rig):
    rt, st, ports, devices, clock = rig
    devices.clear()
    rt.update(st)
    assert not rt.connected
    assert "not connected" in st.perform.midi_status
    devices.append(DEVICE)
    rt.update(st)
    assert not rt.connected, "the port list is polled once a second, not per frame"
    clock.t += 1.5
    rt.update(st)
    assert rt.connected
    ports[0].fail_send = True
    rt.feedback(st, st.sim)
    assert not rt.connected
    assert ports[0].closed
    assert "Lost" in st.perform.midi_status


def test_choosing_no_device_closes_the_one_open(rig):
    rt, st, ports, _, _ = rig
    rt.update(st)
    assert rt.connected
    st.preferences.midi_device = ""
    rt.update(st)
    assert not rt.connected and ports[0].closed


def test_connecting_switches_off_every_encoder_the_set_list_does_not_bind(rig):
    rt, st, ports, _, _ = rig
    rt.update(st)
    rings = {cc for ch, cc, v in ports[0].sent if ch == mc.ENCODER_CHANNEL and v == 0}
    assert rings == set(range(mc.ENCODER_COUNT)) - {0, 1, 2, 3}


def test_the_ring_shows_the_value_the_sim_ran_under(rig):
    rt, st, ports, _, _ = rig
    rt.update(st)
    lo, hi = _range(st, "SENSOR_GAIN")
    st.sim.SENSOR_GAIN = lo
    modulated = replace(st.sim, SENSOR_GAIN=hi)
    ports[0].sent.clear()
    rt.feedback(st, modulated)
    assert (mc.ENCODER_CHANNEL, 0, 127) in ports[0].sent
    assert (mc.SWITCH_CHANNEL, 0, st.perform.set_list.knobs[0].active_colour) in ports[0].sent
    ports[0].sent.clear()
    rt.feedback(st, st.sim)
    assert (mc.ENCODER_CHANNEL, 0, 0) in ports[0].sent
    assert (mc.SWITCH_CHANNEL, 0, st.perform.set_list.knobs[0].rest_colour) in ports[0].sent


def test_an_unchanged_ring_sends_nothing(rig):
    rt, st, ports, _, _ = rig
    rt.update(st)
    rt.feedback(st, st.sim)
    ports[0].sent.clear()
    rt.feedback(st, st.sim)
    assert ports[0].sent == []


def test_a_swept_parameter_dims_its_encoder(rig):
    rt, st, ports, _, _ = rig
    rt.update(st)
    rt.feedback(st, st.sim)
    assert (mc.RGB_ANIM_CHANNEL, 1, mc.RGB_BRIGHTEST) in ports[0].sent
    st.sim.x_sweeps["DRAG"] = 0.5
    ports[0].sent.clear()
    rt.feedback(st, st.sim)
    assert (mc.RGB_ANIM_CHANNEL, 1, mc.RGB_DIM) in ports[0].sent


def test_relative_decoding_and_ring_mapping():
    assert mc.decode_relative(65) == 1
    assert mc.decode_relative(63) == -1
    assert mc.decode_relative(64) == 0
    assert mc.ring_value(0.5, 0.0, 1.0) == 64
    assert mc.ring_value(2.0, 0.0, 1.0) == 127
    assert mc.ring_value(-1.0, 0.0, 1.0) == 0
    assert mc.ring_value(1.0, 1.0, 1.0) == 0
    assert mc.port_base_name("Midi Fighter Twister 0") == "Midi Fighter Twister"
    assert mc.port_base_name("Twister 2 3") == "Twister 2"


def test_an_audio_knob_moves_the_rig_setting_and_a_push_returns_to_the_rig(rig):
    rt, st, ports, _, _ = rig
    st.audio.global_strength = 1.0
    st.perform.recentre_audio_knobs = True
    rt.update(st)
    _turn(ports[0], 2, +2)
    rt.update(st)
    assert st.audio.global_strength == pytest.approx(1.4)
    assert st.undo_tag == "Audio Strength"
    ports[0].queued.append((mc.SWITCH_CHANNEL, 2, mc.SWITCH_PRESSED))
    rt.update(st)
    assert st.audio.global_strength == pytest.approx(1.0)


def test_a_rig_load_recentres_the_audio_knobs_but_not_the_physics_ones(rig):
    rt, st, ports, _, _ = rig
    st.sim.SENSOR_GAIN = 0.3
    st.perform.recentre_knobs = True
    rt.update(st)
    _turn(ports[0], 0, +3)
    rt.update(st)
    twisted = st.sim.SENSOR_GAIN
    st.audio.global_strength = 0.5
    st.perform.recentre_audio_knobs = True
    rt.update(st)
    ports[0].queued.append((mc.SWITCH_CHANNEL, 0, mc.SWITCH_PRESSED))
    ports[0].queued.append((mc.SWITCH_CHANNEL, 2, mc.SWITCH_PRESSED))
    rt.update(st)
    assert st.sim.SENSOR_GAIN == pytest.approx(0.3), "physics centre untouched by a rig"
    assert st.audio.global_strength == pytest.approx(0.5)
    assert twisted != pytest.approx(0.3)


def test_rate_scale_steps_in_log_space(rig):
    rt, st, ports, _, _ = rig
    st.audio.rate_scale = 1.0
    rt.update(st)
    _turn(ports[0], 3, +1)
    rt.update(st)
    up = st.audio.rate_scale
    _turn(ports[0], 3, -2)
    rt.update(st)
    down = st.audio.rate_scale
    assert up / 1.0 == pytest.approx(1.0 / down), "one click up and one down are the same ratio"
    assert up > 1.0 > down
    _turn(ports[0], 3, -50)
    rt.update(st)
    assert st.audio.rate_scale == pytest.approx(0.05)


def test_connecting_with_no_preset_loaded_centres_on_what_is_on_screen(rig):
    rt, st, ports, _, _ = rig
    st.sim.SENSOR_GAIN = 0.2
    st.audio.global_strength = 0.7
    rt.update(st)
    _turn(ports[0], 0, +3)
    _turn(ports[0], 2, +3)
    rt.update(st)
    ports[0].queued.append((mc.SWITCH_CHANNEL, 0, mc.SWITCH_PRESSED))
    ports[0].queued.append((mc.SWITCH_CHANNEL, 2, mc.SWITCH_PRESSED))
    rt.update(st)
    assert st.sim.SENSOR_GAIN == pytest.approx(0.2)
    assert st.audio.global_strength == pytest.approx(0.7)


def test_an_audio_knob_ring_follows_the_setting_and_never_reads_as_bending(rig):
    rt, st, ports, _, _ = rig
    rt.update(st)
    st.audio.global_strength = 2.0
    ports[0].sent.clear()
    rt.feedback(st, st.sim)
    assert (mc.ENCODER_CHANNEL, 2, 127) in ports[0].sent
    assert (mc.SWITCH_CHANNEL, 2, st.perform.set_list.knobs[2].rest_colour) in ports[0].sent
