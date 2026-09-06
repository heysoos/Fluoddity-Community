"""The Brain Inputs tab: one row per channel through the rig's own row
renderer, a stepper that is a layout change, and a drawer with a Channel tab.
"""
import numpy as np
import pytest

from services.audio_mapping import Channel, Mapping, channel_key
from services.brains import MAX_AUDIO_INPUTS
from tests.test_audio_reactive_window_render import _Snap, _host


def _channel_host(k=2, tab="channels"):
    imgui, host = _host(True)
    ast = host.state.audio
    ast.snapshot = _Snap()
    ast.status = "active"
    ast.enabled = True
    ast.tab = tab
    host.state.brain.settings = {"audio_inputs": k}
    return imgui, host


def _draw(imgui, host, frames=2):
    for _ in range(frames):
        imgui.new_frame()
        host.render_audio_reactive_window()
        imgui.end_frame()
        imgui.render()


def _spy_rows(host):
    seen = []
    original = host._render_audio_row

    def spy(ast, group, mappings, target, is_deaf):
        seen.append((group, target.key, target.label))
        return original(ast, group, mappings, target, is_deaf)
    host._render_audio_row = spy
    return seen


def _spy_labels(monkeypatch, imgui, *names):
    seen = []
    for name in names:
        original = getattr(imgui, name)

        def wrapped(label, *a, _o=original, **kw):
            seen.append(str(label).split("##")[0])
            return _o(label, *a, **kw)
        monkeypatch.setattr(imgui, name, wrapped)
    return seen


def test_the_tab_draws_one_row_per_input_through_the_shared_renderer():
    imgui, host = _channel_host(k=2)
    host.state.audio.channels = [Channel(name="kick")]
    seen = _spy_rows(host)
    _draw(imgui, host)
    rows = [s for s in seen if s[0] == "channel"]
    assert [(k, l) for _g, k, l in rows][-2:] == [
        (channel_key(0), "kick"), (channel_key(1), "A2")]


def test_the_modulate_tab_still_draws_the_physics_rows():
    imgui, host = _channel_host(k=2, tab="")
    seen = _spy_rows(host)
    _draw(imgui, host)
    assert any(g == "physics" for g, _k, _l in seen)


def test_a_deaf_brain_shows_the_stepper_and_no_rows(monkeypatch):
    imgui, host = _channel_host(k=0)
    seen = _spy_rows(host)
    labels = _spy_labels(monkeypatch, imgui, "button", "small_button")
    _draw(imgui, host)
    assert not [s for s in seen if s[0] == "channel"]
    assert "+" in labels and "-" in labels


def test_the_drawer_of_a_channel_row_has_a_channel_tab(monkeypatch):
    imgui, host = _channel_host(k=2)
    ast = host.state.audio
    ast.channel_mappings.append(Mapping(signal="bass",
                                        target=channel_key(0)))
    ast.open_target = channel_key(0)
    ast.open_band = ""
    ast.open_channel_tab = True
    labels = _spy_labels(monkeypatch, imgui, "input_text", "slider_float")
    # A tab bar honours a selection a frame late, and the drawer's bar sits
    # inside the panel's, so the Channel tab is drawn on the third frame.
    _draw(imgui, host, frames=3)
    assert "Name" in labels and "Range" in labels
    assert ast.open_channel_tab is False


# ---- the stepper is a layout change ---------------------------------------

def test_bumping_writes_the_brain_windows_setting():
    imgui, host = _channel_host(k=2)
    assert host._bump_audio_inputs(+1) is True
    assert host.state.brain.settings["audio_inputs"] == 3
    assert host._bump_audio_inputs(-2) is True
    assert host.state.brain.settings["audio_inputs"] == 1


def test_bumping_is_clamped():
    imgui, host = _channel_host(k=0)
    assert host._bump_audio_inputs(-1) is False
    assert host.state.brain.settings["audio_inputs"] == 0
    host.state.brain.settings["audio_inputs"] = MAX_AUDIO_INPUTS
    assert host._bump_audio_inputs(+1) is False
    assert host.state.brain.settings["audio_inputs"] == MAX_AUDIO_INPUTS


def test_bumping_is_refused_while_the_grid_or_a_borrow_owns_slot_zero():
    imgui, host = _channel_host(k=2)
    host.state.tournament.enabled = True
    assert host._audio_inputs_locked()
    assert host._bump_audio_inputs(+1) is False
    host.state.tournament.enabled = False
    host.state.brain.borrow_active = True
    assert host._audio_inputs_locked()
    assert host._bump_audio_inputs(+1) is False
    assert host.state.brain.settings["audio_inputs"] == 2


def test_editing_a_channel_pads_the_list_to_its_index():
    imgui, host = _channel_host(k=3)
    c = host._channel_at(host.state.audio, 2)
    assert isinstance(c, Channel)
    assert len(host.state.audio.channels) == 3
    assert host._channel_at(host.state.audio, 2) is c


# ---- the readout ----------------------------------------------------------

def test_channel_overlays_reach_the_panel_without_any_physics_mapping():
    from services.audio_runtime import AudioRuntime
    from services.brains import REGISTRY
    from services.brains.layout_moves import grow_inputs
    from state import UIState

    class Snap:
        signals = {"bass": 1.0}
        mel = np.zeros(40, np.float32)
        seq = 1

    st = UIState()
    st.audio.enabled = True
    st.audio.channels = [Channel(name="kick", range=0.5)]
    st.audio.channel_mappings.append(
        Mapping(signal="bass", target=channel_key(0), mode="add", depth=1.0))
    layout = grow_inputs(REGISTRY["fourier"].layout_from_settings({}), 1)
    rt = AudioRuntime()
    rt.capture._snapshot = Snap()
    sim_out, _ = rt.update(st, 1 / 60, layout, None)
    ov = rt.overlays(st, sim_out)
    assert channel_key(0) in ov
    o = ov[channel_key(0)]
    assert (o["lo"], o["hi"], o["base"]) == (-0.5, 0.5, 0.0)
    assert o["live"] == pytest.approx(0.5)
    assert o["reach"] == pytest.approx(0.5)
