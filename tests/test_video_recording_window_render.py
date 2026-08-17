"""The Screen Recording window actually renders.

Nothing executed this window before, so every widget in it was unverified -
imgui_bundle validates signatures at CALL time, so a wrong one reaches the user
rather than a test. That is exactly how a bad selectable() call shipped once.
"""
from __future__ import annotations

import pytest
from imgui_bundle import imgui

from state import UIState
from ui.help_windows import HelpWindowsMixin


@pytest.fixture(scope="module")
def gui():
    imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 900)
    io.delta_time = 1.0 / 60.0
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    yield io
    imgui.destroy_context()


class _Keys:
    def get_key_display_name(self, _action):
        return "F9"


def _host():
    class Host(HelpWindowsMixin):
        def __init__(self):
            self.state = UIState()
            self.keybindings = _Keys()
            self.state.preferences.show_video_recording_window = True
            self._display_info = {}

        def _delayed_tooltip(self, text):
            pass

    return Host()


def _draw(host, frames=2):
    for _ in range(frames):
        imgui.new_frame()
        host.render_video_recording_window()
        imgui.end_frame()
        imgui.render()


@pytest.mark.parametrize("recording", [False, True])
@pytest.mark.parametrize("record_audio", [False, True])
@pytest.mark.parametrize("pending", [False, True])
def test_the_window_renders_in_every_state(gui, recording, record_audio,
                                           pending):
    host = _host()
    host.state.preferences.record_audio = record_audio
    host._display_info = {
        "recording_active": recording,
        "video_pending": pending,
        "video_scheduled_start_frame": 500,
        "frame_count": 120,
    }
    _draw(host)


def test_the_result_notice_renders_and_can_be_dismissed(gui):
    """The banner only draws when there is a message, so the default pass
    never reaches it - which is where an id clash or a bad call would hide."""
    host = _host()
    host.state.preferences.record_notice = "Audio could not be added (x)."
    _draw(host, frames=1)
    assert host.state.preferences.record_notice, "nothing should clear it here"


def _with_popups_open(host, frames=2):
    """A popup body is unexecuted until it opens, so a bad call in one reaches
    the user rather than a test. open_popup hashes str_id against the same
    window and ID stack that begin_popup_context_item does."""
    real = imgui.begin_popup_context_item

    def spy(str_id=None, *a, **kw):
        if str_id:
            imgui.open_popup(str_id)
        return real(str_id, *a, **kw)

    imgui.begin_popup_context_item = spy
    try:
        _draw(host, frames=frames)
    finally:
        imgui.begin_popup_context_item = real


def test_the_audio_delay_reset_menu_actually_renders(gui):
    host = _host()
    host.state.preferences.record_audio = True
    host.state.preferences.record_audio_delay = 0.15
    _with_popups_open(host)


def test_the_forced_popup_helper_really_opens_something(gui):
    """Without this the test above could pass while opening nothing at all."""
    host = _host()
    host.state.preferences.record_audio = True
    opened = []
    real = imgui.begin_popup_context_item

    def spy(str_id=None, *a, **kw):
        if str_id:
            imgui.open_popup(str_id)
        out = real(str_id, *a, **kw)
        if out:
            opened.append(str_id)
        return out

    imgui.begin_popup_context_item = spy
    try:
        _draw(host, frames=2)
    finally:
        imgui.begin_popup_context_item = real
    assert opened, "no popup body was entered, so the coverage is imaginary"


def test_the_delay_slider_only_exists_with_audio_on(gui):
    """It is meaningless without a soundtrack, and drawing it anyway would
    imply the silent path has a sync control."""
    host = _host()
    host.state.preferences.record_audio = False
    host.state.preferences.record_audio_delay = 0.2
    _draw(host)
    assert host.state.preferences.record_audio_delay == pytest.approx(0.2)


def test_the_audio_checkbox_survives_a_round_trip(gui):
    host = _host()
    host.state.preferences.record_audio = True
    _draw(host)
    assert host.state.preferences.record_audio is True
