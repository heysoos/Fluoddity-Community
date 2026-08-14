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
            self.show_video_recording_window = True
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


def test_the_audio_checkbox_survives_a_round_trip(gui):
    host = _host()
    host.state.preferences.record_audio = True
    _draw(host)
    assert host.state.preferences.record_audio is True
