"""The service decides whether a take gets a soundtrack.

Recording starts from two places - the toggle and the scheduled-start check -
so the choice is pushed in every frame rather than passed at the call site.
"""
from __future__ import annotations

import pytest

from services.video_recorder import VideoRecorderService


class _Capture:
    def __init__(self, status="active"):
        self.status = status
        self.sample_rate = 48000.0
        self.channels = 2
        self.tap = None


@pytest.fixture
def service():
    return VideoRecorderService()


def test_a_take_gets_no_soundtrack_by_default(service):
    """Nothing about the existing silent workflow may change until asked."""
    service.start()
    assert service.recorder.audio is None


def test_asking_for_audio_attaches_a_soundtrack(service):
    service.configure(_Capture(), record_audio=True)
    service.start()
    assert service.recorder.audio is not None


def test_audio_off_attaches_nothing_even_with_a_live_capture(service):
    service.configure(_Capture(), record_audio=False)
    service.start()
    assert service.recorder.audio is None


def test_no_capture_attaches_nothing(service):
    """Audio input off is not an error; the take is simply silent."""
    service.configure(None, record_audio=True)
    service.start()
    assert service.recorder.audio is None


def test_the_soundtrack_is_pointed_at_the_live_capture(service):
    cap = _Capture()
    service.configure(cap, record_audio=True)
    service.start()
    assert service.recorder.audio.capture is cap


def test_starting_an_active_take_does_not_replace_its_soundtrack(service):
    """The scheduled-start check runs every frame and would otherwise swap the
    sidecar out from under a running take."""
    service.configure(_Capture(), record_audio=True)
    service.start()
    first = service.recorder.audio
    service.start()
    assert service.recorder.audio is first


def test_a_stopped_take_does_not_leave_its_soundtrack_behind(service):
    service.configure(_Capture(), record_audio=True)
    service.start()
    service.stop()
    assert service.recorder.audio is None


def test_the_result_of_a_take_is_reported_once(service):
    """A save the user cannot see reads as a no-op, and a notice that never
    clears would outlive the take it describes."""
    service.recorder.last_message = "Audio could not be added"
    assert service.take_message() == "Audio could not be added"
    assert service.take_message() == ""


def test_a_clean_take_reports_nothing(service):
    assert service.take_message() == ""


def test_each_take_gets_its_own_sidecar_path(service):
    """Two takes sharing one path would have the second overwrite the first."""
    cap = _Capture()
    service.configure(cap, record_audio=True)
    service.start()
    first = service.recorder.audio.sidecar_path
    service.stop()
    service.start()
    assert service.recorder.audio.sidecar_path != first
