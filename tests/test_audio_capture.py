"""Capture is IO, so the tests cover only what can be checked without a device:
availability, graceful absence, and that nothing imports at module scope."""
import ast
from pathlib import Path

import pytest

from services import audio_capture

SRC = Path(__file__).resolve().parent.parent / "services" / "audio_capture.py"


def test_pyaudiowpatch_is_never_imported_at_module_scope():
    """CLAUDE.md: optional dependencies must stay out of the startup path."""
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    for node in tree.body:                       # module level only
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names]
            mod = getattr(node, "module", "") or ""
            assert "pyaudio" not in mod.lower()
            assert not any("pyaudio" in n.lower() for n in names)


def test_is_available_answers_without_raising():
    assert isinstance(audio_capture.is_available(), bool)


def test_listing_devices_never_raises():
    devices = audio_capture.list_devices()
    assert isinstance(devices, list)
    for d in devices:
        assert set(d) >= {"index", "name", "loopback", "rate"}


def test_a_fresh_capture_is_idle_with_no_snapshot():
    cap = audio_capture.AudioCapture()
    assert cap.status == "idle"
    assert cap.snapshot() is None


def test_stopping_a_capture_that_never_started_is_harmless():
    audio_capture.AudioCapture().stop()


def test_starting_without_the_package_reports_an_error_rather_than_raising(monkeypatch):
    monkeypatch.setattr(audio_capture, "is_available", lambda: False)
    cap = audio_capture.AudioCapture()
    assert cap.start(None, auto_gain=True) is False
    assert cap.status == "error"
    assert cap.last_error


def test_a_bad_device_index_reports_an_error_rather_than_raising():
    cap = audio_capture.AudioCapture()
    if not audio_capture.is_available():
        pytest.skip("PyAudioWPatch not installed")
    assert cap.start(999999, auto_gain=True) is False
    assert cap.status == "error"
    cap.stop()


# --- the recording tap, which runs on the callback thread -------------------

def _listening_capture(channels=2):
    """A capture wired for analysis without a device, as _on_block sees it."""
    from services.audio_analysis import Analyzer
    cap = audio_capture.AudioCapture()
    cap._analyzer = Analyzer(48000.0, auto_gain=False)
    cap._channels = channels
    return cap


def _block(cap, frames=512):
    import numpy as np
    data = np.zeros(frames * cap._channels, dtype="f4")
    return data.tobytes()


def test_the_tap_receives_the_device_bytes_verbatim():
    """ffmpeg reads the file as raw interleaved f32le, so the mono mix the
    analyser makes for itself must not reach it."""
    cap = _listening_capture()
    seen = []
    cap.tap = seen.append
    raw = _block(cap)
    cap._on_block(raw, 512, None, 0)
    assert seen == [raw]


def test_analysis_still_happens_while_tapping():
    cap = _listening_capture()
    cap.tap = lambda _data: None
    cap._on_block(_block(cap), 512, None, 0)
    assert cap.snapshot() is not None


def test_a_raising_tap_does_not_cost_the_analysis():
    """The tap is the newcomer here; the signals driving the sim are not
    allowed to stop because a disk filled up."""
    cap = _listening_capture()

    def boom(_data):
        raise OSError("no space left on device")

    cap.tap = boom
    cap._on_block(_block(cap), 512, None, 0)
    assert cap.snapshot() is not None, "a failing tap silenced the analyser"


def test_a_raising_tap_never_propagates_into_the_audio_thread():
    cap = _listening_capture()
    cap.tap = lambda _data: 1 / 0
    cap._on_block(_block(cap), 512, None, 0)   # must not raise


def test_stopping_clears_the_tap_before_teardown():
    """A callback still in flight would otherwise write into a file the main
    thread is closing."""
    cap = _listening_capture()
    cap.tap = lambda _data: None
    cap.stop()
    assert cap.tap is None


def test_no_tap_is_the_default():
    assert audio_capture.AudioCapture().tap is None


def test_the_channel_count_is_public():
    """The sidecar is headerless, so whoever writes it has to declare the
    interleaving; reaching into _channels for that would be luck."""
    cap = _listening_capture(channels=2)
    assert cap.channels == 2
