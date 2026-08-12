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
