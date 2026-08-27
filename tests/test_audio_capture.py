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


# --- one row per device ------------------------------------------------------
# Windows lists the same microphone under every host API it supports, and two
# of those copies carry byte-identical names - which a name-keyed selection
# cannot tell apart. These drive the dedup as a pure function over a table, so
# they say the same thing on a machine with no sound card.

def _row(index, name, rank, loopback=False):
    return {"index": index, "name": name, "loopback": loopback,
            "rate": 48000.0, "api": "", "rank": rank}


WASAPI, DSOUND, MME = 0, 2, 3


def test_a_device_listed_under_every_host_api_is_offered_once():
    kept = audio_capture.dedupe_devices([
        _row(1, "Microphone Array (Intel Smart Sound Technology)", MME),
        _row(11, "Microphone Array (Intel Smart Sound Technology)", DSOUND),
        _row(27, "Microphone Array (Intel Smart Sound Technology)", WASAPI),
    ])
    assert len(kept) == 1
    assert kept[0]["index"] == 27, "the WASAPI copy is the one worth keeping"


def test_a_name_cut_short_by_mme_names_the_same_device():
    """MME caps a name at 31 characters, so its row looks like a different
    device until the truncation is allowed for."""
    full = "CABLE Output (VB-Audio Virtual Cable)"
    kept = audio_capture.dedupe_devices([
        _row(3, full[:31], MME),
        _row(26, full, WASAPI),
    ])
    assert [d["index"] for d in kept] == [26]


def test_a_short_name_that_merely_starts_alike_is_a_different_device():
    """The truncation rule may not pair two devices that only share a prefix:
    a name under the MME limit was never cut, so it has to match in full."""
    kept = audio_capture.dedupe_devices([
        _row(1, "Microphone", MME),
        _row(27, "Microphone Array (Intel)", WASAPI),
    ])
    assert len(kept) == 2


def test_a_device_no_better_host_api_offers_is_kept():
    kept = audio_capture.dedupe_devices([
        _row(0, "Microsoft Sound Mapper - Input", MME),
        _row(27, "Microphone Array", WASAPI),
    ])
    assert {d["name"] for d in kept} == {"Microsoft Sound Mapper - Input",
                                         "Microphone Array"}


def test_a_loopback_is_never_the_same_device_as_an_input():
    kept = audio_capture.dedupe_devices([
        _row(30, "Speakers (Realtek)", WASAPI, loopback=True),
        _row(22, "Speakers (Realtek)", WASAPI),
    ])
    assert len(kept) == 2


def test_loopbacks_come_first():
    """Every saved rig points at one, so they are what the list is for."""
    kept = audio_capture.dedupe_devices([
        _row(27, "Microphone Array", WASAPI),
        _row(30, "Speakers [Loopback]", WASAPI, loopback=True),
    ])
    assert [d["loopback"] for d in kept] == [True, False]


def test_no_two_devices_offered_share_a_name():
    """The selection is keyed by name in the combo and again at Start, so a
    repeated name makes the second row unselectable."""
    names = [d["name"] for d in audio_capture.list_devices()]
    assert len(names) == len(set(names)), f"duplicate names: {names}"


# --- (Default) as a choice ---------------------------------------------------

def test_the_default_device_is_offered_as_a_row():
    """Rather than being what nothing-selected happens to fall back to."""
    choices = audio_capture.device_choices()
    assert choices[0]["name"] == audio_capture.DEFAULT_DEVICE_NAME
    assert choices[0]["index"] is None, "no index: resolved when Start is hit"


def test_the_default_row_is_offered_even_with_no_devices(monkeypatch):
    monkeypatch.setattr(audio_capture, "list_devices", lambda: [])
    assert len(audio_capture.device_choices()) == 1


def test_the_real_devices_follow_the_default_row():
    monkey = [{"index": 7, "name": "Mic", "loopback": False, "rate": 48000.0}]
    import unittest.mock as m
    with m.patch.object(audio_capture, "list_devices", lambda: monkey):
        assert [d["name"] for d in audio_capture.device_choices()] == \
            [audio_capture.DEFAULT_DEVICE_NAME, "Mic"]


# --- a device that has gone --------------------------------------------------

def test_a_capture_can_report_a_device_it_could_not_resolve():
    """Named back rather than silently replaced, as an unfindable shader is."""
    cap = audio_capture.AudioCapture()
    cap.fail("Microphone (K66) is not available.")
    assert cap.status == "error"
    assert "K66" in cap.last_error
