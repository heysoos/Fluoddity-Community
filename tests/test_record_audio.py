"""Tying the capture tap to one take, and surviving a failed mux.

A take is expensive, so the invariant under test throughout is that the output
file exists afterwards no matter what went wrong with the soundtrack.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from services import record_audio as ra


class _Capture:
    """Enough of AudioCapture for the tap contract."""

    def __init__(self, status="active", rate=48000.0, channels=2):
        self.status = status
        self.sample_rate = rate
        self.channels = channels
        self.tap = None


@pytest.fixture
def paths(tmp_path):
    video = tmp_path / "take.video.mp4"
    video.write_bytes(b"fake encoded video")
    return video, tmp_path / "take.mp4"


@pytest.fixture
def muxed(monkeypatch):
    """Records mux calls and pretends they succeed."""
    calls = []

    def fake(cmd, out_path):
        calls.append((cmd, out_path))
        Path(out_path).write_bytes(b"muxed")
        return True, ""

    monkeypatch.setattr(ra, "run_mux", fake)
    return calls


# --- starting ---------------------------------------------------------------

def test_starting_installs_the_tap(tmp_path, muxed):
    cap = _Capture()
    audio = ra.RecordingAudio(cap, str(tmp_path / "s.pcm"))
    assert audio.start() is True
    assert cap.tap is not None


def test_a_silent_session_declines_rather_than_failing(tmp_path, muxed):
    """Audio off is not an error - the take proceeds without a soundtrack."""
    cap = _Capture(status="idle")
    audio = ra.RecordingAudio(cap, str(tmp_path / "s.pcm"))
    assert audio.start() is False
    assert cap.tap is None


def test_no_capture_at_all_declines(tmp_path, muxed):
    audio = ra.RecordingAudio(None, str(tmp_path / "s.pcm"))
    assert audio.start() is False


# --- finishing --------------------------------------------------------------

def test_finishing_muxes_and_produces_the_output(paths, tmp_path, muxed):
    video, out = paths
    cap = _Capture()
    audio = ra.RecordingAudio(cap, str(tmp_path / "s.pcm"))
    audio.start()
    cap.tap(b"\x00" * (4 * 2 * 48000))          # one second of stereo
    audio.finish(str(video), str(out), frame_count=30, wall_seconds=1.0)

    assert out.exists()
    assert len(muxed) == 1


def test_the_retime_matches_the_frames_against_the_audio(paths, tmp_path, muxed):
    """30 frames over one second of audio is 30fps; encoded at 50 it must be
    slowed by 50/30 or the soundtrack drifts apart from the picture."""
    video, out = paths
    cap = _Capture()
    audio = ra.RecordingAudio(cap, str(tmp_path / "s.pcm"), nominal_fps=50.0)
    audio.start()
    cap.tap(b"\x00" * (4 * 2 * 48000))
    audio.finish(str(video), str(out), frame_count=30, wall_seconds=1.0)

    cmd, _ = muxed[0]
    assert float(cmd[cmd.index("-itsscale") + 1]) == pytest.approx(50.0 / 30.0)


def test_the_sync_offset_reaches_the_mux(paths, tmp_path, muxed):
    """The picture lags the sound by the analysis block, the band smoother's
    release and a frame or two, so the soundtrack is delayed to meet it."""
    video, out = paths
    cap = _Capture()
    audio = ra.RecordingAudio(cap, str(tmp_path / "s.pcm"), audio_offset=0.15)
    audio.start()
    cap.tap(b"\x00" * (4 * 2 * 48000))
    audio.finish(str(video), str(out), frame_count=30, wall_seconds=1.0)

    cmd, _ = muxed[0]
    assert "delays=150.0" in cmd[cmd.index("-af") + 1]


def test_no_offset_is_the_default(paths, tmp_path, muxed):
    video, out = paths
    cap = _Capture()
    audio = ra.RecordingAudio(cap, str(tmp_path / "s.pcm"))
    audio.start()
    cap.tap(b"\x00" * 4096)
    audio.finish(str(video), str(out), frame_count=1, wall_seconds=1.0)
    assert "-af" not in muxed[0][0]


def test_finishing_removes_the_tap(paths, tmp_path, muxed):
    video, out = paths
    cap = _Capture()
    audio = ra.RecordingAudio(cap, str(tmp_path / "s.pcm"))
    audio.start()
    audio.finish(str(video), str(out), frame_count=1, wall_seconds=1.0)
    assert cap.tap is None, "the tap would keep writing into the next take"


def test_the_sidecar_is_deleted_once_it_is_muxed(paths, tmp_path, muxed):
    video, out = paths
    sidecar = tmp_path / "s.pcm"
    cap = _Capture()
    audio = ra.RecordingAudio(cap, str(sidecar))
    audio.start()
    cap.tap(b"\x00" * 4096)
    audio.finish(str(video), str(out), frame_count=1, wall_seconds=1.0)
    assert not sidecar.exists()


def test_the_temp_video_is_deleted_once_it_is_muxed(paths, tmp_path, muxed):
    video, out = paths
    cap = _Capture()
    audio = ra.RecordingAudio(cap, str(tmp_path / "s.pcm"))
    audio.start()
    cap.tap(b"\x00" * 4096)
    audio.finish(str(video), str(out), frame_count=1, wall_seconds=1.0)
    assert not video.exists()


# --- the take survives everything -------------------------------------------

def test_a_failed_mux_still_leaves_the_silent_take(paths, tmp_path, monkeypatch):
    """Losing a recording to an encoder error is not acceptable; the video is
    already finished and complete on its own."""
    monkeypatch.setattr(ra, "run_mux",
                        lambda cmd, out: (False, "codec exploded"))
    video, out = paths
    cap = _Capture()
    audio = ra.RecordingAudio(cap, str(tmp_path / "s.pcm"))
    audio.start()
    cap.tap(b"\x00" * 4096)
    message = audio.finish(str(video), str(out), frame_count=1,
                           wall_seconds=1.0)

    assert out.exists(), "the take was lost"
    assert out.read_bytes() == b"fake encoded video"
    assert "codec exploded" in message


def test_a_take_with_no_captured_audio_is_just_renamed(paths, tmp_path, muxed):
    """The device produced nothing, so there is nothing to mux."""
    video, out = paths
    cap = _Capture()
    audio = ra.RecordingAudio(cap, str(tmp_path / "s.pcm"))
    audio.start()
    audio.finish(str(video), str(out), frame_count=10, wall_seconds=1.0)
    assert out.exists()
    assert muxed == [], "an empty sidecar must not reach ffmpeg"


def test_a_truncated_stream_is_reported(paths, tmp_path, muxed):
    """Half a second of audio over a ten second take means the device died."""
    video, out = paths
    cap = _Capture()
    audio = ra.RecordingAudio(cap, str(tmp_path / "s.pcm"))
    audio.start()
    cap.tap(b"\x00" * (4 * 2 * 24000))          # half a second
    message = audio.finish(str(video), str(out), frame_count=300,
                           wall_seconds=10.0)
    assert "drift" in message.lower()


def test_finishing_without_starting_is_harmless(paths, tmp_path, muxed):
    video, out = paths
    audio = ra.RecordingAudio(_Capture(), str(tmp_path / "s.pcm"))
    audio.finish(str(video), str(out), frame_count=1, wall_seconds=1.0)
    assert out.exists()
