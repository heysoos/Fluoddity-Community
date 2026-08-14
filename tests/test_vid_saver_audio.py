"""VidSaver hands the take to the soundtrack, and only when there is one."""
from __future__ import annotations

import numpy as np
import pytest

from utilities import vid_saver as vs


class _Tex:
    def __init__(self, size):
        self.size = size


class _FakeRecorder:
    instances: list = []

    def __init__(self, width, height, fps=50, output_path=None, **kw):
        self.input_width, self.input_height = width, height
        self.output_path = output_path
        self.frames = 0
        _FakeRecorder.instances.append(self)

    def write_frame_from_array(self, arr):
        self.frames += 1

    def close(self):
        pass


class _FakeAudio:
    """Stands in for RecordingAudio."""

    def __init__(self, starts=True):
        self.starts = starts
        self.started = False
        self.finished = None

    def start(self):
        self.started = True
        return self.starts

    def finish(self, video_path, out_path, frame_count, wall_seconds):
        self.finished = (video_path, out_path, frame_count, wall_seconds)
        return "done"


@pytest.fixture
def saver(monkeypatch):
    _FakeRecorder.instances = []
    monkeypatch.setattr(vs, "FFmpegVideoRecorder", _FakeRecorder)
    monkeypatch.setattr(vs, "reset_gpu_frame_counter", lambda: None)
    monkeypatch.setattr(
        vs, "save_frame_gpu",
        lambda tex, ctx, supersample_k=1, return_array=True: np.zeros(
            (tex.size[1] // supersample_k, tex.size[0] // supersample_k, 3),
            dtype=np.uint8))
    clock = {"t": 100.0}
    monkeypatch.setattr(vs, "perf_counter", lambda: clock["t"])
    s = vs.VidSaver()
    s.active = True
    s.clock = clock
    return s


def _run(saver, frames=3):
    for _ in range(frames):
        saver.frame(None, _Tex((800, 600)), ssk_w=1)


def test_with_audio_the_encoder_writes_somewhere_temporary(saver):
    """The final name belongs to the muxed file; ffmpeg must not have taken it
    already, or the mux would be asked to overwrite its own input."""
    saver.audio = _FakeAudio()
    _run(saver, 1)
    written = _FakeRecorder.instances[0].output_path
    assert written != saver.output_path
    assert written.endswith(".mp4")


def test_the_soundtrack_is_told_how_many_frames_were_written(saver):
    """That count is half the clock - frames over audio seconds is the rate."""
    audio = _FakeAudio()
    saver.audio = audio
    _run(saver, 7)
    saver.finish()
    assert audio.finished[2] == 7


def test_the_soundtrack_is_told_how_long_the_take_ran(saver):
    audio = _FakeAudio()
    saver.audio = audio
    _run(saver, 1)
    saver.clock["t"] = 112.5
    saver.finish()
    assert audio.finished[3] == pytest.approx(12.5)


def test_the_mux_target_is_the_name_the_user_expects(saver):
    audio = _FakeAudio()
    saver.audio = audio
    _run(saver, 1)
    saver.finish()
    video_path, out_path, _frames, _wall = audio.finished
    assert out_path == saver.output_path
    assert video_path == _FakeRecorder.instances[0].output_path


def test_a_declined_soundtrack_writes_straight_to_the_final_name(saver):
    """Audio off is the common case and must not leave a temp file behind."""
    saver.audio = _FakeAudio(starts=False)
    _run(saver, 1)
    assert _FakeRecorder.instances[0].output_path == saver.output_path


def test_a_declined_soundtrack_is_not_asked_to_mux(saver):
    audio = _FakeAudio(starts=False)
    saver.audio = audio
    _run(saver, 1)
    saver.finish()
    assert audio.finished is None


def test_without_a_soundtrack_nothing_changes(saver):
    """The existing silent path must stay exactly as it was."""
    _run(saver, 2)
    assert _FakeRecorder.instances[0].output_path == saver.output_path
    saver.finish()


def test_hitting_the_frame_limit_still_muxes(saver):
    """max_frames stops the take from inside frame(), not from the UI."""
    audio = _FakeAudio()
    saver.audio = audio
    for _ in range(4):
        saver.frame(None, _Tex((800, 600)), max_frames=2, ssk_w=1)
    assert audio.finished is not None
    assert audio.finished[2] == 2


def test_the_soundtrack_starts_once_per_take(saver):
    audio = _FakeAudio()
    saver.audio = audio
    _run(saver, 5)
    assert audio.started
    assert len(_FakeRecorder.instances) == 1


def test_the_message_from_the_mux_is_kept_for_the_ui(saver):
    """A save the user cannot see reads as a no-op."""
    saver.audio = _FakeAudio()
    _run(saver, 1)
    saver.finish()
    assert saver.last_message == "done"
