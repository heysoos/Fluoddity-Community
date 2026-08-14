"""The soundtrack sidecar, and the clock the mux derives from it.

The tap runs on the audio callback thread, so its failure mode is the point:
nothing it does may reach that thread as an exception.
"""
from __future__ import annotations

import numpy as np
import pytest

from services.audio_track import AudioTrackWriter

SR = 48000
STEREO = 2
BYTES_PER_SECOND = 4 * STEREO * SR       # f32, interleaved


@pytest.fixture
def track(tmp_path):
    w = AudioTrackWriter(str(tmp_path / "take.pcm"), SR, STEREO)
    yield w
    w.close()


# --- accounting: the byte count IS the clock --------------------------------

def test_a_second_of_bytes_is_a_second_of_audio(track):
    track.write(b"\x00" * BYTES_PER_SECOND)
    assert track.duration_seconds == pytest.approx(1.0)


def test_duration_follows_the_declared_format(tmp_path):
    """Mono at a different rate must not be read with the stereo divisor."""
    w = AudioTrackWriter(str(tmp_path / "m.pcm"), 44100, 1)
    w.write(b"\x00" * (4 * 44100))
    assert w.duration_seconds == pytest.approx(1.0)
    w.close()


def test_an_empty_track_has_no_duration(track):
    assert track.duration_seconds == 0.0


def test_writes_accumulate(track):
    for _ in range(4):
        track.write(b"\x00" * (BYTES_PER_SECOND // 4))
    assert track.duration_seconds == pytest.approx(1.0)


def test_the_bytes_land_on_disk_unchanged(track, tmp_path):
    """ffmpeg reads this back as raw f32le, so any conversion here is a bug."""
    samples = np.linspace(-1.0, 1.0, 512, dtype="f4")
    track.write(samples.tobytes())
    track.close()
    back = np.fromfile(track.path, dtype="f4")
    assert np.array_equal(back, samples)


# --- failure: the audio thread must never see an exception ------------------

def test_a_write_that_fails_disables_the_tap_instead_of_raising(track):
    """A full disk mid-take must cost the soundtrack, not the recording."""
    def boom(_data):
        raise OSError("no space left on device")

    track._sink = boom
    track.write(b"\x00" * 16)
    assert track.ok is False
    assert track.last_error
    track.write(b"\x00" * 16)          # still silent afterwards


def test_writing_after_close_is_ignored(track):
    """The audio thread can be mid-callback when the take ends."""
    track.close()
    track.write(b"\x00" * 16)


def test_closing_twice_is_harmless(track):
    track.close()
    track.close()


def test_a_track_that_never_opened_reports_itself_as_unusable(tmp_path):
    w = AudioTrackWriter(str(tmp_path / "no" / "such" / "dir" / "x.pcm"),
                         SR, STEREO)
    assert w.ok is False
    w.write(b"\x00" * 16)
    w.close()
