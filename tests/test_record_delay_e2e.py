"""The audio delay actually moves the sound, checked against real ffmpeg.

An argv test cannot catch a flag ffmpeg accepts and ignores, which is exactly
what happened: -itsoffset in front of a headerless raw input shifted nothing,
and the only thing left in the file was the AAC encoder's own priming delay.
So this one decodes the muxed audio back and finds the click.

Skipped where ffmpeg is absent, as the GL tests are skipped without a GPU.
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np
import pytest

from services.record_audio import RecordingAudio
from utilities.ffmpeg_recorder import FFmpegVideoRecorder, find_ffmpeg

SR, CH = 48000, 2
FRAMES, TAKE_SECONDS = 100, 2.0
CLICK_AT = 0.5
# AAC round-trips through a filterbank, so the click smears either side of its
# true position. Every delay under test is an order of magnitude larger.
TOLERANCE = 0.035


@pytest.fixture(scope="module")
def ffmpeg():
    try:
        return find_ffmpeg()
    except Exception as exc:
        pytest.skip(f"no ffmpeg: {exc}")


class _Capture:
    status, sample_rate, channels, tap = "active", float(SR), CH, None


def _click_lands_at(ffmpeg, offset, tmp_path):
    """Seconds into the muxed file where the click ends up."""
    video = str(tmp_path / f"t{offset}.video.mp4")
    out = str(tmp_path / f"t{offset}.mp4")

    rec = FFmpegVideoRecorder(width=64, height=64, fps=50, output_path=video,
                              realtime=True)
    for _ in range(FRAMES):
        rec.write_frame(bytes(64 * 64 * 4))
    rec.close()

    cap = _Capture()
    audio = RecordingAudio(cap, str(tmp_path / f"s{offset}.pcm"),
                           nominal_fps=50.0, audio_offset=offset)
    assert audio.start()
    mono = np.zeros(int(SR * TAKE_SECONDS), dtype="f4")
    mono[int(SR * CLICK_AT):int(SR * CLICK_AT) + 480] = 0.9
    cap.tap(np.repeat(mono, CH).tobytes())
    audio.finish(video, out, FRAMES, TAKE_SECONDS)
    assert os.path.exists(out)

    raw = subprocess.run(
        [ffmpeg, "-v", "error", "-i", out, "-map", "0:a",
         "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
        capture_output=True).stdout
    loud = np.nonzero(np.abs(np.frombuffer(raw, dtype="f4")) > 0.3)[0]
    assert loud.size, "the click did not survive the mux at all"
    return loud[0] / SR


def test_no_delay_leaves_the_click_where_it_was(ffmpeg, tmp_path):
    assert _click_lands_at(ffmpeg, 0.0, tmp_path) == pytest.approx(
        CLICK_AT, abs=TOLERANCE)


@pytest.mark.parametrize("delay", [0.10, 0.15, 0.20])
def test_the_delay_moves_the_sound_by_what_it_says(ffmpeg, tmp_path, delay):
    """The failure this guards is silent: the file plays, sounds fine alone,
    and is simply not shifted."""
    landed = _click_lands_at(ffmpeg, delay, tmp_path)
    assert landed == pytest.approx(CLICK_AT + delay, abs=TOLERANCE), (
        f"asked for {delay:.3f}s of delay, got "
        f"{landed - CLICK_AT:+.3f}s")


def test_bigger_delays_land_later_than_smaller_ones(ffmpeg, tmp_path):
    """Catches a shift that is real but not proportional - a constant offset
    would satisfy each case above on its own if the tolerance were loose."""
    small = _click_lands_at(ffmpeg, 0.05, tmp_path)
    large = _click_lands_at(ffmpeg, 0.30, tmp_path)
    assert large - small == pytest.approx(0.25, abs=TOLERANCE)
