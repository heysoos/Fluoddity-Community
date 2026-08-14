"""The recording's clock, and the command that muxes the soundtrack on.

The real framerate is not knowable until a take ends - ffmpeg's -framerate is
an input option, fixed when the encoder starts - so the video is encoded at a
nominal rate and retimed during the mux. Getting that scale backwards is
silent: the file still plays, just at the wrong speed with drifting audio.
"""
from __future__ import annotations

import pytest

from utilities.ffmpeg_recorder import mux_command, recording_fps

NOMINAL = 50.0


# --- the clock --------------------------------------------------------------

def test_the_audio_length_sets_the_framerate():
    """Both streams cover the same wall interval by construction, so frames
    divided by audio seconds IS the rate that makes them equal."""
    fps, warning = recording_fps(frame_count=600, audio_seconds=20.0,
                                 wall_seconds=20.0, nominal_fps=NOMINAL)
    assert fps == pytest.approx(30.0)
    assert warning == ""


def test_the_sound_card_clock_wins_over_the_wall_clock():
    """Small disagreement is normal; the audio device's clock is the steadier
    of the two and is what the samples are actually paced by."""
    fps, warning = recording_fps(600, audio_seconds=20.0, wall_seconds=20.4,
                                 nominal_fps=NOMINAL)
    assert fps == pytest.approx(30.0)
    assert warning == ""


def test_a_truncated_stream_falls_back_to_the_wall_clock():
    """A device unplugged halfway leaves audio far shorter than the take; the
    derived rate would double the video's speed to match it."""
    fps, warning = recording_fps(600, audio_seconds=10.0, wall_seconds=20.0,
                                 nominal_fps=NOMINAL)
    assert fps == pytest.approx(30.0)
    assert warning, "a silent 2x speed-up is exactly what must be reported"


def test_no_audio_keeps_the_nominal_rate():
    """A silent recording must behave precisely as it did before this feature."""
    fps, warning = recording_fps(600, audio_seconds=0.0, wall_seconds=20.0,
                                 nominal_fps=NOMINAL)
    assert fps == NOMINAL
    assert warning == ""


def test_a_take_with_no_frames_does_not_divide_by_zero():
    fps, _warning = recording_fps(0, audio_seconds=0.0, wall_seconds=0.0,
                                  nominal_fps=NOMINAL)
    assert fps == NOMINAL


def test_a_very_short_take_is_not_called_a_mismatch():
    """The tap and the first frame cannot start on the same instant, so a
    one-block offset must not read as a dead device."""
    fps, warning = recording_fps(30, audio_seconds=0.98, wall_seconds=1.0,
                                 nominal_fps=NOMINAL)
    assert warning == ""
    assert fps == pytest.approx(30 / 0.98)


# --- the command ------------------------------------------------------------

def _cmd(**kw):
    args = dict(ffmpeg="ffmpeg", video_path="v.mp4", audio_path="a.pcm",
                out_path="out.mp4", itsscale=1.0, sample_rate=48000,
                channels=2, audio_offset=0.0)
    args.update(kw)
    return mux_command(**args)


def test_the_retime_applies_to_the_video_input():
    """-itsscale is an INPUT option and only affects the input it precedes;
    after the -i it would be ignored and the video would keep its wrong rate."""
    cmd = _cmd(itsscale=1.6666)
    assert cmd.index("-itsscale") < cmd.index("v.mp4")
    assert cmd[cmd.index("-itsscale") + 1].startswith("1.666")


def test_the_video_is_stream_copied():
    """Re-encoding a finished take would cost a second generation of quality
    for nothing - the pixels are already right."""
    cmd = _cmd()
    assert cmd[cmd.index("-c:v") + 1] == "copy"


def test_the_raw_audio_format_is_declared_before_its_input():
    """Headerless f32le carries no rate or channel count, so ffmpeg cannot
    guess; declared after the -i these apply to the wrong thing."""
    cmd = _cmd(sample_rate=44100, channels=1)
    i = cmd.index("a.pcm")
    assert cmd[cmd.index("-f") + 1] == "f32le"
    assert cmd[cmd.index("-ar") + 1] == "44100"
    assert cmd[cmd.index("-ac") + 1] == "1"
    for flag in ("-f", "-ar", "-ac"):
        assert cmd.index(flag) < i


def test_the_audio_is_encoded_rather_than_copied():
    assert _cmd()[_cmd().index("-c:a") + 1] == "aac"


def test_the_output_is_last():
    assert _cmd()[-1] == "out.mp4"


def test_the_streams_are_cut_to_the_shorter_one():
    """Retiming makes them nearly equal, and a trailing block of audio over a
    frozen last frame is the artefact of not doing this."""
    assert "-shortest" in _cmd()


def test_an_existing_file_is_overwritten_without_a_prompt():
    """ffmpeg blocks on a y/n prompt otherwise, and nothing is reading stdin."""
    assert "-y" in _cmd()


# --- the sync delay ---------------------------------------------------------
#
# These assert SHAPE only. That an argv is well formed says nothing about
# whether ffmpeg acts on it - -itsoffset was accepted here and silently did
# nothing. tests/test_record_delay_e2e.py is what proves the sound moves.

def test_no_delay_leaves_the_command_exactly_as_it_was():
    """A delay of zero is not a shift, and every take made before this control
    existed has to keep muxing identically."""
    assert "-af" not in _cmd(audio_offset=0.0)


def test_the_delay_is_a_filter_rather_than_an_input_offset():
    """-itsoffset in front of a headerless raw input shifts nothing at all."""
    cmd = _cmd(audio_offset=0.15)
    assert "-itsoffset" not in cmd
    assert "adelay" in cmd[cmd.index("-af") + 1]


def test_the_delay_is_expressed_in_milliseconds():
    """adelay takes ms; handing it seconds would under-delay by 1000x."""
    cmd = _cmd(audio_offset=0.15)
    assert "delays=150.0" in cmd[cmd.index("-af") + 1]


def test_the_delay_covers_every_channel():
    """Without all=1 only the first channel moves, which reads as the sound
    coming apart rather than as a sync control."""
    assert "all=1" in _cmd(audio_offset=0.15)[
        _cmd(audio_offset=0.15).index("-af") + 1]


def test_the_delay_does_not_disturb_the_retime():
    cmd = _cmd(audio_offset=0.15, itsscale=1.6666)
    assert cmd.index("-itsscale") < cmd.index("v.mp4")
    assert cmd[cmd.index("-c:v") + 1] == "copy"
