"""The parameter log rides a take and lands beside it.

Row k must be video frame k, so the log is truncated to the frame count the
recorder reports rather than trusted to agree with it.
"""
from __future__ import annotations

import csv

import pytest

from services.param_log import ParamLog
from services.video_recorder import VideoRecorderService


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_the_log_lands_beside_the_video_with_one_row_per_frame(tmp_path):
    log = ParamLog()
    for i in range(3):
        log.row({"SENSOR_DISTANCE": 0.1 * i, "V_MAX": 0.5},
                {"bass": 0.2, "mid": 0.3})
    out = log.write(str(tmp_path / "take-01.mp4"), frames=3)
    assert out == tmp_path / "take-01_params.csv"
    rows = _read(out)
    assert [r["frame"] for r in rows] == ["0", "1", "2"]
    assert float(rows[2]["SENSOR_DISTANCE"]) == pytest.approx(0.2)
    assert float(rows[0]["sig:bass"]) == pytest.approx(0.2)


def test_the_log_is_cut_to_the_frames_the_recorder_wrote(tmp_path):
    """The readback runs a frame late, so a limited take logs one row more
    than it writes."""
    log = ParamLog()
    for i in range(5):
        log.row({"A": float(i)}, {})
    out = log.write(str(tmp_path / "t.mp4"), frames=4)
    assert len(_read(out)) == 4


def test_an_empty_log_writes_nothing(tmp_path):
    log = ParamLog()
    assert log.write(str(tmp_path / "t.mp4"), frames=0) is None
    assert list(tmp_path.iterdir()) == []


def test_writing_empties_the_log(tmp_path):
    log = ParamLog()
    log.row({"A": 1.0}, {})
    log.write(str(tmp_path / "t.mp4"), frames=1)
    assert len(log) == 0


class _Recorder:
    """VidSaver's surface as the service sees it, with no GL."""

    def __init__(self, tmp_path):
        self.active = False
        self.audio = None
        self.output_path = str(tmp_path / "take.mp4")
        self.last_frames = 0
        self.last_message = ""

    def finish(self):
        self.last_frames = 2
        self.active = False


@pytest.fixture
def service(tmp_path):
    s = VideoRecorderService()
    s.recorder = _Recorder(tmp_path)
    return s


def test_rows_are_ignored_unless_the_box_is_ticked(service):
    service.start()
    service.log_row({"A": 1.0}, {})
    assert len(service.param_log) == 0


def test_rows_are_ignored_while_no_take_is_running(service):
    service.configure(None, record_audio=False, log_parameters=True)
    service.log_row({"A": 1.0}, {})
    assert len(service.param_log) == 0


def test_a_ticked_take_writes_its_log_when_it_stops(service, tmp_path):
    service.configure(None, record_audio=False, log_parameters=True)
    service.start()
    for i in range(3):
        service.log_row({"A": float(i)}, {"bass": 0.1})
    service.stop()
    rows = _read(tmp_path / "take_params.csv")
    assert len(rows) == 2                      # cut to what the recorder wrote
    assert float(rows[1]["A"]) == pytest.approx(1.0)


def test_a_take_that_hits_its_limit_inside_the_frame_call_still_writes(
        service, tmp_path):
    """finish() runs from inside frame() at max_frames, not from stop()."""
    service.configure(None, record_audio=False, log_parameters=True)
    service.start()
    service.log_row({"A": 0.0}, {})
    service.recorder.finish()                   # what frame() does at the limit
    service.after_frame()
    assert (tmp_path / "take_params.csv").exists()


def test_a_new_take_starts_from_an_empty_log(service):
    service.configure(None, record_audio=False, log_parameters=True)
    service.start()
    service.log_row({"A": 0.0}, {})
    service.recorder.active = False
    service.recorder.last_frames = 0            # nothing written: log dropped
    service.after_frame()
    service.start()
    assert len(service.param_log) == 0


def test_the_preference_is_off_by_default_and_not_undoable():
    from state.preferences_state import NOT_UNDOABLE, PreferencesState
    assert PreferencesState().log_parameters is False
    assert "log_parameters" in NOT_UNDOABLE
