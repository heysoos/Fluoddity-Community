"""A slow frame is recorded with where its time went; a fast one costs nothing."""
from services.frame_watch import FrameWatch


class _Clock:
    def __init__(self, *ticks):
        self.ticks = list(ticks)

    def __call__(self):
        return self.ticks.pop(0)


def _frame(watch, clock_ticks, flags=""):
    watch._clock = _Clock(*clock_ticks)
    watch.begin()
    watch.mark("state")
    watch.mark("commands")
    watch.mark("sim")
    return watch.end(flags)


def test_a_fast_frame_records_nothing():
    w = FrameWatch(threshold_ms=60.0)
    assert _frame(w, [0.0, 0.001, 0.002, 0.010, 0.012]) is None
    assert w.recent == [] and w.frames == 1 and w.stalls == 0


def test_a_slow_frame_is_kept_with_its_phases_and_flags():
    w = FrameWatch(threshold_ms=60.0)
    st = _frame(w, [0.0, 0.001, 0.201, 0.211, 0.212], flags="explore")
    assert st is not None and w.stalls == 1
    assert abs(st.total_ms - 212.0) < 1e-6
    got = dict(st.phases)
    assert set(got) == {"state", "commands", "sim"}
    assert abs(got["commands"] - 200.0) < 1e-6 and abs(got["sim"] - 10.0) < 1e-6
    assert "commands 200" in st.line() and "[explore]" in st.line()
    assert "state 1" in st.line()


def test_sub_millisecond_phases_stay_out_of_the_line():
    w = FrameWatch(threshold_ms=10.0)
    st = _frame(w, [0.0, 0.0001, 0.0500, 0.0501, 0.0502])
    assert "state" not in st.line() and "commands 50" in st.line()


def test_the_log_file_gets_one_line_per_stall(tmp_path):
    log = tmp_path / "perf.log"
    w = FrameWatch(threshold_ms=10.0, path=log)
    _frame(w, [0.0, 0.0, 0.0, 0.0, 0.001])
    _frame(w, [0.0, 0.0, 0.0, 0.0, 0.100])
    _frame(w, [0.0, 0.0, 0.0, 0.0, 0.200])
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "100 ms" in lines[0] and "200 ms" in lines[1]


def test_only_the_last_keep_stalls_are_held():
    w = FrameWatch(threshold_ms=10.0, keep=3)
    for i in range(6):
        _frame(w, [0.0, 0.0, 0.0, 0.0, 0.020 + i * 0.001])
    assert len(w.recent) == 3
    assert w.stalls == 6
    assert abs(w.recent[-1].total_ms - 25.0) < 1e-6


def test_an_unwritable_log_never_costs_the_frame(tmp_path):
    w = FrameWatch(threshold_ms=10.0, path=tmp_path / "missing" / "perf.log")
    assert _frame(w, [0.0, 0.0, 0.0, 0.0, 0.5]) is not None


def test_marks_and_end_without_begin_are_harmless():
    w = FrameWatch()
    w.mark("x")
    assert w.end() is None
