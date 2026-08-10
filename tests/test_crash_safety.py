"""A crash in the frame loop must not cost the run.

An unhandled exception used to propagate straight out of run(), so cleanup()
never ran. That lost the archive entries admitted since the last 200-admission
vector flush, the goal list, the settings, AND any record of what happened -
the user saw only a screenful of `Texture.__del__` errors from interpreter
teardown, with the real traceback scrolled off the top.
"""
import pytest

import main as main_mod
from main import App


class _FakeGlfw:
    """Enough glfw for run(): a loop that ends when told."""

    def __init__(self, frames=3):
        self.frames = frames
        self.polled = 0
        self.terminated = False

    def window_should_close(self, _w):
        return self.frames <= 0

    def poll_events(self):
        self.polled += 1

    def swap_buffers(self, _w):
        self.frames -= 1

    def terminate(self):
        self.terminated = True


class _App:
    """A stand-in wearing the REAL shutdown methods.

    Borrowed rather than reimplemented: the ordering and the guards inside
    cleanup are exactly what these tests exist to pin down.
    """

    run = App.run
    cleanup = App.cleanup
    _step = staticmethod(App._step)
    _write_crash_log = App._write_crash_log
    _cleanup_safely = App._cleanup_safely

    def __init__(self, boom=None):
        self.window = object()
        self.log = []
        self._boom = boom
        self.archive = None
        self.goal_list = None
        self.archive_store = None
        self.thumb_cache = None
        self.auto_service = None
        self.ui = self
        self.advanced_drawing_processor = self
        self.video_service = self

    # -- the frame loop
    def orchestrate_frame(self):
        self.log.append("frame")
        if self._boom is not None:
            raise self._boom

    # -- things cleanup reaches for
    def get_state(self):
        self.log.append("get_state")
        return type("UI", (), {"preferences": object(),
                               "archive": object()})()

    def _restore_auto_overrides(self, _ui):
        self.log.append("restore")

    def _save_archive_settings(self, _ui):
        self.log.append("settings")



def _stub(**fns):
    """An object whose attributes are plain callables.

    A lambda in a `type()` class body becomes a METHOD, so it would be handed
    self and raise TypeError - which the cleanup guard then reports as a
    failure, hiding whatever the test meant to check.
    """
    return type("Stub", (), {k: staticmethod(v) for k, v in fns.items()})()


def _wire(monkeypatch, tmp_path, app, frames=3):
    fake = _FakeGlfw(frames)
    monkeypatch.setattr(main_mod, "glfw", fake)
    monkeypatch.setattr(main_mod, "save_preferences",
                        lambda p: app.log.append("prefs"))
    import utilities.paths as paths
    monkeypatch.setattr(paths, "get_user_data_dir", lambda: tmp_path)
    # cleanup() calls self.ui.cleanup() etc; App.cleanup is the real one, so
    # the subsystems must not recurse into it.
    app.ui = _stub(get_state=app.get_state,
                   cleanup=lambda: app.log.append("ui_cleanup"))
    app.advanced_drawing_processor = _stub(
        cleanup=lambda: app.log.append("draw_cleanup"))
    app.video_service = _stub(cleanup=lambda: app.log.append("video_cleanup"))
    return fake


def _run(app):
    return App.run(app)


# ---- the normal path is unchanged ---------------------------------------

def test_a_normal_exit_still_runs_cleanup(monkeypatch, tmp_path):
    app = _App()
    fake = _wire(monkeypatch, tmp_path, app)
    _run(app)
    assert app.log.count("frame") == 3
    assert "prefs" in app.log
    assert fake.terminated is True


def test_a_normal_exit_writes_no_crash_log(monkeypatch, tmp_path):
    app = _App()
    _wire(monkeypatch, tmp_path, app)
    _run(app)
    assert not (tmp_path / "crash.log").exists()


# ---- the crash path ------------------------------------------------------

def test_a_crash_still_runs_cleanup(monkeypatch, tmp_path):
    """The whole point: the archive flush and the goal list are inside
    cleanup, and they are hours of work."""
    app = _App(boom=RuntimeError("gpu went away"))
    _wire(monkeypatch, tmp_path, app)
    with pytest.raises(RuntimeError):
        _run(app)
    assert "prefs" in app.log
    assert "settings" in app.log


def test_a_crash_is_re_raised(monkeypatch, tmp_path):
    """Swallowing it would turn a crash into a silent, wrong success."""
    app = _App(boom=ValueError("nope"))
    _wire(monkeypatch, tmp_path, app)
    with pytest.raises(ValueError, match="nope"):
        _run(app)


def test_the_traceback_lands_in_a_file(monkeypatch, tmp_path):
    """Overnight, the console has scrolled away by morning."""
    app = _App(boom=RuntimeError("gpu went away"))
    _wire(monkeypatch, tmp_path, app)
    with pytest.raises(RuntimeError):
        _run(app)
    text = (tmp_path / "crash.log").read_text(encoding="utf-8")
    assert "gpu went away" in text
    assert "orchestrate_frame" in text, "the traceback, not just the message"


def test_crash_logs_accumulate_rather_than_overwrite(monkeypatch, tmp_path):
    for msg in ("first failure", "second failure"):
        app = _App(boom=RuntimeError(msg))
        _wire(monkeypatch, tmp_path, app)
        with pytest.raises(RuntimeError):
            _run(app)
    text = (tmp_path / "crash.log").read_text(encoding="utf-8")
    assert "first failure" in text and "second failure" in text


def test_a_keyboard_interrupt_is_handled_like_any_other_exit(monkeypatch, tmp_path):
    """Ctrl+C is how a long run is usually stopped; it must flush too, which
    is why the guard catches BaseException rather than Exception."""
    app = _App(boom=KeyboardInterrupt())
    _wire(monkeypatch, tmp_path, app)
    with pytest.raises(KeyboardInterrupt):
        _run(app)
    assert "prefs" in app.log


# ---- cleanup must not give up halfway -----------------------------------

def test_one_failing_step_does_not_skip_the_rest(monkeypatch, tmp_path):
    """On a lost device every GL call raises. Unguarded, the first failure
    would skip everything below it - including the archive flush."""
    app = _App()
    _wire(monkeypatch, tmp_path, app)

    def explode():
        raise RuntimeError("context is gone")

    app.ui = _stub(get_state=app.get_state, cleanup=explode)
    _run(app)
    assert "prefs" in app.log
    assert "video_cleanup" in app.log


def test_the_archive_is_flushed_even_if_reading_ui_state_fails(monkeypatch, tmp_path):
    """The flush is pure numpy and disk, so it must not depend on the UI."""
    flushed = []

    class _Arc:
        def maybe_flush(self, force=False):
            flushed.append(force)

    app = _App()
    _wire(monkeypatch, tmp_path, app)
    app.archive = _Arc()

    def explode():
        raise RuntimeError("ui is gone")

    app.ui = _stub(get_state=explode, cleanup=lambda: None)
    _run(app)
    assert flushed == [True]


def test_a_cleanup_failure_does_not_mask_the_real_exception(monkeypatch, tmp_path):
    """Otherwise the user debugs the teardown instead of the cause."""
    app = _App(boom=RuntimeError("the real cause"))
    _wire(monkeypatch, tmp_path, app)

    def explode():
        raise RuntimeError("teardown noise")

    app.ui = _stub(get_state=explode, cleanup=explode)
    with pytest.raises(RuntimeError, match="the real cause"):
        _run(app)


def test_a_failure_to_write_the_crash_log_does_not_mask_the_crash(monkeypatch, tmp_path):
    app = _App(boom=RuntimeError("the real cause"))
    _wire(monkeypatch, tmp_path, app)
    import utilities.paths as paths

    def explode():
        raise OSError("disk full")

    monkeypatch.setattr(paths, "get_user_data_dir", explode)
    with pytest.raises(RuntimeError, match="the real cause"):
        _run(app)
