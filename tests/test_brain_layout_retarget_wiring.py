"""A layout change re-points the archive; it does not tear it down.

The archive is keyed by signature only in its DIRECTORY layout - every
layout's entries load into one Archive - so the switch that used to flush,
close, reload and refit now moves one pointer.
"""
from __future__ import annotations

from services.brains import BrainLayout, default_layout
from state.archive_state import ArchiveState
from state.auto_tournament_state import AutoTournamentState
from state.preferences_state import PreferencesState

FOURIER = default_layout()
GABOR = BrainLayout("gabor", (12,), 168)


class _Bag:
    pass


class _Log(list):
    def note(self, what):
        self.append(what)


class _Sim:
    def __init__(self, log):
        self.brain_layout = FOURIER
        self._log = log

    def realloc_brain_buffers(self, layout):
        self.brain_layout = layout
        self._log.note("realloc")

    def apply_rule(self, rule):
        self._log.note("apply_rule")

    def set_brain_scales(self, layout):
        self._log.note("set_brain_scales")


class _Archive:
    def __init__(self, log):
        self._log = log
        self.layout = FOURIER

    def retarget(self, layout):
        self.layout = layout
        self._log.note("retarget")


class _Service:
    def __init__(self, log):
        self._log = log
        self.driver = None

    def set_layout(self, layout):
        self._log.note("set_layout")

    def pause(self):
        self._log.note("pause")


class _App:
    def __init__(self, log):
        self._log = log
        self.sim = _Sim(log)
        self.archive = _Archive(log)
        self.archive_store = _Bag()
        self.tournament_service = _Service(log)
        self.auto_service = _Service(log)
        self.imgep_driver = None
        self.command_handler = None
        self.goal_list = None
        self.thumb_cache = None

    def _refresh_driver_specs(self, layout, reset=False):
        self._log.note(f"refresh_specs(reset={bool(reset)})")

    def _save_archive_settings(self, ui_state):
        self._log.note("save_settings")

    def _release_archive(self, ui_state):
        self._log.note("RELEASE")

    def _build_archive_set(self, path):
        self._log.note("REBUILD")

    def _load_archive_settings(self, ui_state):
        self._log.note("load_settings")

    def _apply_brain_layout(self, layout, ui_state, **kw):
        from main import App

        return App._apply_brain_layout(self, layout, ui_state, **kw)


class _UI:
    def __init__(self):
        self.archive = ArchiveState()
        self.auto_tournament = AutoTournamentState()
        self.preferences = PreferencesState()


def test_a_layout_change_retargets_instead_of_rebuilding():
    log = _Log()
    app, ui = _App(log), _UI()
    assert app._apply_brain_layout(GABOR, ui) is True
    assert "retarget" in log
    assert "RELEASE" not in log, "the archive must not be released"
    assert "REBUILD" not in log, "the archive must not be reloaded from disk"


def test_the_search_is_still_stopped_and_the_optimizer_still_reset():
    """The search SPACE moved, so the covariance and the population are
    meaningless whatever the archive cost is."""
    log = _Log()
    app, ui = _App(log), _UI()
    ui.archive.running = True
    app._apply_brain_layout(GABOR, ui)
    assert ui.archive.running is False
    assert "pause" in log
    assert "refresh_specs(reset=True)" in log


def test_a_move_the_search_asked_for_keeps_it_running():
    """The search moved its own space deliberately and has an expedition ready
    to start in the new one, so the pause and the optimizer reset a hand switch
    needs would cost the whole run."""
    log = _Log()
    app, ui = _App(log), _UI()
    ui.archive.running = True
    app._apply_brain_layout(GABOR, ui, keep_running=True)
    assert ui.archive.running is True
    assert "pause" not in log
    assert "refresh_specs(reset=True)" not in log
    # Everything that is about the SPACE still happens.
    assert "retarget" in log
    assert "realloc" in log


def test_the_settings_are_written_after_the_switch_not_before():
    """They record the layout the archive is now ON, so a save taken before
    the switch would file the outgoing one."""
    log = _Log()
    app, ui = _App(log), _UI()
    app._apply_brain_layout(GABOR, ui)
    assert log.index("retarget") < log.index("save_settings")


def test_a_scales_only_change_still_takes_the_light_path():
    """What a z MEANS changed, not how wide it is, so there is nothing to
    retarget and nothing to reset."""
    log = _Log()
    app, ui = _App(log), _UI()
    scaled = BrainLayout(FOURIER.modality, FOURIER.shape, FOURIER.length,
                         scales=(("freq_scale", 4.0),))
    assert app._apply_brain_layout(scaled, ui) is True
    assert "set_brain_scales" in log
    assert "retarget" not in log
    assert "realloc" not in log


def test_an_unchanged_layout_does_nothing_at_all():
    log = _Log()
    app, ui = _App(log), _UI()
    assert app._apply_brain_layout(FOURIER, ui) is False
    assert log == []


def test_a_layout_change_with_no_archive_open_still_works():
    """The Brain window works before Explore has ever been opened."""
    log = _Log()
    app, ui = _App(log), _UI()
    app.archive = None
    app.archive_store = None
    assert app._apply_brain_layout(GABOR, ui) is True
    assert "realloc" in log
    assert "retarget" not in log
