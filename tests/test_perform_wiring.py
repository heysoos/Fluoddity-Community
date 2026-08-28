"""The orchestrator's half of perform mode: opening, closing and moving.

Drives the REAL App methods against a fake window. A method-level test of
PerformWindow cannot see any of this - the faults live in when the orchestrator
decides to call it.
"""
import pytest

import main as app_main
from main import App
from services.perform_window import MonitorInfo
from state.ui_state import UIState


def _mon(name, primary=False, x=0, phys=(380, 210)):
    return MonitorInfo(name=name, width=1920, height=1080, refresh=60,
                       x=x, y=0, is_primary=primary, phys_mm=phys)


LAPTOP = _mon("Laptop", primary=True, phys=(382, 215))
PROJECTOR = _mon("Projector", x=1920, phys=(508, 286))


class _FakeWindow:
    """A PerformWindow that records what it was asked to do."""

    def __init__(self, fail=False):
        self.monitor_key = ""
        self.monitor_label = ""
        self.opens = []
        self.closes = 0
        self.fail = fail

    @property
    def is_open(self):
        return bool(self.monitor_key)

    def open(self, monitor):
        self.opens.append(monitor.name)
        if self.fail:
            raise RuntimeError("no GL for you")
        self.monitor_key = monitor.key
        self.monitor_label = monitor.label()

    def close(self):
        self.closes += 1
        self.monitor_key = ""
        self.monitor_label = ""


class _App:
    """A stand-in wearing the real orchestrator methods."""

    _drive_perform_window = App._drive_perform_window

    def __init__(self, fail=False):
        self.perform_window = _FakeWindow(fail=fail)
        self._perform_requested = None


@pytest.fixture
def displays(monkeypatch):
    """Both connected, unless a test narrows it."""
    present = [LAPTOP, PROJECTOR]
    monkeypatch.setattr(app_main, "list_monitors", lambda: list(present))
    return present


def test_enabling_opens_the_window(displays):
    app, state = _App(), UIState()
    state.perform.enabled = True
    state.preferences.perform_monitor = PROJECTOR.key
    app._drive_perform_window(state)
    assert app.perform_window.opens == ["Projector"]
    assert "Projector" in state.perform.active_monitor


def test_staying_enabled_does_not_reopen_it(displays):
    app, state = _App(), UIState()
    state.perform.enabled = True
    state.preferences.perform_monitor = PROJECTOR.key
    for _ in range(10):
        app._drive_perform_window(state)
    assert app.perform_window.opens == ["Projector"]


def test_an_absent_display_falls_back_without_reopening_every_frame(displays):
    """The fallback's name is not what was asked for, so comparing the request
    against where it LANDED recreates the window on every single frame."""
    displays[:] = [LAPTOP]
    app, state = _App(), UIState()
    state.perform.enabled = True
    state.preferences.perform_monitor = PROJECTOR.key
    for _ in range(30):
        app._drive_perform_window(state)
    assert app.perform_window.opens == ["Laptop"]
    assert state.perform.notice


def test_picking_another_display_moves_the_window(displays):
    app, state = _App(), UIState()
    state.perform.enabled = True
    state.preferences.perform_monitor = PROJECTOR.key
    app._drive_perform_window(state)
    state.preferences.perform_monitor = LAPTOP.key
    app._drive_perform_window(state)
    app._drive_perform_window(state)
    assert app.perform_window.opens == ["Projector", "Laptop"]


def test_disabling_closes_the_window(displays):
    app, state = _App(), UIState()
    state.perform.enabled = True
    state.preferences.perform_monitor = PROJECTOR.key
    app._drive_perform_window(state)
    state.perform.enabled = False
    app._drive_perform_window(state)
    assert app.perform_window.closes == 1
    assert state.perform.active_monitor == ""


def test_staying_disabled_closes_nothing(displays):
    app, state = _App(), UIState()
    for _ in range(5):
        app._drive_perform_window(state)
    assert app.perform_window.closes == 0
    assert app.perform_window.opens == []


def test_a_failed_open_turns_perform_mode_off_and_says_so(displays):
    app, state = _App(fail=True), UIState()
    state.perform.enabled = True
    state.preferences.perform_monitor = PROJECTOR.key
    app._drive_perform_window(state)
    assert state.perform.enabled is False
    assert "no GL for you" in state.perform.notice
    # And it does not retry forever.
    app._drive_perform_window(state)
    assert app.perform_window.opens == ["Projector"]


def test_no_displays_at_all_turns_it_off(displays):
    displays[:] = []
    app, state = _App(), UIState()
    state.perform.enabled = True
    app._drive_perform_window(state)
    assert state.perform.enabled is False
    assert state.perform.notice
    assert app.perform_window.opens == []


def test_a_first_run_remembers_the_display_it_chose(displays):
    """Nothing remembered, so the choice becomes the memory."""
    app, state = _App(), UIState()
    state.perform.enabled = True
    app._drive_perform_window(state)
    assert state.preferences.perform_monitor == PROJECTOR.key
    for _ in range(5):
        app._drive_perform_window(state)
    assert app.perform_window.opens == ["Projector"]


def test_a_fallback_never_overwrites_the_remembered_display(displays):
    """Re-plugging the display you asked for must resume on it."""
    displays[:] = [LAPTOP]
    app, state = _App(), UIState()
    state.perform.enabled = True
    state.preferences.perform_monitor = PROJECTOR.key
    app._drive_perform_window(state)
    assert state.preferences.perform_monitor == PROJECTOR.key
