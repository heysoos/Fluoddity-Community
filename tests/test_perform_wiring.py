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
    """A stand-in wearing the real orchestrator methods.

    The fixed-viewpoint render is faked rather than built: it needs a GL
    context and a sim, and what these tests are about is whether it is
    installed and released in step with the window.
    """

    _drive_perform_window = App._drive_perform_window
    _load_calibration = App._load_calibration
    _sync_calibration = App._sync_calibration

    def __init__(self, fail=False):
        self.perform_window = _FakeWindow(fail=fail)
        self._perform_requested = None
        self._perform_device = ""
        self.perform_view = None
        self.installed = []          # projector sizes handed to the renderer
        self.released = 0

    def _install_perform_view(self, monitor):
        self.perform_view = object()
        self.installed.append((monitor.width, monitor.height))

    def _release_perform_view(self):
        self.perform_view = None
        self.released += 1


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
    # The projector renders its own frame; without this it has nothing to show.
    assert app.perform_view is not None
    assert app.installed == [(1920, 1080)]


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
    # And the renderer follows it to the new display's resolution.
    assert app.installed == [(1920, 1080), (1920, 1080)]


def test_disabling_closes_the_window(displays):
    app, state = _App(), UIState()
    state.perform.enabled = True
    state.preferences.perform_monitor = PROJECTOR.key
    app._drive_perform_window(state)
    state.perform.enabled = False
    app._drive_perform_window(state)
    assert app.perform_window.closes == 1
    assert state.perform.active_monitor == ""
    # The fixed-viewpoint render costs a particle pass per motion-blur sample,
    # so it must not outlive the window it feeds.
    assert app.perform_view is None
    assert app.released == 1


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
    # A failed open must not leave a renderer running with nothing to feed.
    assert app.perform_view is None
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


# ---- the corner calibration follows the DISPLAY -------------------------

SKEW = ((0.10, 0.95), (0.90, 0.80), (0.97, 0.12), (0.05, 0.05))
STORED = [[c[0], c[1]] for c in SKEW]


def _running(state, monitor=PROJECTOR):
    state.perform.enabled = True
    state.preferences.perform_monitor = monitor.key
    return state


def test_opening_adopts_the_calibration_that_display_was_left_with(displays):
    app, state = _App(), _running(UIState())
    state.preferences.perform_calibrations[PROJECTOR.device_key] = STORED
    app._drive_perform_window(state)
    assert state.perform.corners == SKEW


def test_a_projector_nobody_calibrated_starts_on_the_letterbox(displays):
    app, state = _App(), _running(UIState())
    state.preferences.perform_calibrations[LAPTOP.device_key] = STORED
    app._drive_perform_window(state)
    assert state.perform.corners is None, "it adopted another display's wall"


def test_a_corrupt_stored_calibration_reads_as_uncalibrated(displays):
    """preferences.config is a file a user can edit."""
    app, state = _App(), _running(UIState())
    state.preferences.perform_calibrations[PROJECTOR.device_key] = "rubbish"
    app._drive_perform_window(state)
    assert state.perform.corners is None


def test_dragging_a_corner_is_written_back_under_the_device_key(displays):
    app, state = _App(), _running(UIState())
    app._drive_perform_window(state)
    state.perform.corners = SKEW
    app._drive_perform_window(state)
    assert state.preferences.perform_calibrations == {
        PROJECTOR.device_key: STORED}


def test_moving_the_display_in_windows_keeps_its_calibration(displays):
    """device_key drops the POSITION, so rearranging monitors is not a new wall."""
    moved = _mon("Projector", x=0, phys=(508, 286))
    assert moved.key != PROJECTOR.key
    displays[:] = [LAPTOP, moved]

    app, state = _App(), _running(UIState())
    state.preferences.perform_calibrations[PROJECTOR.device_key] = STORED
    app._drive_perform_window(state)
    assert state.perform.corners == SKEW


def test_reset_drops_the_entry_rather_than_storing_the_letterbox(displays):
    """An uncalibrated display and a display reset to square are one thing."""
    app, state = _App(), _running(UIState())
    state.preferences.perform_calibrations[PROJECTOR.device_key] = STORED
    app._drive_perform_window(state)

    state.perform.reset_corners_requested = True
    app._drive_perform_window(state)

    assert state.perform.corners is None
    assert state.perform.reset_corners_requested is False
    assert PROJECTOR.device_key not in state.preferences.perform_calibrations


def test_stopping_clears_the_live_corners_but_not_the_stored_ones(displays):
    app, state = _App(), _running(UIState())
    app._drive_perform_window(state)
    state.perform.corners = SKEW
    state.perform.calibrating = True
    app._drive_perform_window(state)

    state.perform.enabled = False
    app._drive_perform_window(state)

    assert state.perform.corners is None
    assert state.perform.calibrating is False
    assert state.perform.held_corner == -1
    assert state.preferences.perform_calibrations[PROJECTOR.device_key] == STORED


def test_reopening_restores_what_the_last_session_dragged(displays):
    app, state = _App(), _running(UIState())
    app._drive_perform_window(state)
    state.perform.corners = SKEW
    app._drive_perform_window(state)

    state.perform.enabled = False
    app._drive_perform_window(state)
    state.perform.enabled = True
    app._drive_perform_window(state)

    assert state.perform.corners == SKEW


def test_nothing_is_written_for_a_display_left_alone(displays):
    app, state = _App(), _running(UIState())
    for _ in range(5):
        app._drive_perform_window(state)
    assert state.preferences.perform_calibrations == {}
