"""Auto mode's transient overrides must never be persisted.

Reported symptom: after running the optimizer, the next launch showed a black
canvas with a working UI and no particles. Resetting all UI settings fixed it.

Cause: App.cleanup() persists preferences on exit, and Auto mode drives the
physics step count by writing straight into preferences.speedmult.
_drive_auto_tournament returns 0 on capture, score and write-rules frames, so
`speedmult = max(0, steps)` is legitimately 0 on those frames. Quitting on one
of them persisted speedmult = 0, and SimulationRunner then steps the sim zero
times per frame forever - nothing moves, nothing is drawn, and nothing on
screen explains why.

Auto mode restores its overrides on the Auto->off edge, but quitting while it
is still enabled never crosses that edge.
"""
import json

import pytest

from state.preferences_state import PreferencesState, load_preferences, save_preferences


def test_a_poisoned_file_heals_on_load(tmp_path):
    """Users who already have speedmult 0 on disk must recover without being
    told to reset their settings."""
    p = tmp_path / "prefs.json"
    prefs = PreferencesState()
    prefs.speedmult = 0
    save_preferences(prefs, p)
    assert json.loads(p.read_text())["speedmult"] == 0

    assert load_preferences(p).speedmult >= 1


@pytest.mark.parametrize("bad", [0, -1, -10])
def test_any_non_stepping_speedmult_is_repaired(tmp_path, bad):
    p = tmp_path / "prefs.json"
    prefs = PreferencesState()
    prefs.speedmult = bad
    save_preferences(prefs, p)
    assert load_preferences(p).speedmult >= 1


def test_a_healthy_speedmult_is_left_alone(tmp_path):
    p = tmp_path / "prefs.json"
    prefs = PreferencesState()
    prefs.speedmult = 7
    save_preferences(prefs, p)
    assert load_preferences(p).speedmult == 7


# -- the write side ----------------------------------------------------------


class FakeUIState:
    def __init__(self, enabled, speedmult, aspect, archive_enabled=False,
                 motion_blur=True):
        self.preferences = PreferencesState()
        self.preferences.speedmult = speedmult
        self.preferences.canvas_aspect_ratio = aspect
        self.preferences.motion_blur = motion_blur
        self.auto_tournament = type("A", (), {"enabled": enabled})()
        # Explore mode commandeers the same three preferences, so the restore
        # has to see it too - checking only auto_tournament let a quit from the
        # Explore tab persist all of them.
        self.archive = type("B", (), {"enabled": archive_enabled})()


def _restore(app_like, ui_state):
    from main import App
    return App._restore_auto_overrides(app_like, ui_state)


class FakeApp:
    def __init__(self, prev_speedmult, prev_aspect, prev_motion_blur=None):
        self._auto_prev_speedmult = prev_speedmult
        self._auto_prev_aspect = prev_aspect
        self._auto_prev_motion_blur = prev_motion_blur


def test_quitting_mid_run_restores_the_user_speedmult():
    """Auto mode is still enabled, so the Auto->off edge never fired."""
    ui = FakeUIState(enabled=True, speedmult=0, aspect="1:1")
    _restore(FakeApp(4, "16:9"), ui)
    assert ui.preferences.speedmult == 4
    assert ui.preferences.canvas_aspect_ratio == "16:9"


def test_quitting_outside_auto_mode_changes_nothing():
    ui = FakeUIState(enabled=False, speedmult=3, aspect="16:9")
    _restore(FakeApp(99, "1:1"), ui)
    assert ui.preferences.speedmult == 3
    assert ui.preferences.canvas_aspect_ratio == "16:9"


def test_restore_is_a_no_op_before_auto_has_ever_run():
    """Nothing was captured yet, so there is nothing to put back."""
    ui = FakeUIState(enabled=True, speedmult=2, aspect="1:1")
    _restore(FakeApp(None, None), ui)
    assert ui.preferences.speedmult == 2


# -- Explore mode and motion blur --------------------------------------------

def test_quitting_from_the_explore_tab_also_restores():
    """Explore commandeers the same three preferences as Auto. The restore
    originally checked only auto_tournament.enabled, so quitting from the
    Explore tab persisted every override - including speedmult 0."""
    ui = FakeUIState(enabled=False, speedmult=0, aspect="1:1",
                     archive_enabled=True)
    _restore(FakeApp(6, "16:9"), ui)
    assert ui.preferences.speedmult == 6
    assert ui.preferences.canvas_aspect_ratio == "16:9"


def test_motion_blur_is_restored_on_quit():
    """An automatic mode turns motion blur off so captures are single crisp
    frames rather than a five-frame average; leaving it off would silently
    change how the app looks on the next launch."""
    ui = FakeUIState(enabled=True, speedmult=0, aspect="1:1", motion_blur=False)
    _restore(FakeApp(4, "16:9", prev_motion_blur=True), ui)
    assert ui.preferences.motion_blur is True


def test_motion_blur_is_left_off_if_that_is_what_the_user_had():
    ui = FakeUIState(enabled=True, speedmult=0, aspect="1:1", motion_blur=False)
    _restore(FakeApp(4, "16:9", prev_motion_blur=False), ui)
    assert ui.preferences.motion_blur is False


def test_nothing_is_restored_when_neither_mode_is_enabled():
    ui = FakeUIState(enabled=False, speedmult=3, aspect="16:9", motion_blur=False)
    _restore(FakeApp(99, "1:1", prev_motion_blur=True), ui)
    assert ui.preferences.speedmult == 3
    assert ui.preferences.motion_blur is False
