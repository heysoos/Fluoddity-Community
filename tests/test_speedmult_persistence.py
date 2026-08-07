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
    def __init__(self, enabled, speedmult, aspect):
        self.preferences = PreferencesState()
        self.preferences.speedmult = speedmult
        self.preferences.canvas_aspect_ratio = aspect
        self.auto_tournament = type("A", (), {"enabled": enabled})()


def _restore(app_like, ui_state):
    from main import App
    return App._restore_auto_overrides(app_like, ui_state)


class FakeApp:
    def __init__(self, prev_speedmult, prev_aspect):
        self._auto_prev_speedmult = prev_speedmult
        self._auto_prev_aspect = prev_aspect


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
