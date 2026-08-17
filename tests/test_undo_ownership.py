"""An undo step must be something the USER did.

Three parts of the app commandeer `speedmult`, `motion_blur` and
`blur_quality` for as long as they are running - an automatic mode, a
screenshot and a recording. Each of those fields is undoable, so every one of
those writes looked to the per-frame diff exactly like a slider being moved.
A generation deposits several; the journal caps at MAX_STEPS, so leaving
Explore running long enough throws away every real step the user made, and
half the survivors restore `speedmult = 0`, which stops the sim.

The rule is that the journal records what the user set, and while something
else owns the preferences what is on screen is not that.
"""
import numpy as np
import pytest

from main import App
from services import undo_history as uh
from services.undo_history import UndoHistory
from state.ui_state import UIState


class _Video:
    def __init__(self, active=False):
        self._active = active

    def is_active(self):
        return self._active


class _Handler:
    preview_active = False

    def __init__(self):
        self.restored = []

    def apply_undo_snapshot(self, snap, ui_state):
        self.restored.append(snap)
        return False


class _Sim:
    brain_layout = None


class _Rules:
    @staticmethod
    def get_current_rule():
        return None


class _App:
    """A stand-in wearing the REAL capture method."""

    _record_undo_step = App._record_undo_step
    _preferences_are_borrowed = App._preferences_are_borrowed
    _handle_undo = App._handle_undo

    def __init__(self, recording=False, screenshot=False):
        self.undo_history = UndoHistory()
        self._undo_preview_base = None
        self._undo_preview_showing = -1
        self.command_handler = _Handler()
        self.sim = _Sim()
        self.rule_manager = _Rules()
        self.video_service = _Video(recording)
        self.screenshot_in_progress = screenshot


def frame(app, st):
    before = len(app.undo_history.steps)
    app._record_undo_step(st)
    return len(app.undo_history.steps) > before


def test_an_ordinary_change_still_commits():
    """The guards below must not switch the feature off."""
    app, st = _App(), UIState()
    assert frame(app, st) is True
    st.sim.DRAG = 0.9
    assert frame(app, st) is True
    assert frame(app, st) is False


@pytest.mark.parametrize("mode", ("auto_tournament", "archive"))
def test_an_automatic_mode_deposits_no_steps(mode):
    """It drives speedmult from its own state machine, several times a
    generation, forever."""
    app, st = _App(), UIState()
    frame(app, st)
    setattr(getattr(st, mode), "enabled", True)
    for value in (0, 5, 0, 0, 5, 0) * 20:
        st.preferences.speedmult = value
        assert frame(app, st) is False
    assert len(app.undo_history.steps) == 1


def test_a_recording_deposits_no_steps():
    app, st = _App(recording=True), UIState()
    frame(app, st)
    st.preferences.speedmult = st.preferences.motion_blur_samples
    st.preferences.blur_quality = 1
    assert frame(app, st) is False


def test_a_screenshot_deposits_no_steps():
    app, st = _App(screenshot=True), UIState()
    frame(app, st)
    st.preferences.speedmult = st.preferences.motion_blur_samples
    assert frame(app, st) is False


def test_the_users_own_change_survives_a_whole_explore_run():
    """The point of the guard: what the user set is still the step at the
    cursor after the mode has churned for a while."""
    app, st = _App(), UIState()
    frame(app, st)
    st.sim.DRAG = 0.42
    frame(app, st)

    st.archive.enabled = True
    for value in (0, 5) * 200:
        st.preferences.speedmult = value
        frame(app, st)
    st.archive.enabled = False
    st.preferences.speedmult = 5

    assert len(app.undo_history.steps) == 2, "the mode filled the journal"
    assert app.undo_history.current().fields["sim"]["DRAG"] == pytest.approx(0.42)
    assert app.undo_history.can_undo()


def test_an_undo_is_refused_while_something_is_borrowed():
    """A hover puts someone else's rule and physics on screen, and the
    un-hover restores what was there before it wholesale. An undo applied
    underneath one is therefore reverted a moment later, silently - and the
    rebase has already overwritten the step it undid, so it is gone. Refusing
    with a message beats losing the step.
    """
    app, st = _App(), UIState()
    frame(app, st)
    st.sim.DRAG = 0.9
    frame(app, st)

    app.command_handler.preview_active = True
    st.request_undo = True
    app._handle_undo(st)

    assert app.command_handler.restored == [], "the undo ran under a preview"
    assert app.undo_history.can_undo(), "the step was consumed anyway"
    assert st.undo_notice


def test_an_undo_still_works_with_nothing_borrowed():
    app, st = _App(), UIState()
    frame(app, st)
    st.sim.DRAG = 0.9
    frame(app, st)

    st.request_undo = True
    app._handle_undo(st)

    assert len(app.command_handler.restored) == 1


# ---- the brain half ------------------------------------------------------

def _mlp(w_scale):
    from services import brains

    return brains.get("mlp").layout_from_settings(
        {"layers": [[16, 0]], "w_scale": w_scale})


def test_a_decode_scale_change_is_a_step_the_restore_can_actually_undo():
    """A scale is not in the signature - deliberately, so that dragging one
    does not tear down the archive. But the snapshot carries it, so a drag
    commits a step, and a restore gated on the signature alone puts nothing
    back. The rebase that follows then overwrites the old value with the new
    one, and it is gone for good.
    """
    from command_handler import CommandHandler

    class FakeSim:
        def __init__(self, layout):
            self.brain_layout = layout

        def apply_rule(self, rule):
            pass

    class FakeRules:
        def push_rule(self, rule, seed):
            pass

    applied = []
    st = UIState()
    rule = np.zeros((10, 8), dtype=np.float32)
    quiet, loud = _mlp(2.0), _mlp(6.0)
    assert quiet.signature() == loud.signature(), "a scale is not structural"

    snap = uh.capture(st, rule, quiet)
    handler = CommandHandler(
        sim=FakeSim(loud), camera=None, ui=None, rule_manager=FakeRules(),
        entity_picker=None, video_service=None, config_saver=None,
        multi_load_service=None, user_configs_dir=None)
    handler.apply_brain_layout = lambda layout, ui_state: applied.append(layout)

    handler.apply_undo_snapshot(snap, st)

    assert applied, "the scale was never put back"
    assert dict(applied[-1].scales)["w_scale"] == pytest.approx(2.0)
