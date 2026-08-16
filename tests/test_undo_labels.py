"""What a step is CALLED, which is the only thing telling two rows apart.

The journal can name a step two ways: guess it from what changed, or let the
code that caused the change say. The second existed and nothing ever called
it, so every row was a guess - and a preset load, which moves a dozen fields
at once, guessed "Settings" for every preset in the library.

A tag is advisory: the DIFF is what commits a step, so a call site that
forgets to tag costs a name and never coverage.
"""
import numpy as np
import pytest

from main import App
from services import undo_history as uh
from services.undo_history import UndoHistory
from state.ui_state import UIState


class _Handler:
    preview_active = False


class _Sim:
    brain_layout = None


class _Rules:
    @staticmethod
    def get_current_rule():
        return None


class _App:
    _record_undo_step = App._record_undo_step
    _preferences_are_borrowed = App._preferences_are_borrowed

    def __init__(self):
        self.undo_history = UndoHistory()
        self._undo_preview_base = None
        self.command_handler = _Handler()
        self.sim = _Sim()
        self.rule_manager = _Rules()
        self.video_service = type("V", (), {"is_active": lambda _s: False})()
        self.screenshot_in_progress = False


def test_a_tag_names_the_step_it_rides_with():
    app, st = _App(), UIState()
    app._record_undo_step(st)                     # baseline

    st.undo_tag = "Load Karst"
    st.sim.DRAG = 0.9
    app._record_undo_step(st)

    assert app.undo_history.current().label == "Load Karst"
    assert st.undo_tag == "", "the tag must not name a second step"


def test_an_untagged_change_still_gets_its_guessed_name():
    app, st = _App(), UIState()
    app._record_undo_step(st)
    st.sim.DRAG = 0.9
    app._record_undo_step(st)
    assert app.undo_history.current().label == "Drag"


def test_a_tag_survives_a_frame_that_commits_nothing():
    """A load whose capture is deferred - a widget still active, say - must
    keep its name until the frame that actually records it."""
    app, st = _App(), UIState()
    app._record_undo_step(st)

    st.undo_tag = "Load Karst"
    st.any_widget_active = True
    st.sim.DRAG = 0.9
    app._record_undo_step(st)                     # deferred
    assert st.undo_tag == "Load Karst"

    st.any_widget_active = False
    app._record_undo_step(st)
    assert app.undo_history.current().label == "Load Karst"


def test_a_tag_for_a_load_that_changed_nothing_is_dropped():
    """Loading the preset already on screen moves nothing, so there is no step
    to name - and a tag left lying about would name the NEXT one."""
    app, st = _App(), UIState()
    app._record_undo_step(st)

    st.undo_tag = "Load Karst"
    app._record_undo_step(st)                     # nothing changed
    assert st.undo_tag == ""

    st.sim.DRAG = 0.9
    app._record_undo_step(st)
    assert app.undo_history.current().label == "Drag"


# ---- the guesser's own gap ----------------------------------------------

def _mlp(w_scale):
    from services import brains

    return brains.get("mlp").layout_from_settings(
        {"layers": [[16, 0]], "w_scale": w_scale})


def test_a_decode_scale_change_is_named_after_the_scale():
    """It moves no declared field and no rule, so the guesser used to fall
    through to a bare "Change" - the least useful row in the panel, on the
    one modality whose scales a rig modulates."""
    st = UIState()
    rule = np.zeros((10, 8), dtype=np.float32)
    old = uh.capture(st, rule, _mlp(2.0))
    new = uh.capture(st, rule, _mlp(6.0))

    assert uh.describe(old, new) == "W Scale"


def test_a_genuinely_unidentifiable_change_still_says_change():
    st = UIState()
    old = uh.capture(st, None, None)
    assert uh.describe(old, old) == "Change"


# ---- the call sites ------------------------------------------------------

def test_the_load_paths_really_set_a_tag():
    """Source-level, because the alternative is a full config-load harness
    per path - and the defect being guarded is a call site that quietly stops
    setting it, which no behavioural test of the journal can see.
    """
    import inspect

    import command_handler

    src = inspect.getsource(command_handler)
    assert src.count("ui_state.undo_tag = ") >= 4, (
        "a load path stopped naming its step")


def test_a_tag_goes_through_state_and_never_through_the_journal():
    """CommandHandler holds no journal, deliberately: it sets a one-shot on
    state and the orchestrator forwards it, which is the pattern every other
    command already follows. It reads the undo_history MODULE for CONTAINERS,
    which is a declaration, not the object.
    """
    import inspect

    import command_handler

    src = inspect.getsource(command_handler)
    assert "self.undo_history" not in src
    assert "UndoHistory(" not in src
