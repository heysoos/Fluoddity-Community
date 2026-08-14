"""A hover preview must write no history.

A preview applies a snapshot, which is the very thing the per-frame diff
watches. Recorded, each hovered row committed a step; the growing list shifted
the row under the pointer, so a different step previewed and committed in turn,
and the history filled in seconds.

These drive the REAL App methods with fake collaborators. The first version of
the frame-loop tests modelled the sequence with a local helper instead, which
is exactly why this loop was not caught.
"""
import types

import pytest

import main
from command_handler import CommandHandler
from services.undo_history import UndoHistory
from state.ui_state import UIState


class FakeSim:
    brain_layout = None

    def __init__(self):
        self.applied = []

    def apply_rule(self, rule):
        self.applied.append(rule)


class FakeRuleManager:
    def get_current_rule(self):
        return None

    def push_rule(self, rule, seed):
        pass


def build():
    st = UIState()
    app = types.SimpleNamespace(
        undo_history=UndoHistory(),
        _undo_preview_base=None,
        _undo_preview_showing=-1,
        rule_manager=FakeRuleManager(),
        sim=FakeSim(),
        ui=types.SimpleNamespace(undo_steps=[], undo_cursor=-1),
    )
    app.command_handler = CommandHandler(
        sim=app.sim, camera=None, ui=None, rule_manager=app.rule_manager,
        entity_picker=None, video_service=None, config_saver=None,
        multi_load_service=None, user_configs_dir=None)
    return app, st


def tick(app, st):
    """One orchestrate_frame, in the order main.py runs these in.

    The panel writes undo_preview_index and undo_jump_index during
    ui.render(); a test sets them directly to stand in for the pointer.
    """
    main.App._handle_undo(app, st)
    main.App._record_undo_step(app, st)
    main.App._handle_undo_preview(app, st)
    main.App._push_undo_rows(app, st)
    st.request_undo = False          # ui.get_state() clears these
    st.request_redo = False


def test_hovering_a_step_does_not_grow_the_history():
    app, st = build()
    tick(app, st)                                  # baseline
    st.sim.DRAG = 0.9
    tick(app, st)                                  # one real step
    assert len(app.undo_history.steps) == 2

    st.undo_preview_index = 0                      # pointer over the old row
    for _ in range(30):
        tick(app, st)

    assert len(app.undo_history.steps) == 2


def test_leaving_the_row_puts_the_live_state_back():
    app, st = build()
    tick(app, st)
    original = st.sim.DRAG
    st.sim.DRAG = 0.9
    tick(app, st)

    st.undo_preview_index = 0
    tick(app, st)
    tick(app, st)
    assert st.sim.DRAG == pytest.approx(original), "the preview should show"

    st.undo_preview_index = -1
    tick(app, st)
    assert st.sim.DRAG == pytest.approx(0.9), "un-hover restores the live state"
    assert len(app.undo_history.steps) == 2


def test_clicking_a_step_survives_the_pointer_leaving():
    """A click makes the previewed state permanent. Handing the pre-hover
    state back afterwards would silently undo the click."""
    app, st = build()
    tick(app, st)
    original = st.sim.DRAG
    st.sim.DRAG = 0.9
    tick(app, st)

    st.undo_preview_index = 0
    tick(app, st)
    st.undo_jump_index = 0                         # the click
    tick(app, st)
    st.undo_preview_index = -1                     # pointer leaves
    tick(app, st)

    assert st.sim.DRAG == pytest.approx(original)
    assert app.undo_history.cursor == 0


def test_a_real_edit_after_a_preview_still_records():
    """Suppression must end with the preview, not outlive it."""
    app, st = build()
    tick(app, st)
    st.undo_preview_index = 0
    tick(app, st)
    st.undo_preview_index = -1
    tick(app, st)

    st.sim.SENSOR_GAIN = 0.77
    tick(app, st)

    assert len(app.undo_history.steps) == 2
    assert app.undo_history.current().fields["sim"]["SENSOR_GAIN"] == pytest.approx(0.77)
