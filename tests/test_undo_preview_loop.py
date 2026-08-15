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
    def __init__(self):
        self.applied = []
        self.brain_layout = None

    def apply_rule(self, rule):
        self.applied.append(rule)


class FakeRuleManager:
    def __init__(self):
        self.rule = None

    def get_current_rule(self):
        return self.rule

    def push_rule(self, rule, seed):
        self.rule = rule


def build():
    st = UIState()
    app = types.SimpleNamespace(
        undo_history=UndoHistory(),
        _undo_preview_base=None,
        _undo_preview_showing=-1,
        rule_manager=FakeRuleManager(),
        sim=FakeSim(),
        ui=types.SimpleNamespace(undo_steps=[], undo_cursor=-1),
        # Nothing has commandeered the render preferences in these tests.
        video_service=types.SimpleNamespace(is_active=lambda: False),
        screenshot_in_progress=False,
    )
    app._preferences_are_borrowed = types.MethodType(
        main.App._preferences_are_borrowed, app)
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


# ---- the OTHER previews ----------------------------------------------------
#
# The undo panel's own hover is not the only thing that borrows state onto the
# screen. File > Load previews a preset on hover, the archive browser previews
# an entry, and the clipboard previews a pasted config - all three write the
# rule and the physics the per-frame diff watches, and none of them is a change
# the user made.

@pytest.mark.parametrize("kind", ["menu", "gallery", "clipboard"])
def test_a_borrowed_preview_writes_no_history(kind):
    """Sliding down File > Load hovers a preset per row. Recorded, each one
    commits - twice, since un-hovering restores and differs again - so the
    history fills in seconds and the real steps fall off the end."""
    app, st = build()
    tick(app, st)
    st.sim.DRAG = 0.9
    tick(app, st)
    assert len(app.undo_history.steps) == 2

    ch = app.command_handler
    for row in range(20):
        # what the preview itself writes: a rule onto the stack, and the
        # physics the previewed config ran under
        if kind == "menu":
            ch.preview_rule_active = True
        elif kind == "gallery":
            ch._archive_preview_id = row
        else:
            ch.clipboard_preview_active = True
        ch.rule_manager.push_rule([float(row)], 0.0)
        st.sim.SENSOR_GAIN = 0.1 + 0.01 * row
        st.sim.rule_seed = float(row)
        tick(app, st)

    assert len(app.undo_history.steps) == 2, (
        f"the {kind} preview recorded {len(app.undo_history.steps) - 2} steps")


@pytest.mark.parametrize("kind", ["menu", "gallery", "clipboard"])
def test_committing_a_preview_does_record(kind):
    """Suppression is for the hover, not for the click. A preset the user
    actually loads is a change like any other."""
    app, st = build()
    tick(app, st)

    ch = app.command_handler
    if kind == "menu":
        ch.preview_rule_active = True
    elif kind == "gallery":
        ch._archive_preview_id = 4
    else:
        ch.clipboard_preview_active = True
    st.sim.SENSOR_GAIN = 0.42
    tick(app, st)
    assert len(app.undo_history.steps) == 1

    # the click: the preview ends and the state stays
    ch.preview_rule_active = False
    ch._archive_preview_id = -1
    ch.clipboard_preview_active = False
    tick(app, st)

    assert len(app.undo_history.steps) == 2
    assert app.undo_history.current().fields["sim"]["SENSOR_GAIN"] == pytest.approx(0.42)


# ---- a brain change must be undoable -----------------------------------

def _mlp_layout(layers):
    from services.brains import REGISTRY

    return REGISTRY["mlp"].layout_from_settings({"layers": layers})


def test_undoing_a_brain_change_puts_the_WINDOW_back_too():
    """_handle_brain_layout re-applies ui_state.brain EVERY frame, so a restore
    that moves only sim.brain_layout is reverted on the next one - the symptom
    being an undo that appears to work and then keeps the new brain on top."""
    from services.brains import default_layout, settings_of
    from ui.brain_window import layout_for

    app, st = build()
    ch = app.command_handler
    switched = []
    ch.apply_brain_layout = lambda layout, ui_state: (
        switched.append(layout), setattr(app.sim, "brain_layout", layout))[0]

    start = default_layout()
    app.sim.brain_layout = start
    st.brain.modality = start.modality
    st.brain.settings = dict(settings_of(start))
    ch.rule_manager.push_rule([1.0], 0.0)
    tick(app, st)

    after = _mlp_layout([[16, 0], [16, 0]])
    app.sim.brain_layout = after
    st.brain.modality = after.modality
    st.brain.settings = dict(settings_of(after))
    ch.rule_manager.push_rule([2.0], 0.0)
    tick(app, st)
    assert len(app.undo_history.steps) == 2

    st.request_undo = True
    tick(app, st)

    assert app.sim.brain_layout.signature() == start.signature()
    # The invariant that makes the next frame a no-op instead of a revert.
    assert layout_for(st.brain.modality, st.brain.settings).signature() == \
        start.signature(), "the Brain window still names the undone layout"

    st.request_redo = True
    tick(app, st)
    assert app.sim.brain_layout.signature() == after.signature()
    assert layout_for(st.brain.modality, st.brain.settings).signature() == \
        after.signature(), "redo left the window on the undone layout"
    # Neither direction may append: rebase() is what keeps the journal from
    # growing in the direction it was asked to shrink.
    assert len(app.undo_history.steps) == 2
