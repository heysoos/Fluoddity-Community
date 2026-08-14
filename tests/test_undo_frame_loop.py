"""The per-frame capture rule: when a step is committed, and when it is not."""
import pytest

from services import undo_history as uh
from services.undo_history import UndoHistory
from state.ui_state import UIState


def record(history, ui_state, rule=None, layout=None):
    """The orchestrator's per-frame step, isolated from App."""
    if ui_state.any_widget_active:
        return False
    snap = uh.capture(ui_state, rule, layout)
    held = history.current()
    if held is not None and uh.same(held, snap):
        return False
    history.commit(snap)
    return True


def test_the_first_call_commits_a_baseline():
    h, st = UndoHistory(), UIState()
    assert record(h, st) is True
    assert len(h.steps) == 1
    assert not h.can_undo()


def test_an_unchanged_frame_commits_nothing():
    h, st = UndoHistory(), UIState()
    record(h, st)
    for _ in range(10):
        assert record(h, st) is False
    assert len(h.steps) == 1


def test_a_change_commits_exactly_one_step():
    h, st = UndoHistory(), UIState()
    record(h, st)
    st.sim.DRAG = 0.9
    assert record(h, st) is True
    assert record(h, st) is False
    assert len(h.steps) == 2


def test_a_drag_in_flight_commits_nothing_and_releases_as_one_step():
    h, st = UndoHistory(), UIState()
    record(h, st)
    st.any_widget_active = True
    for value in (0.1, 0.2, 0.3, 0.4):
        st.sim.DRAG = value
        assert record(h, st) is False
    st.any_widget_active = False
    assert record(h, st) is True
    assert len(h.steps) == 2
    assert h.current().fields["sim"]["DRAG"] == pytest.approx(0.4)


def test_rebasing_after_a_restore_stops_the_undo_committing_itself():
    h, st = UndoHistory(), UIState()
    record(h, st)
    st.sim.DRAG = 0.9
    record(h, st)

    restored = h.undo()
    for name, module, _panel in uh.CONTAINERS:
        target = getattr(st, name)
        for f in module.UNDOABLE_FIELDS:
            setattr(target, f, restored.fields[name][f])
    h.rebase(uh.capture(st, None, None))

    assert record(h, st) is False
    assert len(h.steps) == 2
    assert h.can_redo()
