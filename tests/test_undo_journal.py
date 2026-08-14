"""The journal's cursor arithmetic."""
from services.undo_history import Snapshot, UndoHistory


def snap(n: int) -> Snapshot:
    return Snapshot(fields={"sim": {"AXIAL_FORCE": float(n)}})


def test_a_fresh_journal_can_neither_undo_nor_redo():
    h = UndoHistory()
    assert not h.can_undo() and not h.can_redo()
    assert h.undo() is None and h.redo() is None


def test_one_committed_step_is_the_baseline_and_cannot_be_undone():
    h = UndoHistory()
    h.commit(snap(0))
    assert not h.can_undo()


def test_undo_returns_the_previous_step():
    h = UndoHistory()
    h.commit(snap(0))
    h.commit(snap(1))
    assert h.undo().fields["sim"]["AXIAL_FORCE"] == 0.0
    assert not h.can_undo()


def test_redo_returns_the_step_undone():
    h = UndoHistory()
    h.commit(snap(0))
    h.commit(snap(1))
    h.undo()
    assert h.redo().fields["sim"]["AXIAL_FORCE"] == 1.0
    assert not h.can_redo()


def test_a_commit_after_an_undo_truncates_the_redo_tail():
    h = UndoHistory()
    for n in range(4):
        h.commit(snap(n))
    h.undo()
    h.undo()
    h.commit(snap(99))
    assert [s.fields["sim"]["AXIAL_FORCE"] for s in h.steps] == [0.0, 1.0, 99.0]
    assert not h.can_redo()


def test_jump_moves_the_cursor_without_growing_the_history():
    h = UndoHistory()
    for n in range(4):
        h.commit(snap(n))
    assert h.jump(1).fields["sim"]["AXIAL_FORCE"] == 1.0
    assert len(h.steps) == 4
    assert h.can_undo() and h.can_redo()


def test_jump_out_of_range_returns_none_and_does_not_move():
    h = UndoHistory()
    h.commit(snap(0))
    h.commit(snap(1))
    assert h.jump(7) is None
    assert h.jump(-1) is None
    assert h.cursor == 1


def test_the_cap_drops_the_oldest_and_keeps_the_cursor_on_the_newest():
    h = UndoHistory()
    for n in range(UndoHistory.MAX_STEPS + 10):
        h.commit(snap(n))
    assert len(h.steps) == UndoHistory.MAX_STEPS
    assert h.cursor == UndoHistory.MAX_STEPS - 1
    assert h.steps[0].fields["sim"]["AXIAL_FORCE"] == 10.0


def test_a_tag_names_the_next_commit_and_is_consumed():
    h = UndoHistory()
    h.commit(snap(0))
    h.tag("Load Karst")
    h.commit(snap(1))
    assert h.steps[1].label == "Load Karst"
    h.commit(snap(2))
    assert h.steps[2].label != "Load Karst"


def test_an_untagged_commit_is_described():
    h = UndoHistory()
    h.commit(snap(0))
    h.commit(snap(1))
    assert h.steps[1].label == "Axial Force"


def test_rebase_replaces_the_step_at_the_cursor_without_moving_it():
    h = UndoHistory()
    h.commit(snap(0))
    h.commit(snap(1))
    h.rebase(snap(5))
    assert h.cursor == 1
    assert len(h.steps) == 2
    assert h.current().fields["sim"]["AXIAL_FORCE"] == 5.0
