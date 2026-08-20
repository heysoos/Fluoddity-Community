"""Every layout move a run makes, at the archive root.

Beside settings_history.jsonl and for the same reason: a run that changes brain
silently redirects where its results are filed, and nothing else records that it
happened. It is also what BANS a move, which is why it belongs to the archive
rather than to the run.
"""
from __future__ import annotations

from services.archive_io import ArchiveStore


def a_store(tmp_path):
    return ArchiveStore(tmp_path / "arc", signature="fourier-n10")


def test_the_ledger_sits_at_the_archive_root_not_under_a_layout(tmp_path):
    """One archive holds every brain, and a move is BETWEEN two of them - it
    belongs to neither signature directory."""
    s = a_store(tmp_path)
    assert s.layouts_path.parent == s.base
    assert s.layouts_path.name == "layouts.jsonl"


def test_an_archive_with_no_moves_reads_as_empty(tmp_path):
    assert a_store(tmp_path).load_layout_moves() == []


def test_a_row_survives_the_round_trip(tmp_path):
    s = a_store(tmp_path)
    row = {"parent": "fourier-n10", "child": "fourier-n11", "op": "grow",
           "gens": 50, "admitted": 3, "kept": True, "gen": 120}
    s.append_layout_move(row)
    assert s.load_layout_moves() == [row]


def test_rows_append_in_order(tmp_path):
    s = a_store(tmp_path)
    for i in range(3):
        s.append_layout_move({"op": "grow", "gen": i})
    assert [r["gen"] for r in s.load_layout_moves()] == [0, 1, 2]


def test_a_torn_line_costs_one_row_not_the_file(tmp_path):
    """Same discipline load_history already applies: a crash mid-write must not
    make the whole ledger unreadable."""
    s = a_store(tmp_path)
    s.append_layout_move({"op": "grow", "gen": 0})
    with open(s.layouts_path, "a", encoding="utf-8") as fh:
        fh.write('{"op": "shr')
    assert [r["gen"] for r in s.load_layout_moves()] == [0]


def test_a_disabled_store_writes_nothing_and_does_not_raise(tmp_path):
    s = a_store(tmp_path)
    s.enabled = False
    s.append_layout_move({"op": "grow"})
    assert not s.layouts_path.exists()


# ---- the archive's view of it -------------------------------------------

def an_archive(tmp_path=None):
    from services.archive import Archive

    store = a_store(tmp_path) if tmp_path is not None else None
    return Archive(store=store, dim=8)


def test_a_kept_move_bans_nothing():
    arc = an_archive()
    arc.record_layout_move("fourier-n10", "fourier-n11", "grow", 50, 3, True, 7)
    assert arc.reverted_pairs() == set()


def test_a_reverted_move_bans_its_pair():
    arc = an_archive()
    arc.record_layout_move("fourier-n10", "fourier-n11", "grow", 50, 0, False, 7)
    assert arc.reverted_pairs() == {("fourier-n10", "fourier-n11")}


def test_the_ban_is_directional():
    """Growing n10 -> n11 failing says nothing about shrinking n11 -> n10."""
    arc = an_archive()
    arc.record_layout_move("fourier-n10", "fourier-n11", "grow", 50, 0, False, 7)
    assert ("fourier-n11", "fourier-n10") not in arc.reverted_pairs()


def test_a_store_less_archive_still_bans():
    """Every test harness and every browse-only session runs without one."""
    arc = an_archive()
    arc.record_layout_move("a", "b", "grow", 1, 0, False, 0)
    assert arc.reverted_pairs() == {("a", "b")}


def test_a_ban_survives_reopening_the_archive(tmp_path):
    """The whole reason it is on disk: a move that produced nothing in this
    archive will not be re-proposed in the next session either."""
    arc = an_archive(tmp_path)
    arc.record_layout_move("fourier-n10", "fourier-n11", "grow", 50, 0, False, 7)
    again = an_archive(tmp_path)
    assert again.reverted_pairs() == {("fourier-n10", "fourier-n11")}


def test_the_row_records_what_the_move_was(tmp_path):
    arc = an_archive(tmp_path)
    arc.record_layout_move("fourier-n10", "fourier-n11", "grow", 50, 3, True, 7)
    row = arc.layout_moves()[-1]
    assert row["parent"] == "fourier-n10"
    assert row["child"] == "fourier-n11"
    assert row["op"] == "grow"
    assert row["gens"] == 50
    assert row["admitted"] == 3
    assert row["kept"] is True
    assert row["gen"] == 7
