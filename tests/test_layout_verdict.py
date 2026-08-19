"""The archive is the judge, and it is already judging.

Admission means finite, viable, alive and separated from everything stored. A
layout that cannot produce ONE such tile in a whole expedition has answered the
question, so the verdict needs no second opinion and no threshold.

Comparing the admission RATE against the parent was rejected: a generation's
tiles share one CMA-ES population, so they clear or miss any bar together - the
same reason the adaptive admission threshold was removed.
"""
from __future__ import annotations

import numpy as np

from tests.test_imgep_driver import snaps
from tests.test_layout_expedition import a_driver, land


def run_expedition(d, admit: bool):
    """Score the expedition out, admitting something or nothing."""
    for g in range(int(d.expedition_gens)):
        z = d.ask(4)
        # A moving tile is alive; an all-zero one fails the viability gate.
        v = [(10 + 37 * g) % 250 for _ in range(4)] if admit else [0, 0, 0, 0]
        w = [(20 + 37 * g) % 250 for _ in range(4)] if admit else [0, 0, 0, 0]
        d.tell(z, snaps(4, [v, w]))


def a_move(**kw):
    """A driver with a layout move LIVE."""
    d, arc, ts = a_driver(**kw)
    d.start_expedition()
    land(d, arc)
    assert d.begin_moved_expedition() is True
    return d, arc, ts


def test_a_move_that_admits_nothing_asks_for_the_parent_back():
    d, _arc, _ts = a_move(expedition_gens=2)
    parent = d.layout_move.parent
    run_expedition(d, admit=False)
    assert d._move_admitted == 0
    assert d.requested_layout == parent
    assert d.layout_move is None


def test_a_reverted_move_is_not_immediately_re_proposed():
    d, arc, _ts = a_move(expedition_gens=2)
    pair = d.layout_move.pair
    run_expedition(d, admit=False)
    assert pair in arc.reverted_pairs()


def test_a_reverted_move_leaves_the_native_count_where_it_started():
    """It admitted nothing - that IS the revert condition - so putting the
    layout back puts every native row back with it."""
    d, arc, _ts = a_move(expedition_gens=2)
    parent = d.layout_move.parent
    run_expedition(d, admit=False)
    arc.retarget(d.requested_layout)
    assert d.requested_layout == parent
    assert len(arc.native_rows()) == len(arc)


def test_a_move_that_admits_keeps_the_layout():
    d, arc, _ts = a_move(expedition_gens=2)
    run_expedition(d, admit=True)
    assert d._move_admitted == 0          # consumed by the verdict
    assert d.requested_layout is None
    assert d.layout_move is None
    assert not arc.reverted_pairs()


def test_the_verdict_is_delivered_exactly_once():
    d, _arc, _ts = a_move(expedition_gens=2)
    run_expedition(d, admit=False)
    d.requested_layout = None
    d.end_expedition()
    assert d.requested_layout is None


def test_an_abandoned_move_is_neither_kept_nor_banned():
    """A grid change or a brain switch of the user's own ends the expedition
    without a verdict. Nothing was learned, so nothing is recorded."""
    d, arc, _ts = a_move(expedition_gens=50)
    d.end_expedition()
    assert d.requested_layout is None
    assert not arc.layout_moves()
    assert not arc.reverted_pairs()
    assert d.layout_move is None


def test_a_layout_that_only_repeats_the_archive_is_reverted():
    """The one that makes the verdict mean anything.

    keeper, summit and record all pass `force`, which bypasses separation so a
    generation is never absent from the record - so a layout producing nothing
    but near-duplicates still ADMITS every generation. Counting those would
    keep every layout that renders a picture at all.
    """
    d, arc, _ts = a_move(expedition_gens=3)
    # Nothing can ever separate from what is stored again. 2.0 is the largest
    # cosine distance two unit vectors can be apart, so this is 'never'.
    arc.min_separation = 2.0
    before = len(arc)
    parent = d.layout_move.parent
    run_expedition(d, admit=True)
    assert len(arc) > before, "keeper and summit still deposited tiles"
    assert d.requested_layout == parent
    assert arc.reverted_pairs()


def test_the_verdict_reaches_the_archives_ledger():
    d, arc, _ts = a_move(expedition_gens=2)
    mv = d.layout_move
    run_expedition(d, admit=False)
    row = arc.layout_moves()[-1]
    assert (row["parent"], row["child"]) == mv.pair
    assert row["kept"] is False
    assert row["op"] == mv.operator


def test_a_kept_move_is_filed_too():
    """The ledger is the record of the WALK, not only of its failures."""
    d, arc, _ts = a_move(expedition_gens=2)
    run_expedition(d, admit=True)
    assert arc.layout_moves()[-1]["kept"] is True


def test_a_ban_already_in_the_archive_is_honoured():
    """The ledger outlives the run, so a driver that has proposed nothing yet
    still knows what did not work."""
    from services.brains import default_layout
    from tests.test_layout_expedition import a_driver

    d, arc, _ts = a_driver()
    for child in ("fourier-n9", "fourier-n11"):
        arc.record_layout_move(default_layout().signature(), child,
                               "grow", 1, 0, False, 0)
    assert d.start_expedition() is True
    assert d.requested_layout is None      # every move from here is banned


def test_a_reset_does_not_forgive_a_reverted_move():
    """Reset clears the SEARCH, never the archive - and the ledger is a fact
    about the archive."""
    d, arc, _ts = a_move(expedition_gens=2)
    run_expedition(d, admit=False)
    d.reset()
    assert arc.reverted_pairs()


def test_an_ordinary_expedition_asks_for_nothing_when_it_ends():
    """The revert must belong to the MOVE, not to every expedition that
    happens to admit nothing."""
    d, _arc, _ts = a_driver(expedition_gens=2)
    d.layout_search = False
    assert d.start_expedition() is True
    run_expedition(d, admit=False)
    assert d.requested_layout is None
