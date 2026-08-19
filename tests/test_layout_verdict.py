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
    d, _arc, _ts = a_move(expedition_gens=2)
    pair = d.layout_move.pair
    run_expedition(d, admit=False)
    assert pair in d._reverted


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
    d, _arc, _ts = a_move(expedition_gens=2)
    run_expedition(d, admit=True)
    assert d._move_admitted == 0          # consumed by the verdict
    assert d.requested_layout is None
    assert d.layout_move is None
    assert not d._reverted


def test_the_verdict_is_delivered_exactly_once():
    d, _arc, _ts = a_move(expedition_gens=2)
    run_expedition(d, admit=False)
    d.requested_layout = None
    d.end_expedition()
    assert d.requested_layout is None


def test_an_abandoned_move_is_neither_kept_nor_banned():
    """A grid change or a brain switch of the user's own ends the expedition
    without a verdict. Nothing was learned, so nothing is recorded."""
    d, _arc, _ts = a_move(expedition_gens=50)
    d.end_expedition()
    assert d.requested_layout is None
    assert not d._reverted
    assert d.layout_move is None


def test_an_ordinary_expedition_asks_for_nothing_when_it_ends():
    """The revert must belong to the MOVE, not to every expedition that
    happens to admit nothing."""
    d, _arc, _ts = a_driver(expedition_gens=2)
    d.layout_search = False
    assert d.start_expedition() is True
    run_expedition(d, admit=False)
    assert d.requested_layout is None
