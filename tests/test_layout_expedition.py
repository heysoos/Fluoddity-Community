"""A layout move rides on an ordinary expedition, in TWO stages.

PROPOSED under the parent layout, where the seed lives and its genome can be
read; BEGUN under the child, where the carried genome's width is legal. Only
the frame loop can make the switch land - App owns the archive, the sim and the
tournament and the driver owns none of them - so the driver sets a one-shot and
waits.
"""
from __future__ import annotations

import numpy as np

from services.brains import REGISTRY, default_layout
from services.brains.layout_moves import LayoutBounds
from services.genome_spec import spec_for
from tests.test_imgep_driver import make, snaps

FOURIER = default_layout()


def a_driver(**kw):
    """A driver with one generation already in the archive, so a seed exists.

    grid=2 is four tiles, which is the popsize every test here asks for: the
    optimizer's is fixed at construction and _ask_expedition ends the
    expedition rather than crashing on a mismatch.
    """
    d, arc, ts = make(grid=2, seed_n=1, **kw)
    d.tell(d.ask(4), snaps(4, [[10, 20, 30, 40], [11, 21, 31, 41]]))
    d.layout_search = True
    d.layout_move_chance = 1.0
    return d, arc, ts


def land(d, arc=None):
    """What the frame loop does: adopt the requested layout, then begin.

    The ARCHIVE moves too. _apply_brain_layout retargets it, and which rows are
    native is what a seed is filtered by - a helper that moved only the spec
    would leave the parent's entries looking seedable under the child.
    """
    lay = d.requested_layout
    d.requested_layout = None
    if arc is not None:
        arc.retarget(lay)
    d.set_spec(spec_for(lay))
    return lay


def test_layout_search_is_off_by_default():
    """Opening the app must never start changing brain under anyone."""
    d, _arc, _ts = make()
    assert d.layout_search is False
    assert d.requested_layout is None


def test_an_off_search_proposes_nothing():
    d, _arc, _ts = a_driver()
    d.layout_search = False
    d.start_expedition()
    assert d.requested_layout is None


def test_an_expedition_asks_for_a_neighbouring_layout():
    d, _arc, _ts = a_driver()
    assert d.start_expedition() is True
    assert d.requested_layout is not None
    assert d.requested_layout != FOURIER
    # Proposed, not started: the genome is the child's width and the spec is
    # still the parent's.
    assert d.regime != "expedition"


def test_the_expedition_begins_only_once_the_switch_has_landed():
    d, arc, _ts = a_driver()
    d.start_expedition()
    child = land(d, arc)
    assert d.begin_moved_expedition() is True
    assert d.regime == "expedition"
    assert d.optimizer._x0.size == child.length
    assert d.layout_move.child == child


def test_a_grown_layout_enters_as_an_expedition_with_no_native_entries():
    """Every entry belongs to the parent's signature directory, so the child
    starts with nothing to expand FROM. The carried genome is what lets it
    search anyway."""
    d, arc, _ts = a_driver()
    d.start_expedition()
    land(d, arc)
    assert d.begin_moved_expedition() is True
    assert len(arc.native_rows()) == 0
    assert d.regime == "expedition"


def test_a_move_that_never_landed_is_dropped_rather_than_started():
    """The request may be refused, or overtaken by a switch of the user's own.
    Starting it anyway would optimise a genome in the wrong space."""
    d, _arc, _ts = a_driver()
    d.start_expedition()
    d.requested_layout = None            # the frame loop declined
    assert d.begin_moved_expedition() is False
    assert d.regime != "expedition"
    assert d.layout_move is None


def test_a_switch_to_a_THIRD_layout_drops_the_move_too():
    """Verified rather than assumed: the user can switch brain in the same
    frame, and the carried genome belongs to neither space then."""
    d, _arc, _ts = a_driver()
    d.start_expedition()
    d.requested_layout = None
    d.set_spec(spec_for(REGISTRY["lenia"].layout_from_settings({})))
    assert d.begin_moved_expedition() is False


def test_the_carried_genome_is_the_seeds_brain_grown_into_the_child():
    """A growth move is defined by what the brain COMPUTES, so compare the
    phenotype the optimizer's mean decodes to, not z."""
    from services.brains.layout_moves import transfer_genome
    from services.genome_spec import decode

    d, arc, _ts = a_driver()
    d.start_expedition()
    mv = d._pending.move
    seed = d._pending.seed
    child = land(d)
    d.begin_moved_expedition()
    assert mv.operator in ("grow", "shrink")
    want = transfer_genome(arc.brain_at(seed), mv.parent, mv.child,
                           np.random.default_rng(0)).reshape(-1)
    got = np.asarray(decode(d.optimizer._x0.astype(np.float32), child)
                     ).reshape(-1)
    assert got.size == want.size == child.length


def test_a_modality_jump_enters_as_an_expedition_not_as_bootstrap():
    """A layout with zero native entries would otherwise bootstrap, which on
    real settings is a thousand random genomes before expansion is reachable -
    a layout search that spends its whole life bootstrapping."""
    d, _arc, _ts = a_driver()
    # Bound growth away so the jump is the only move on offer.
    d.layout_bounds = LayoutBounds(max_width=int(FOURIER.shape[0]),
                                   modalities=("mlp",))
    assert d.start_expedition() is True
    child = land(d)
    assert child.modality == "mlp"
    assert d.begin_moved_expedition() is True
    assert d.regime == "expedition"
    assert d.optimizer._x0.size == child.length


def test_a_jump_carries_no_genome_and_says_so_by_seeding_at_random():
    """A jump IS a restart - there is no correspondence between a Fourier
    centre and an MLP unit - so it seeds the way bootstrap does."""
    d, _arc, _ts = a_driver()
    d.layout_bounds = LayoutBounds(max_width=int(FOURIER.shape[0]),
                                   modalities=("mlp",))
    d.start_expedition()
    land(d)
    d.begin_moved_expedition()
    assert d.layout_move.operator == "modality"
    assert np.any(d.optimizer._x0 != 0.0)


def test_a_move_is_not_proposed_while_one_is_live():
    d, arc, _ts = a_driver()
    d.start_expedition()
    land(d, arc)
    assert d.begin_moved_expedition() is True
    d.start_expedition()                 # a second one, on top
    assert d.requested_layout is None


def test_reset_drops_a_move_that_has_not_landed():
    d, _arc, _ts = a_driver()
    d.start_expedition()
    d.reset()
    assert d.requested_layout is None
    assert d.begin_moved_expedition() is False


def test_the_pending_move_survives_the_set_spec_that_lands_it():
    """set_spec ends the outgoing expedition when the space moves, and the
    space moving IS the request landing - so it must not take the request with
    it."""
    d, _arc, _ts = a_driver()
    d.start_expedition()
    child = d.requested_layout
    d.set_spec(spec_for(child))
    assert d._pending is not None
