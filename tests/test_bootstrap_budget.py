"""Bootstrap ends at seed_n natives OR a generation budget, whichever first.

Every layout deserves its own bootstrap - it is the cheapest exploration there
is, and a child's entries all cluster round the genome its expedition converged
on, so expanding from those alone explores a pinhole.

What the count alone cannot do is BOUND that in time. Separation is measured
against the POOLED archive, so a layout born into a full one admits ever more
slowly, and a fixed native count takes ever more generations to reach. The
budget is the ceiling that keeps each layout's bootstrap the same size.
"""
from __future__ import annotations

import numpy as np

from services.brains import REGISTRY
from services.genome_spec import spec_for
from tests.test_imgep_driver import make, snaps


def run(d, n, base=10):
    """`n` generations of tiles that admit."""
    for g in range(n):
        v = [(base + 37 * g + 7 * i) % 250 for i in range(4)]
        w = [(base + 37 * g + 7 * i + 11) % 250 for i in range(4)]
        d.tell(d.ask(4), snaps(4, [v, w]))


def test_a_fresh_archive_still_leaves_bootstrap_on_the_native_count():
    """The historical behaviour: with a generous budget the count is what
    ends it, exactly as before."""
    d, _arc, _ts = make(grid=2, seed_n=4, bootstrap_gens=1000)
    assert d.regime == "bootstrap"
    run(d, 3)
    assert d._native_n >= 4
    assert d.regime == "expansion"


def test_a_layout_that_admits_too_slowly_leaves_when_the_budget_is_spent():
    """The case the count cannot answer: seed_n is out of reach, so without a
    budget this layout bootstraps forever."""
    d, _arc, _ts = make(grid=2, seed_n=100000, bootstrap_gens=3)
    run(d, 2)
    assert d.regime == "bootstrap", "the budget is not spent yet"
    run(d, 1)
    assert d.regime == "expansion"


def test_nothing_to_expand_from_stays_in_bootstrap_whatever_the_budget():
    """A spent budget must not put parent sampling in front of an empty
    archive."""
    d, arc, _ts = make(grid=2, seed_n=1, bootstrap_gens=0)
    assert len(arc) == 0
    assert d.regime == "bootstrap"


def test_a_zero_budget_means_no_bootstrap_at_all():
    d, _arc, _ts = make(grid=2, seed_n=100000, bootstrap_gens=0)
    run(d, 1)
    assert d._native_n > 0
    assert d.regime == "expansion"


def test_only_bootstrap_generations_are_counted():
    """An expansion or expedition generation is not this layout's random
    exploration and must not spend its budget."""
    d, _arc, _ts = make(grid=2, seed_n=4, bootstrap_gens=1000)
    run(d, 3)
    assert d.regime == "expansion"
    spent = d._bootstrap_gens
    run(d, 3)
    assert d._bootstrap_gens == spent


def test_a_layout_change_hands_the_new_one_a_fresh_budget():
    """The whole point of making it PER layout."""
    d, _arc, _ts = make(grid=2, seed_n=100000, bootstrap_gens=3)
    run(d, 3)
    assert d.regime == "expansion"
    d.set_spec(spec_for(REGISTRY["gabor"].layout_from_settings({})))
    assert d._bootstrap_gens == 0


def test_a_scale_tweak_is_not_a_new_layout():
    """BrainLayout keeps scales out of ==, so a decode-scale change is the
    same space and must not restart the budget."""
    from services.brains import default_layout

    d, _arc, _ts = make(grid=2, seed_n=100000, bootstrap_gens=3)
    run(d, 2)
    lay = default_layout()
    d.set_spec(spec_for(lay))
    assert d._bootstrap_gens == 2


def test_reset_returns_the_budget():
    d, _arc, _ts = make(grid=2, seed_n=100000, bootstrap_gens=3)
    run(d, 2)
    d.reset()
    assert d._bootstrap_gens == 0


def test_an_expedition_becomes_reachable_once_the_budget_is_spent():
    """The gate that would otherwise undo all of this: it used to ask for
    seed_n natives too, so a layout out of bootstrap still could not run an
    expedition - and a layout move only ever rides on one."""
    d, _arc, _ts = make(grid=2, seed_n=100000, bootstrap_gens=2,
                        expansion_between=1, expedition_gens=5)
    run(d, 2)
    assert d.regime == "expansion"
    run(d, 2)
    assert d.regime == "expedition"


def test_the_phase_readout_names_whichever_limit_is_nearer():
    d, _arc, _ts = make(grid=2, seed_n=100000, bootstrap_gens=4)
    run(d, 3)
    ph = d.phase()
    assert ph["unit"] == "generations"
    assert (ph["done"], ph["total"]) == (3, 4)


def test_the_readout_still_counts_entries_while_they_are_the_nearer_limit():
    d, _arc, _ts = make(grid=2, seed_n=8, bootstrap_gens=1000)
    run(d, 1)
    assert d.phase()["unit"] == "entries"
