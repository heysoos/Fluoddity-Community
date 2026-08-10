"""The expedition high-water mark.

Separation and fitness rank tiles by different things, and admission was only
ever asking the separation question. An expedition climbing toward a goal
necessarily produces tiles that resemble the ones it has just produced, so its
BEST-matching tile - the actual result of the whole chase - is exactly what
separation is built to reject. `keeper` did not cover it: that is the most
NOVEL viable tile, which during a converging chase is close to the least
goal-matching one.

So a tile that beats its expedition's best gets in regardless, and the ratchet
is what keeps that bounded: a climb over rough ground leaves a checkpoint at
every gain, a converged expedition stops improving and therefore stops
admitting.
"""
from __future__ import annotations

import numpy as np

from tests.test_imgep_driver import DIM, make, moving, snaps


def start(d, axis=0):
    """Begin an expedition toward a fixed one-hot goal."""
    g = np.eye(DIM, dtype=np.float32)[axis]
    assert d.start_expedition_with(g, "chase", "") is True
    return g


def converged(n, a=10, b=11):
    """A generation whose tiles are all the same pattern, and still alive."""
    return snaps(n, [[a] * n, [b] * n])


def seed(d, gens=2):
    """Enough archive for an expedition to have somewhere to start."""
    for i in range(gens):
        d.tell(d.ask(4), moving(4, base=30 * i + 10))


def sources(arc):
    return [e.source for e in arc.entries]


def test_a_converged_expedition_still_records_its_best():
    """The case from the report: at a separation the whole batch fails, the
    novelty keeper gets in and the goal-matcher does not."""
    d, arc, _ = make(seed_n=4, min_separation=0.9, expedition_gens=10)
    seed(d)
    start(d)
    before = len(arc)
    d.tell(d.ask(4), converged(4))
    assert "summit" in sources(arc)[before:]


def test_the_mark_ratchets_so_a_plateau_stops_admitting():
    d, arc, _ = make(seed_n=4, min_separation=0.9, expedition_gens=10)
    seed(d)
    start(d)
    d.tell(d.ask(4), converged(4))
    first = sources(arc).count("summit")
    for _ in range(4):
        d.tell(d.ask(4), converged(4))       # identical tiles, identical fitness
    assert sources(arc).count("summit") == first, \
        "a flat expedition kept admitting summits"


def test_an_improvement_is_admitted_even_when_the_archive_has_one_already():
    """The point of a checkpoint on rough ground: each genuine gain is kept,
    not only the first."""
    d, arc, _ = make(seed_n=4, min_separation=0.9, expedition_gens=20)
    seed(d)
    start(d, axis=0)
    n = []
    # FakeScorer sends a tile to axis (mean brightness % dim). Walking the
    # bright tile toward axis 0 walks its contrastive fitness up with it.
    for v in (3, 2, 1, DIM):                 # DIM % DIM == 0, the goal axis
        d.tell(d.ask(4), snaps(4, [[10, 10, 10, v], [12, 12, 12, v + DIM]]))
        n.append(sources(arc).count("summit"))
    assert n[-1] > n[0], "no later gain was recorded"


def test_summits_do_not_happen_outside_an_expedition():
    """Expansion has no goal, so there is no fitness and nothing to beat."""
    d, arc, _ = make(seed_n=4, min_separation=0.9, expansion_between=0)
    for _ in range(4):
        d.tell(d.ask(4), converged(4))
    assert "summit" not in sources(arc)


def test_a_settled_pattern_is_still_a_summit():
    """Liveness does NOT gate the summit.

    Liveness is higher during the transient after a reset than once a pattern
    settles into its attractor - so a converged chase, which is the thing an
    expedition is for, scores low on it. A floor on the bulk of the archive is
    not a veto over the one entry that was chosen.
    """
    d, arc, _ = make(seed_n=4, min_separation=0.9, expedition_gens=10)
    for i in range(2):                       # seed while the floor is off
        d.tell(d.ask(4), moving(4, base=30 * i + 10))
    d.liveness_min = 0.5                     # far above anything real
    start(d)
    before = len(arc)
    d.tell(d.ask(4), snaps(4, [[10] * 4, [10] * 4]))     # identical frames
    assert sources(arc)[before:] == ["summit"], "the settled endpoint was lost"


def test_a_blank_generation_still_produces_no_summit():
    """Viability is not liveness and keeps its veto. A black frame is not a
    result whatever it scores against the goal."""
    d, arc, _ = make(seed_n=4, min_separation=0.9, expedition_gens=10)
    seed(d)
    start(d)
    before = len(arc)
    d.tell(d.ask(4), snaps(4, [[0] * 4, [0] * 4]))       # black
    assert sources(arc)[before:] == []


def test_the_keeper_still_respects_liveness():
    """Only the summit loses the veto. keeper fires every generation forever,
    so a dead preset would otherwise deposit one frozen tile per generation
    without bound - the debug05 pattern."""
    d, arc, _ = make(seed_n=4, min_separation=0.9, liveness_min=0.5,
                     expansion_between=0)
    d.tell(d.ask(4), snaps(4, [[10] * 4, [10] * 4]))     # alive-but-frozen
    assert len(arc) == 0


def test_a_new_expedition_starts_from_a_clean_mark():
    """Fitness is contrastive against a different goal each time, so a mark
    carried over would suppress every summit of the next chase."""
    d, arc, _ = make(seed_n=4, min_separation=0.9, expedition_gens=10)
    seed(d)
    start(d, axis=0)
    d.tell(d.ask(4), converged(4))
    assert d._expedition_best > -np.inf
    start(d, axis=3)
    assert d._expedition_best == -np.inf


def test_ending_an_expedition_clears_the_mark():
    d, _, _ = make(seed_n=4, expedition_gens=1)
    seed(d)
    start(d)
    d.tell(d.ask(4), converged(4))
    assert d.regime != "expedition"
    assert d._expedition_best == -np.inf


def test_the_count_is_reported_and_survives_the_expedition():
    d, _, _ = make(seed_n=4, min_separation=0.9, expedition_gens=10)
    seed(d)
    start(d)
    d.tell(d.ask(4), converged(4))
    assert d.status()["n_summits"] >= 1


def test_reset_clears_the_count():
    d, _, _ = make(seed_n=4, min_separation=0.9, expedition_gens=10)
    seed(d)
    start(d)
    d.tell(d.ask(4), converged(4))
    d.reset()
    assert d.status()["n_summits"] == 0


def test_the_fitness_returned_is_unchanged():
    """The ratchet reads the expedition fitness; it must not alter what the
    optimizer is told, or the search itself would change."""
    a, _, _ = make(seed_n=4, min_separation=0.0, expedition_gens=10)
    seed(a)
    start(a)
    fit = a.tell(a.ask(4), converged(4))
    assert len(fit) == 4 and np.all(np.isfinite(fit))
