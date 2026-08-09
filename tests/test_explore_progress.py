"""What the Explore tab shows about a running search.

Three things that were previously unanswerable from the UI:

  where in the phase is it     phase(): a regime name alone said nothing about
                               whether an expedition had 2 generations left or 48
  is the goal chase working    expedition_trace(): a converged expedition looks
                               identical to a climbing one from the outside
  is the archive still growing OUTWARD  the diversity trace: size alone only
                               ever goes up

and the batch side of the separation rule, which lives in the driver because
the archive sees one candidate at a time.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.test_imgep_driver import make, moving, snaps


def test_bootstrap_measures_progress_in_entries():
    d, _arc, _ts = make(seed_n=16)
    ph = d.phase()
    assert ph["total"] == 16 and ph["done"] == 0
    assert ph["unit"] == "entries"
    d.tell(d.ask(4), moving(4))
    assert d.phase()["done"] == 4


def test_expansion_counts_down_to_the_next_expedition():
    d, _a, _t = make(seed_n=4, expansion_between=5)
    for _ in range(2):
        d.tell(d.ask(4), moving(4))
    ph = d.phase()
    assert ph["total"] == 5 and ph["unit"] == "generations"
    assert "next expedition in" in ph["note"]


def test_expeditions_switched_off_have_no_finish_line():
    """A progress bar needs an end. Drawing one at 0% or 100% would both be
    lies when Expansion Between is 0, so the total is 0 and the UI says so."""
    d, _a, _t = make(seed_n=4, expansion_between=0)
    for _ in range(2):
        d.tell(d.ask(4), moving(4))
    ph = d.phase()
    assert ph["total"] == 0
    assert "Expansion Between = 0" in ph["label"]


def test_an_expedition_counts_down_its_own_generations():
    d, _a, _t = make(seed_n=4, expansion_between=0, expedition_gens=5)
    for _ in range(3):
        d.tell(d.ask(4), moving(4, base=10 * len(d.trace["gen"]) + 10))
    assert d.start_expedition_with(
        np.eye(8, dtype=np.float32)[0], "chase", "") is True
    ph = d.phase()
    assert ph["total"] == 5 and ph["done"] == 0
    d.tell(d.ask(4), moving(4))
    ph = d.phase()
    assert ph["done"] == 1 and "4 left" in ph["note"]


# ---- traces -------------------------------------------------------------

def test_every_generation_appends_one_row():
    d, _a, _t = make(seed_n=4)
    for _ in range(3):
        d.tell(d.ask(4), moving(4))
    t = d.trace
    assert len(t["gen"]) == 3
    assert all(len(v) == 3 for v in t.values())


def test_archive_size_and_mean_novelty_are_recorded():
    d, arc, _t = make(seed_n=4)
    d.tell(d.ask(4), moving(4))
    assert d.trace["archive_size"][-1] == len(arc)
    assert d.trace["mean_novelty"][-1] == pytest.approx(arc.mean_novelty())


def test_fitness_is_nan_outside_an_expedition():
    """NaN rather than absent, so the two series stay index-aligned against a
    shared x axis and a gap in the plot is a real gap."""
    d, _a, _t = make(seed_n=4, expansion_between=0)
    d.tell(d.ask(4), moving(4))
    assert np.isnan(d.trace["fit_best"][-1])


def test_the_expedition_trace_covers_exactly_the_expedition():
    d, _a, _t = make(seed_n=4, expansion_between=0, expedition_gens=3)
    for i in range(3):
        d.tell(d.ask(4), moving(4, base=30 * i + 10))
    assert d.expedition_trace()["gens"] == 0, "nothing has chased a goal yet"
    d.start_expedition_with(np.eye(8, dtype=np.float32)[0], "chase", "")
    for _ in range(2):
        d.tell(d.ask(4), moving(4))
    ex = d.expedition_trace()
    assert ex["gens"] == 2
    assert not any(np.isnan(v) for v in ex["best"])


def test_the_trace_survives_the_expedition_ending():
    """'Did that get anywhere' is only askable afterwards, so the last
    expedition's fitness must still be readable once it has finished."""
    d, _a, _t = make(seed_n=4, expansion_between=0, expedition_gens=2)
    for i in range(3):
        d.tell(d.ask(4), moving(4, base=30 * i + 10))
    d.start_expedition_with(np.eye(8, dtype=np.float32)[0], "chase", "")
    for _ in range(2):
        d.tell(d.ask(4), moving(4))
    assert d.regime != "expedition", "the expedition should have ended"
    assert d.expedition_trace()["gens"] == 2


def test_the_trace_is_capped():
    d, _a, _t = make(seed_n=4)
    d.TRACE_CAP = 3
    for _ in range(6):
        d.tell(d.ask(4), moving(4))
    assert all(len(v) == 3 for v in d.trace.values())


# ---- the batch side of separation ---------------------------------------

def converged(n, a=10, b=11):
    """A generation whose tiles are all the SAME pattern, and still alive.

    FakeScorer puts a tile on the axis of its mean brightness, so one value
    across every tile means one descriptor across every tile - the converged
    expedition, exactly. The two snapshots differ so liveness is not zero and
    the tiles are refused for being duplicates rather than for being dead.
    """
    return snaps(n, [[a] * n, [b] * n])


def test_identical_tiles_in_one_generation_do_not_all_get_in():
    """Without separating the batch from ITSELF, all four measure against the
    pre-generation archive, find nothing close, and all go in."""
    d, arc, _t = make(seed_n=4, min_separation=0.5)
    d.tell(d.ask(4), converged(4))
    assert len(arc) == 1


def test_a_generation_always_leaves_at_least_one_entry():
    d, arc, _t = make(seed_n=4, min_separation=0.9)
    for _ in range(4):
        d.tell(d.ask(4), converged(4))
    assert len(arc) == 4, "one per generation, even at an absurd separation"


def test_a_generation_of_dead_tiles_leaves_nothing():
    """The keeper is forced past separation, never past liveness: forcing a
    frozen tile in would admit one dud per generation forever."""
    d, arc, _t = make(seed_n=4, min_separation=0.9, liveness_min=0.5)
    d.tell(d.ask(4), snaps(4, [[10] * 4, [10] * 4]))     # identical frames
    assert len(arc) == 0


def test_the_admitted_count_is_reported():
    d, _a, _t = make(seed_n=4, min_separation=0.5)
    d.tell(d.ask(4), converged(4))
    st = d.status()
    assert st["last_tiles"] == 4 and st["last_admitted"] == 1


# ---- reset --------------------------------------------------------------

def test_reset_clears_the_traces():
    """The plots are drawn against a generation counter that restarts at 0
    here. Keeping the old points would draw the new run on top of the old one
    with no way to tell them apart."""
    d, _a, _t = make(seed_n=4)
    for _ in range(3):
        d.tell(d.ask(4), moving(4))
    assert d.trace["gen"]
    d.reset()
    assert all(v == [] for v in d.trace.values())
    assert d.expedition_trace()["gens"] == 0


def test_reset_does_not_clear_the_archive():
    """Reset abandons the trajectory, not the product."""
    d, arc, _t = make(seed_n=4)
    d.tell(d.ask(4), moving(4))
    n = len(arc)
    d.reset()
    assert len(arc) == n
