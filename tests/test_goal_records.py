"""The archive keeps its best-ever match for every text goal.

A run chasing "pepperoni pizza" wanders through shapes, and one of them can be
the best "a smiley face" the archive has ever seen. Nothing else in the
pipeline keeps it: novelty does not know the goal list exists, and separation
asks whether the archive already holds something SIMILAR, which is a different
question from whether it holds something BETTER.

So the record book runs in every regime, not only during an expedition toward
that goal, and it is measured against the archive rather than against the run.
"""
from __future__ import annotations

import numpy as np

from services.goal_source import GoalList
from tests.test_imgep_driver import DIM, make, moving, snaps


def with_goals(texts, **kw):
    d, arc, ts = make(**kw)
    g = GoalList()
    for t in texts:
        g.add(t)
    d.goals = g
    return d, arc, g


def sources(arc):
    return [e.source for e in arc.entries]


def goals_of(arc):
    return [e.goal for e in arc.entries]


def axis_tile(n, axis, others=1):
    """A generation where tile 0 sits on `axis` and the rest sit elsewhere.

    FakeScorer puts a tile on axis (mean brightness % dim), so brightness IS
    the behaviour coordinate and a goal's axis is reachable on demand.
    """
    return snaps(n, [[axis] + [others] * (n - 1),
                     [axis + DIM] + [others + DIM] * (n - 1)])


def test_a_tile_that_beats_the_archive_for_a_goal_is_kept():
    """Separation refuses the whole batch; the record still gets through."""
    d, arc, _ = with_goals(["alpha"], seed_n=4, min_separation=0.9)
    d.tell(d.ask(4), moving(4))              # something to beat
    before = len(arc)
    # FakeScorer assigns text axes in first-seen order, and the distractors are
    # embedded first, so "alpha" is not axis 0. Sweep to find its axis.
    d.goals.ensure_embedded(d.scorer)
    axis = int(np.argmax(d.goals.enabled_goals()[0].embedding))
    d.tell(d.ask(4), axis_tile(4, axis))
    assert "record" in sources(arc)[before:]


def test_the_record_carries_the_goal_it_beat():
    """Not the goal the run happens to be chasing - that is the only way to
    find it again afterwards."""
    d, arc, _ = with_goals(["alpha"], seed_n=4, min_separation=0.9)
    d.tell(d.ask(4), moving(4))
    d.goals.ensure_embedded(d.scorer)
    axis = int(np.argmax(d.goals.enabled_goals()[0].embedding))
    before = len(arc)
    d.tell(d.ask(4), axis_tile(4, axis))
    recs = [e for e in arc.entries[before:] if e.source == "record"]
    assert recs and recs[0].goal == "alpha"


def test_a_record_only_counts_once():
    """It is a ratchet against the archive: repeating the same tile cannot
    beat the entry it just created."""
    d, arc, _ = with_goals(["alpha"], seed_n=4, min_separation=0.9)
    d.tell(d.ask(4), moving(4))
    d.goals.ensure_embedded(d.scorer)
    axis = int(np.argmax(d.goals.enabled_goals()[0].embedding))
    d.tell(d.ask(4), axis_tile(4, axis))
    n = sources(arc).count("record")
    for _ in range(3):
        d.tell(d.ask(4), axis_tile(4, axis))
    assert sources(arc).count("record") == n


def test_records_happen_during_expansion_not_just_expeditions():
    """The whole point: the run is not chasing this goal."""
    d, arc, _ = with_goals(["alpha"], seed_n=1, min_separation=0.9,
                           expansion_between=0)
    d.tell(d.ask(4), moving(4))
    assert d.regime == "expansion"
    d.goals.ensure_embedded(d.scorer)
    axis = int(np.argmax(d.goals.enabled_goals()[0].embedding))
    before = len(arc)
    d.tell(d.ask(4), axis_tile(4, axis))
    assert d.regime == "expansion"
    assert "record" in sources(arc)[before:]


def test_a_disabled_goal_keeps_no_records():
    """The enabled checkbox is the only switch there is, so it has to mean
    'this goal is not in play'."""
    d, arc, g = with_goals(["alpha"], seed_n=4, min_separation=0.9)
    d.tell(d.ask(4), moving(4))
    d.goals.ensure_embedded(d.scorer)
    axis = int(np.argmax(d.goals.enabled_goals()[0].embedding))
    g.set_enabled(0, False)
    before = len(arc)
    d.tell(d.ask(4), axis_tile(4, axis))
    assert "record" not in sources(arc)[before:]


def test_a_blank_tile_cannot_set_a_record():
    """Viability keeps its veto here too. A black frame is not a match."""
    d, arc, _ = with_goals(["alpha"], seed_n=4, min_separation=0.9)
    d.tell(d.ask(4), moving(4))
    before = len(arc)
    d.tell(d.ask(4), snaps(4, [[0] * 4, [0] * 4]))
    assert "record" not in sources(arc)[before:]


def test_a_settled_tile_can_still_set_a_record():
    """Liveness is a floor on the bulk, not a veto over a chosen entry - the
    same rule the summit follows."""
    d, arc, _ = with_goals(["alpha"], seed_n=4, min_separation=0.9)
    d.tell(d.ask(4), moving(4))
    d.goals.ensure_embedded(d.scorer)
    axis = int(np.argmax(d.goals.enabled_goals()[0].embedding))
    d.liveness_min = 0.5                     # far above anything real
    before = len(arc)
    # axis + DIM, not axis: brightness lands on the same one-hot axis either
    # way, but is_viable_tile rejects a mean below 2 and the low axes are
    # exactly there. A test that used `axis` would pass or fail on which axis
    # FakeScorer happened to assign the goal.
    d.tell(d.ask(4), snaps(4, [[axis + DIM] * 4, [axis + DIM] * 4]))   # frozen
    assert "record" in sources(arc)[before:]


def test_no_goals_means_no_records_and_no_crash():
    d, arc, _ = with_goals([], seed_n=4, min_separation=0.9)
    for _ in range(3):
        d.tell(d.ask(4), moving(4))
    assert "record" not in sources(arc)
    assert d.status()["n_records"] == 0


def test_a_driver_without_a_goal_list_still_runs():
    """tools/ scripts and the older tests build one with goals=None."""
    d, arc, _ = make(seed_n=4, min_separation=0.9)
    assert d.goals is None
    d.tell(d.ask(4), moving(4))
    assert len(arc) > 0


def test_an_empty_archive_sets_no_records():
    """There is no incumbent to beat, and bootstrap admits generously anyway;
    stamping the first generation as 'record' would only mislabel it."""
    d, arc, _ = with_goals(["alpha"], seed_n=4, min_separation=0.9)
    d.tell(d.ask(4), moving(4))
    assert "record" not in sources(arc)


def test_the_count_is_reported_and_reset():
    d, _, _ = with_goals(["alpha"], seed_n=4, min_separation=0.9)
    d.tell(d.ask(4), moving(4))
    d.goals.ensure_embedded(d.scorer)
    axis = int(np.argmax(d.goals.enabled_goals()[0].embedding))
    d.tell(d.ask(4), axis_tile(4, axis))
    assert d.status()["n_records"] >= 1
    d.reset()
    assert d.status()["n_records"] == 0


def test_every_goal_keeps_its_own_best_and_no_tile_is_admitted_twice():
    """Two goals can be won by two different tiles, and both are kept - each
    goal's record book is its own. What must never happen is one tile going
    into the archive twice, which is why _goal_records is keyed BY TILE."""
    d, arc, _ = with_goals(["alpha", "beta"], seed_n=4, min_separation=0.9)
    d.tell(d.ask(4), moving(4))
    d.goals.ensure_embedded(d.scorer)
    axis = int(np.argmax(d.goals.enabled_goals()[0].embedding))
    before = len(arc)
    d.tell(d.ask(4), axis_tile(4, axis))
    added = arc.entries[before:]
    assert len(added) == len({(e.gen, e.tile) for e in added}), "a tile went in twice"
    assert {e.goal for e in added if e.source == "record"} <= {"alpha", "beta"}


def test_a_tile_is_credited_to_one_goal_only():
    """_goal_records is a dict keyed by tile, so a tile that tops several goals
    is admitted once, carrying the goal it beat by the largest margin."""
    d, _, _ = with_goals(["alpha", "beta"], seed_n=4, min_separation=0.9)
    d.tell(d.ask(4), moving(4))
    d.goals.ensure_embedded(d.scorer)
    b = np.zeros((4, DIM), dtype=np.float32)
    b[0] = 1.0 / np.sqrt(DIM)                # tile 0 matches everything equally
    ok = np.array([True, False, False, False])
    recs = d._goal_records(b, ok)
    assert set(recs) <= {0}, "only the one eligible tile can hold a record"
