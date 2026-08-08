import numpy as np
import pytest

from services.archive import Archive, Candidate
from services.archive_io import ArchiveStore
from services.goal_source import Goal, GoalList, latent_goal


def _unit(a):
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)


class FakeScorer:
    """One axis per distinct prompt, so goals are trivially distinguishable.

    Axes are assigned in first-seen order rather than by hash(). Python
    randomises string hashing per process, so `hash(p) % dim` collides between
    two prompts about a quarter of the time at dim=4 - measured, not
    hypothetical - which would make the least_matched test fail roughly one run
    in four for reasons having nothing to do with the code under test.
    """

    def __init__(self, dim=4):
        self.dim = dim
        self.calls = 0
        self._axis: dict[str, int] = {}

    def _axis_of(self, prompt):
        if prompt not in self._axis:
            self._axis[prompt] = len(self._axis) % self.dim
        return self._axis[prompt]

    def embed_text(self, prompts):
        self.calls += 1
        out = np.zeros((len(prompts), self.dim), dtype=np.float32)
        for i, p in enumerate(prompts):
            out[i, self._axis_of(p)] = 1.0
        return _unit(out)


def archive_with(vectors, dim=4, novelties=None):
    a = Archive(store=None, dim=dim, seed_n=0, liveness_min=0.0, capacity=100)
    a.threshold.value = 0.0
    for j, v in enumerate(vectors):
        e = np.zeros(dim, dtype=np.float32)
        e[: len(v)] = v
        a.consider(
            Candidate(brain=np.zeros((10, 8), np.float32),
                      physics=np.zeros(8, np.float32),
                      embedding=_unit(e[None])[0], liveness=1.0, spec="brain:80"),
            novelty=(novelties[j] if novelties else 1.0),
        )
    return a


# ---- the goal list ------------------------------------------------------

def test_add_remove_and_reorder():
    g = GoalList()
    g.add("coral reef")
    g.add("lightning")
    assert [i["text"] for i in g.items] == ["coral reef", "lightning"]
    g.move(0, +1)
    assert [i["text"] for i in g.items] == ["lightning", "coral reef"]
    g.remove(0)
    assert [i["text"] for i in g.items] == ["coral reef"]


def test_move_at_the_boundary_is_a_no_op():
    g = GoalList()
    g.add("a")
    g.add("b")
    g.move(0, -1)
    g.move(1, +1)
    assert [i["text"] for i in g.items] == ["a", "b"]


def test_new_goals_are_enabled_and_can_be_disabled():
    g = GoalList()
    g.add("a")
    assert g.items[0]["enabled"] is True
    g.set_enabled(0, False)
    assert g.enabled_items() == []


def test_blank_and_duplicate_goals_are_ignored():
    g = GoalList()
    assert g.add("   ") is False
    assert g.add("coral") is True
    assert g.add("coral") is False
    assert len(g.items) == 1


def test_next_goal_cycles_round_robin_over_enabled_entries():
    g = GoalList()
    for t in ("a", "b", "c"):
        g.add(t)
    g.set_enabled(1, False)
    g.ensure_embedded(FakeScorer())
    assert [g.next_goal().text for _ in range(4)] == ["a", "c", "a", "c"]


def test_next_goal_is_none_when_the_list_is_empty_or_all_disabled():
    g = GoalList()
    assert g.next_goal() is None
    g.add("a")
    g.set_enabled(0, False)
    g.ensure_embedded(FakeScorer())
    assert g.next_goal() is None


def test_goal_embeddings_are_cached_until_the_text_changes():
    s = FakeScorer()
    g = GoalList()
    g.add("a")
    g.ensure_embedded(s)
    g.ensure_embedded(s)
    assert s.calls == 1, "re-embedding an unchanged list wastes a text pass"
    g.add("b")
    g.ensure_embedded(s)
    assert s.calls == 2


def test_a_goal_embedding_is_unit_norm():
    g = GoalList()
    g.add("coral")
    g.ensure_embedded(FakeScorer())
    goal = g.next_goal()
    assert goal.kind == "text"
    assert np.linalg.norm(goal.embedding) == pytest.approx(1.0, abs=1e-5)


def test_least_matched_picks_the_goal_the_archive_cannot_reach():
    """Ground truth: the archive fans out around goal 0 and barely brushes
    goal 1, so goal 1 is the one worth an expedition.

    The archive needs real SPREAD for this question to mean anything - three
    copies of one vector give both goals a zero-variance, zero-reach
    distribution and the answer is a coin toss.
    """
    g = GoalList()
    g.add("reachable")
    g.add("unreachable")
    g.ensure_embedded(FakeScorer())
    a = archive_with([[1.0, 0.0, 0.0, 0.0],
                      [0.9, 0.05, 0.4, 0.0],
                      [0.7, 0.02, 0.7, 0.0],
                      [0.0, 0.10, 1.0, 0.0]])
    picked = g.least_matched(a.embeddings)
    assert picked.text == "unreachable"


def test_least_matched_is_not_fooled_by_a_constant_affinity_offset():
    """The modality gap is a per-phrase constant added to every image's cosine.
    Removing the archive mean removes it; anything that survives that shift is
    ranking the phrase rather than the archive."""
    g = GoalList()
    g.add("reachable")
    g.add("unreachable")
    g.ensure_embedded(FakeScorer())
    a = archive_with([[1.0, 0.0, 0.0, 0.0],
                      [0.9, 0.05, 0.4, 0.0],
                      [0.7, 0.02, 0.7, 0.0],
                      [0.0, 0.10, 1.0, 0.0]])
    e = a.embeddings
    baseline = np.full((len(e), 1), 0.5, dtype=np.float32)
    shifted = _unit(np.concatenate([e, baseline], axis=1))[:, : e.shape[1]]
    assert g.least_matched(shifted).text == "unreachable"


def test_least_matched_is_none_with_an_empty_archive():
    g = GoalList()
    g.add("a")
    g.ensure_embedded(FakeScorer())
    assert g.least_matched(np.zeros((0, 4), np.float32)) is None


def test_goals_persist_through_the_store(tmp_path):
    store = ArchiveStore(tmp_path)
    g = GoalList(store=store)
    g.add("coral reef")
    g.set_enabled(0, False)
    g.save()
    store.close()

    g2 = GoalList(store=ArchiveStore(tmp_path))
    g2.load()
    assert g2.items == [{"text": "coral reef", "enabled": False}]


# ---- latent goals -------------------------------------------------------

def test_latent_goal_is_unit_norm_and_labelled():
    a = archive_with([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    goal = latent_goal(a, np.random.default_rng(0))
    assert goal.kind == "latent"
    assert goal.text == ""
    assert np.linalg.norm(goal.embedding) == pytest.approx(1.0, abs=1e-5)


def test_latent_goal_points_further_out_than_its_source():
    """The whole point: extrapolate past the frontier, do not sit on it."""
    a = archive_with([[1, 0, 0, 0], [1, 0.1, 0, 0], [1, 0.2, 0, 0], [0, 0, 1, 0]],
                     novelties=[0.1, 0.1, 0.1, 0.9])
    c = a.centroid()
    goal = latent_goal(a, np.random.default_rng(0), alpha=8.0, beta=0.5)
    src = a.embeddings[3]                      # the novel outlier is the source
    assert float(goal.embedding @ c) < float(src @ c)


def test_latent_goal_on_an_empty_archive_is_none():
    a = Archive(store=None, dim=4)
    assert latent_goal(a, np.random.default_rng(0)) is None


def test_latent_goal_on_a_single_entry_returns_that_entry():
    a = archive_with([[1, 0, 0, 0]])
    goal = latent_goal(a, np.random.default_rng(0))
    assert np.allclose(goal.embedding, a.embeddings[0], atol=1e-5)
