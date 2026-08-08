import numpy as np
import pytest

from services.archive import Archive, Candidate
from services.archive_io import ArchiveStore
from services.archive_projection import Projection
from services.expedition_fitness import IMAGE_LOGIT_SCALE, contrastive
from services.goal_source import LATENT_DIMS, Goal, GoalList, latent_goal


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

def cone_archive(n=300, dim=256, seed=0):
    """An anisotropic cone in a space much larger than LATENT_DIMS.

    All three properties are load-bearing, and a fixture missing any of them
    makes the whitened construction look identical to the one it replaces:

      TIGHT       the real archive's mean pairwise <b, b'> is 0.897. Geometry
                  on a spread-out cloud says nothing about the app.
      ANISOTROPIC half the real variance is in 3 of 512 components. Whitening
                  an isotropic cloud is a uniform rescale, i.e. a no-op, and the
                  goal collapses back to plain radial extrapolation.
      dim >> 8    the goal lies entirely inside the 8-d subspace, while every
                  entry keeps most of its energy outside it. That is what makes
                  <e, g> rank differently from <e, seed>, and the effect scales
                  with dim/LATENT_DIMS: measured at 7/20 goals leaving room past
                  the seed at dim 64, 10/20 at 128, 14/20 at 256, and 54/60 on
                  the real archive at dim 512.
    """
    rng = np.random.default_rng(seed)
    axis = _unit(rng.normal(size=dim).astype(np.float32)[None])[0]
    scales = (0.09 * (0.985 ** np.arange(dim))).astype(np.float32)
    vecs = _unit(axis + rng.normal(size=(n, dim)).astype(np.float32) * scales)
    # Its own Archive rather than archive_with(): that helper caps capacity at
    # 100, which would silently evict two thirds of this fixture and quietly
    # weaken every measurement taken on it.
    a = Archive(store=None, dim=dim, seed_n=0, liveness_min=0.0, capacity=n)
    a.threshold.value = 0.0
    for v in vecs:
        a.consider(
            Candidate(brain=np.zeros((10, 8), np.float32),
                      physics=np.zeros(8, np.float32),
                      embedding=v.astype(np.float32), liveness=1.0,
                      spec="brain:80"),
            novelty=1.0,
        )
    assert len(a) == n, "the fixture must not be evicting itself"
    return a


def beaten_by(archive, goal):
    """How many archived entries out-score the entry the expedition SEEDS at.

    Archive.nearest() returns argmax(embeddings @ goal), so the seed is the
    archive's best entry under the goal by construction - for any goal. 0 here
    means CMA-ES starts on the optimum of its own objective and every move it
    can make scores worse. That is the bug, stated as a number.
    """
    e = archive.embeddings
    seed = int(np.argmax(e @ goal))
    fit = contrastive(e[None, :, :], goal, archive.centroid()[None, :],
                      logit_scale=IMAGE_LOGIT_SCALE)
    return int((fit > fit[seed]).sum())


def old_goal(b, c):
    """g = normalise(b + 0.5(b - c)), the construction this replaced."""
    g = np.asarray(b, np.float32) + 0.5 * (np.asarray(b, np.float32)
                                           - np.asarray(c, np.float32))
    return (g / np.linalg.norm(g)).astype(np.float32)


def test_latent_goal_is_unit_norm_and_labelled():
    a = cone_archive()
    goal = latent_goal(a, np.random.default_rng(0), Projection(LATENT_DIMS))
    assert goal.kind == "latent"
    assert goal.text == ""
    assert np.linalg.norm(goal.embedding) == pytest.approx(1.0, abs=1e-5)


def test_the_old_construction_left_nothing_to_climb():
    """Kept as a regression witness. If someone reinstates centroid
    extrapolation, this is what they are reinstating."""
    a = cone_archive()
    c = a.centroid()
    beaten = [beaten_by(a, old_goal(a.embeddings[i], c)) for i in range(20)]
    assert max(beaten) == 0, "the seed was unbeatable under its own goal"


def test_the_whitened_goal_leaves_real_room_past_the_seed():
    """THE fix. Measured on the real archive: 0.0 entries beat the seed under
    the old construction, 23.8 under this one."""
    a = cone_archive()
    proj = Projection(LATENT_DIMS)
    beaten = [beaten_by(a, latent_goal(a, np.random.default_rng(t), proj).embedding)
              for t in range(20)]
    # 10/20, not 20/20: this fixture is dim 256 against the real archive's 512,
    # and the effect scales with dim/LATENT_DIMS (see cone_archive). Measured
    # 14/20 here and 54/60 on the real archive.
    assert sum(1 for b in beaten if b > 0) >= 10, \
        f"most goals must leave something above the seed, got {beaten}"


def test_the_goal_does_not_sit_on_an_entry_the_archive_already_has():
    """Extrapolate past the frontier, do not sit on it.

    Stated as REACH - the best score anything already archived gets - because
    that is the quantity the old construction pinned at ~0.987, i.e. an entry
    the archive was already holding. Measured on the real archive: 0.9871 for
    the old goal against 0.9613 for this one.

    Not stated as distance from the centroid: the goal lies inside the top-8
    subspace, which is naturally BETTER aligned with the centroid than a real
    entry is, since entries carry most of their energy outside it.
    """
    a = cone_archive()
    c = a.centroid()
    proj = Projection(LATENT_DIMS)
    new = [float((a.embeddings @ latent_goal(
        a, np.random.default_rng(t), proj).embedding).max()) for t in range(10)]
    old = [float((a.embeddings @ old_goal(a.embeddings[i], c)).max())
           for i in range(10)]
    assert np.mean(new) < np.mean(old)


def test_a_bigger_push_lands_further_from_the_centroid():
    a = cone_archive()
    c = a.centroid()
    near = latent_goal(a, np.random.default_rng(5), Projection(LATENT_DIMS),
                       push_sd=1.0)
    far = latent_goal(a, np.random.default_rng(5), Projection(LATENT_DIMS),
                      push_sd=6.0)
    assert float(far.embedding @ c) < float(near.embedding @ c)


def test_latent_goal_on_an_empty_archive_is_none():
    a = Archive(store=None, dim=4)
    assert latent_goal(a, np.random.default_rng(0), Projection(LATENT_DIMS)) is None


def test_a_projection_that_cannot_fit_yields_no_goal():
    """It must NOT fall back to the seed or to centroid extrapolation - both
    hand back a goal the seed already maximises, which is the bug. The caller
    falls through to a text goal instead."""
    a = archive_with([[1, 0, 0, 0]])
    assert latent_goal(a, np.random.default_rng(0), Projection(LATENT_DIMS)) is None


def test_no_projection_at_all_yields_no_goal():
    a = cone_archive()
    assert latent_goal(a, np.random.default_rng(0), None) is None
