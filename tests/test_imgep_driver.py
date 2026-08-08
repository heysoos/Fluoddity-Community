import numpy as np
import pytest

from services.archive import Archive
from services.genome_spec import BRAIN_PHYSICS_SPEC, BRAIN_SPEC
from services.goal_source import GoalList
from services.imgep_driver import ImgepDriver
from services.tournament_service import TournamentService

DIM = 8


class FakeScorer:
    """Embeds each tile onto an axis chosen by its mean brightness, so tests
    control exactly where a tile lands in behaviour space.

    Text axes are assigned in first-seen order rather than by hash(): Python
    randomises string hashing per process, so hash(p) % dim collides between two
    prompts often enough to make a goal test fail for unrelated reasons.
    """

    def __init__(self, dim=DIM):
        self.dim = dim
        self._axis: dict[str, int] = {}

    def embed(self, images, n_views=1):
        n = len(images)
        out = np.zeros((n, self.dim), dtype=np.float32)
        for i, img in enumerate(images):
            out[i, int(img.reshape(-1).mean()) % self.dim] = 1.0
        return out

    def embed_text(self, prompts):
        out = np.zeros((len(prompts), self.dim), dtype=np.float32)
        for i, p in enumerate(prompts):
            if p not in self._axis:
                self._axis[p] = len(self._axis) % self.dim
            out[i, self._axis[p]] = 1.0
        return out


def make(grid=2, **kw):
    ts = TournamentService(grid=grid)
    ts.init_population()
    seed_n = kw.pop("seed_n", 4)
    liveness_min = kw.pop("liveness_min", 0.0)
    arc = Archive(store=None, dim=DIM, seed_n=seed_n,
                  liveness_min=liveness_min, capacity=100)
    arc.threshold.value = 0.0
    d = ImgepDriver(ts, FakeScorer(), arc, rng=np.random.default_rng(0))
    # The DRIVER owns these two: tell() pushes them onto the archive every
    # generation, so setting them only on the archive is silently undone on the
    # first tell and the test would be measuring the defaults.
    d.seed_n = seed_n
    d.liveness_min = liveness_min
    for k, v in kw.items():
        setattr(d, k, v)
    return d, arc, ts


def snaps(n, values):
    """One (n,224,224,3) crop array per snapshot; values[s][i] sets tile i."""
    out = []
    for row in values:
        a = np.zeros((n, 224, 224, 3), dtype=np.uint8)
        for i, v in enumerate(row):
            a[i] = v
        out.append(a)
    return out


def moving(n, base=10):
    """Two snapshots whose brightness differs, for a plain two-frame rollout."""
    return snaps(n, [[base + i for i in range(n)], [base + 40 + i for i in range(n)]])


# ---- regimes -----------------------------------------------------------

def test_starts_in_bootstrap_with_an_empty_archive():
    d, _, _ = make()
    assert d.regime == "bootstrap"


def test_bootstrap_asks_for_scattered_vectors_not_archive_children():
    d, _, _ = make()
    z = d.ask(4)
    assert z.shape == (4, BRAIN_SPEC.dim)
    assert z.dtype == np.float32
    assert z.std() > 0.1, "a bootstrap population must be scattered"


def test_regime_becomes_expansion_once_the_archive_reaches_seed_n():
    d, arc, _ = make(seed_n=4)
    z = d.ask(4)
    d.tell(z, moving(4))
    assert len(arc) == 4
    assert d.regime == "expansion"


def test_an_empty_archive_stays_in_bootstrap_even_at_seed_n_zero():
    """There is nothing to expand FROM. Without this guard, parent sampling
    raises on an empty archive rather than falling back."""
    d, arc, _ = make(seed_n=0)
    assert len(arc) == 0
    assert d.regime == "bootstrap"
    assert d.ask(4).shape == (4, BRAIN_SPEC.dim)


def test_expansion_children_stay_near_their_parents():
    d, arc, _ = make(seed_n=4, sigma_expand=0.01)
    d.tell(d.ask(4), moving(4))
    assert d.regime == "expansion"
    z = d.ask(4)
    # every child decodes to a brain close to some archived parent
    for row in z:
        brain = BRAIN_SPEC.decode(row)["brain"]
        gaps = np.abs(arc.brains - brain).max(axis=(1, 2))
        assert gaps.min() < 0.3


def test_sigma_expand_controls_how_far_children_travel():
    near, arc_n, _ = make(seed_n=4, sigma_expand=0.01)
    near.tell(near.ask(4), moving(4))
    far, arc_f, _ = make(seed_n=4, sigma_expand=1.0)
    far.tell(far.ask(4), moving(4))
    assert far.ask(16).std() > near.ask(16).std()


# ---- tell / the admission call site ------------------------------------

def test_tell_admits_one_entry_per_viable_live_novel_tile():
    d, arc, _ = make(seed_n=0)
    arc.threshold.value = 0.0
    d.tell(d.ask(4), moving(4))
    assert len(arc) == 4


def test_a_dead_tile_is_rejected_by_the_viability_gate():
    d, arc, _ = make(seed_n=0)
    s = moving(4)
    for a in s:
        a[2] = 0                       # tile 2 is pure black
    d.tell(d.ask(4), s)
    assert len(arc) == 3
    assert all(e.tile != 2 for e in arc.entries)


def test_a_frozen_tile_is_rejected_by_the_liveness_gate():
    d, arc, _ = make(seed_n=0, liveness_min=0.5)
    same = [v for v in range(10, 14)]
    d.tell(d.ask(4), snaps(4, [same, same]))    # identical snapshots
    assert len(arc) == 0


def test_a_genuinely_changing_tile_passes_the_liveness_gate():
    """The other direction of the gate. moving() lands both of its snapshots on
    the same FakeScorer axis - 40 % DIM is 0 - so it scores liveness 0 and
    proves nothing here; these two snapshots really do land apart."""
    d, arc, _ = make(seed_n=0, liveness_min=0.5)
    d.tell(d.ask(4), snaps(4, [[10, 11, 12, 13], [14, 15, 16, 17]]))
    assert len(arc) == 4
    assert all(e.liveness > 0.5 for e in arc.entries)


def test_the_liveness_gate_is_disabled_with_a_single_snapshot():
    """One snapshot is no evidence of change; rejecting everything would be
    wrong, not conservative."""
    d, arc, _ = make(seed_n=0, liveness_min=0.5)
    d.tell(d.ask(4), snaps(4, [[10, 11, 12, 13]]))
    assert len(arc) == 4


def test_tell_with_no_snapshots_returns_zeros_and_admits_nothing():
    d, arc, _ = make()
    assert np.array_equal(d.tell(d.ask(4), []), np.zeros(4, np.float32))
    assert len(arc) == 0


def test_expansion_returns_novelty_as_the_display_score():
    d, arc, _ = make(seed_n=0)
    out = d.tell(d.ask(4), moving(4))
    assert out.shape == (4,)
    assert np.all(out >= 0.0)
    assert d.status()["score_label"] == "novelty"


def test_a_selected_tile_is_pinned_and_the_selection_is_cleared():
    d, arc, ts = make(seed_n=0, liveness_min=0.9)
    ts.toggle_select(1)
    same = [10, 11, 12, 13]
    d.tell(d.ask(4), snaps(4, [same, same]))     # everything frozen
    assert len(arc) == 1, "only the pinned tile gets in"
    assert arc.entries[0].pinned is True
    assert arc.entries[0].tile == 1
    assert ts.selected == set()


def test_entries_record_their_generation_tile_and_spec():
    d, arc, _ = make(seed_n=0)
    d.gen = 7
    d.tell(d.ask(4), moving(4))
    assert {e.tile for e in arc.entries} == {0, 1, 2, 3}
    assert all(e.gen == 7 for e in arc.entries)
    assert all(e.spec == "brain:80" for e in arc.entries)


def test_refresh_runs_every_generation():
    d, arc, _ = make(seed_n=0, refresh_per_gen=4)
    d.tell(d.ask(4), moving(4))
    before = [e.novelty for e in arc.entries]
    d.tell(d.ask(4), moving(4, base=100))
    assert [e.novelty for e in arc.entries[:4]] != before


# ---- physics -----------------------------------------------------------

def test_physics_genes_are_stored_as_absolute_values():
    from services.physics_genome import PHYSICS_PARAMS, decode_physics

    d, arc, _ = make(seed_n=0)
    d.set_spec(BRAIN_PHYSICS_SPEC)
    d.physics_enabled = True
    d.physics_origin = {n: (lo + hi) / 2 for n, _g, lo, hi in PHYSICS_PARAMS}
    z = d.ask(4)
    d.tell(z, moving(4))
    expected = decode_physics(z[0][BRAIN_SPEC.dim:], d.physics_origin)
    stored = arc.physics[[e.tile for e in arc.entries].index(0)]
    for j, (n, _g, _lo, _hi) in enumerate(PHYSICS_PARAMS):
        assert stored[j] == pytest.approx(expected[n], abs=1e-4)
    assert all(e.spec == "brain:80,physics:8" for e in arc.entries)


def test_a_brain_only_parent_gets_zero_physics_genes_in_a_physics_run():
    """z=0 for the physics block decodes to the current origin exactly, so a
    brain-only archive entry is well-defined rather than an error."""
    d, arc, _ = make(seed_n=4)
    d.tell(d.ask(4), moving(4))                  # 4 brain-only entries
    d.set_spec(BRAIN_PHYSICS_SPEC)
    d.physics_enabled = True
    d.sigma_expand = 0.0                         # no mutation, so z is exact
    z = d.ask(4)
    assert np.allclose(z[:, BRAIN_SPEC.dim:], 0.0, atol=1e-6)


def test_set_spec_changes_the_ask_dimension():
    d, _, _ = make()
    assert d.ask(4).shape[1] == BRAIN_SPEC.dim
    d.set_spec(BRAIN_PHYSICS_SPEC)
    assert d.ask(4).shape[1] == BRAIN_PHYSICS_SPEC.dim


# ---- lifecycle ---------------------------------------------------------

def test_reset_clears_the_search_but_keeps_the_archive():
    d, arc, _ = make(seed_n=0)
    d.tell(d.ask(4), moving(4))
    n = len(arc)
    d.reset()
    assert len(arc) == n, "the archive is the product; reset is about the search"
    assert d.regime in ("bootstrap", "expansion")


def test_status_reports_what_the_ui_needs():
    d, _, _ = make()
    st = d.status()
    assert set(st) >= {"regime", "goal", "archive_size", "threshold",
                       "admission_rate", "score_label", "sigma"}


# ---- expeditions -------------------------------------------------------

def seeded(**kw):
    """A driver past bootstrap, with 8 archive entries.

    Tests that count expedition generations pass expansion_between=0 so the
    cadence cannot fire during these two warm-up tells and quietly consume part
    of the expedition they are trying to measure.
    """
    d, arc, ts = make(seed_n=4, **kw)
    d.tell(d.ask(4), moving(4))
    d.tell(d.ask(4), moving(4, base=100))
    return d, arc, ts


def test_no_expedition_fires_while_expansion_between_is_zero():
    d, _, _ = seeded(expansion_between=0)
    for _ in range(10):
        d.tell(d.ask(4), moving(4))
    assert d.regime == "expansion"


def test_an_expedition_fires_after_expansion_between_generations():
    d, _, _ = seeded(expansion_between=2, expedition_gens=5, latent_share=1.0)
    for _ in range(3):
        d.tell(d.ask(4), moving(4))
    assert d.regime == "expedition"


def test_an_expedition_lasts_exactly_expedition_gens_generations():
    d, arc, _ = seeded(expansion_between=0, expedition_gens=3)
    d.start_expedition_with(arc.embeddings[0].copy(), kind="latent", text="")
    for _ in range(3):
        assert d.regime == "expedition"
        d.tell(d.ask(4), moving(4))
    assert d.regime == "expansion"


def test_an_expedition_builds_a_fresh_optimizer_per_goal():
    d, arc, _ = seeded(expansion_between=0, expedition_gens=2)
    d.start_expedition_with(arc.embeddings[0].copy(), kind="latent", text="")
    first = d.optimizer
    assert first is not None and first.popsize == 4
    for _ in range(2):
        d.tell(d.ask(4), moving(4))
    assert d.regime == "expansion"
    d.start_expedition_with(arc.embeddings[1].copy(), kind="latent", text="")
    assert d.optimizer is not first, "a new goal must not inherit a covariance"


def test_an_expedition_seeds_at_the_archive_entry_nearest_the_goal():
    d, arc, _ = seeded(expansion_between=0)
    goal = arc.embeddings[2].copy()
    d.start_expedition_with(goal, kind="chase", text="")
    assert d._x0_index == 2


def test_expedition_fitness_is_alignment_with_the_goal():
    d, arc, _ = seeded(expansion_between=0, expedition_gens=5)
    goal = np.zeros(DIM, dtype=np.float32)
    goal[3] = 1.0
    d.start_expedition_with(goal, kind="latent", text="")
    z = d.ask(4)
    # FakeScorer puts tile i on axis (mean brightness % DIM); brightness 3 -> axis 3
    out = d.tell(z, snaps(4, [[1, 2, 3, 4], [41, 42, 43, 44]]))
    assert d.status()["score_label"] == "goal match"
    assert int(np.argmax(out)) == 2, "the tile landing on the goal axis wins"


def test_expedition_tiles_still_reach_the_archive():
    """Deliberate deviation from E&E: the path to a goal is territory too."""
    d, arc, _ = seeded(expansion_between=0, expedition_gens=5)
    before = len(arc)
    d.start_expedition_with(arc.embeddings[0].copy(), kind="latent", text="")
    d.tell(d.ask(4), moving(4, base=150))
    assert len(arc) > before
    assert any(e.source == "expedition" for e in arc.entries)


def test_latent_share_of_one_always_picks_a_latent_goal():
    g = GoalList()
    g.add("coral")
    g.ensure_embedded(FakeScorer())
    d, _, _ = seeded(expansion_between=0, latent_share=1.0)
    d.goals = g
    d.start_expedition()
    assert d._goal.kind == "latent"


def test_latent_share_of_zero_uses_the_text_goal_list():
    g = GoalList()
    g.add("coral")
    g.ensure_embedded(FakeScorer())
    d, _, _ = seeded(expansion_between=0, latent_share=0.0)
    d.goals = g
    d.start_expedition()
    assert d._goal.kind == "text"
    assert d._goal.text == "coral"


def test_an_empty_goal_list_falls_back_to_latent():
    d, _, _ = seeded(expansion_between=0, latent_share=0.0)
    d.goals = GoalList()
    d.start_expedition()
    assert d._goal.kind == "latent"


def test_goal_order_least_matched_is_selectable():
    g = GoalList()
    g.add("a")
    g.add("b")
    g.ensure_embedded(FakeScorer())
    d, _, _ = seeded(expansion_between=0, latent_share=0.0)
    d.goals = g
    d.goal_order = "least_matched"
    d.start_expedition()
    assert d._goal.kind == "text"


def test_an_expedition_on_an_empty_archive_falls_back_to_expansion():
    d, arc, _ = make(seed_n=0)
    d.goals = GoalList()
    assert d.start_expedition() is False
    assert d.regime == "bootstrap"


def test_chase_starts_an_expedition_on_that_tiles_descriptor():
    d, arc, _ = seeded(expansion_between=0)
    d.tell(d.ask(4), snaps(4, [[1, 2, 3, 4], [41, 42, 43, 44]]))
    assert d.chase(2) is True
    assert d.regime == "expedition"
    assert d._goal.kind == "chase"
    assert float(d._goal.embedding[3]) == pytest.approx(1.0, abs=1e-5)


def test_chase_before_any_generation_is_refused():
    d, _, _ = make()
    assert d.chase(0) is False


def test_chase_pre_empts_the_expansion_cadence():
    d, _, _ = seeded(expansion_between=1000)
    d.tell(d.ask(4), moving(4))
    d.chase(0)
    assert d.regime == "expedition"


def test_the_goal_label_appears_in_status_and_on_entries():
    g = GoalList()
    g.add("coral reef")
    g.ensure_embedded(FakeScorer())
    d, arc, _ = seeded(expansion_between=0, latent_share=0.0, expedition_gens=5)
    d.goals = g
    d.start_expedition()
    assert d.status()["goal"] == "coral reef"
    d.tell(d.ask(4), moving(4, base=150))
    assert any(e.goal == "coral reef" for e in arc.entries)


def test_a_grid_change_ends_the_expedition_rather_than_crashing():
    """cmaes fixes popsize at construction and asserts on it in tell()."""
    d, arc, ts = seeded(expansion_between=0, expedition_gens=50)
    d.start_expedition_with(arc.embeddings[0].copy(), kind="latent", text="")
    d.ask(4)
    ts.set_grid(4)
    z = d.ask(16)
    assert z.shape == (16, BRAIN_SPEC.dim)
    d.tell(z, moving(16))          # must not raise


def test_set_spec_ends_an_active_expedition():
    d, arc, _ = seeded(expansion_between=0, expedition_gens=50)
    d.start_expedition_with(arc.embeddings[0].copy(), kind="latent", text="")
    d.set_spec(BRAIN_PHYSICS_SPEC)
    assert d.regime != "expedition"
    assert d.optimizer is None


def test_reset_ends_an_expedition_but_keeps_the_archive():
    d, arc, _ = seeded(expansion_between=0, expedition_gens=50)
    n = len(arc)
    d.start_expedition_with(arc.embeddings[0].copy(), kind="latent", text="")
    d.reset()
    assert d.regime != "expedition"
    assert len(arc) == n
