import numpy as np
import pytest

from services.archive import Archive
from services.genome_spec import BRAIN_PHYSICS_SPEC, BRAIN_SPEC
from services.goal_source import LATENT_DIMS, GoalList
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
        self.calls = 0
        self.prompt_set_calls = 0

    def embed(self, images, n_views=1):
        n = len(images)
        out = np.zeros((n, self.dim), dtype=np.float32)
        for i, img in enumerate(images):
            out[i, int(img.reshape(-1).mean()) % self.dim] = 1.0
        return out

    def set_prompt(self, text, distractors=None):
        """Explore must NEVER call this - it writes _text_emb, which the Auto
        (CLIP) tab owns. Counted so a test can prove it stays untouched."""
        self.prompt_set_calls += 1

    def embed_text(self, prompts):
        self.calls += 1
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
    arc = Archive(store=None, dim=DIM, 
                  liveness_min=liveness_min, capacity=100)
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
        gaps = np.abs(arc.brains - brain.reshape(-1)).max(axis=1)
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
    d, arc, _ = make(seed_n=0, refresh_sweep_gens=1)
    d.tell(d.ask(4), moving(4))
    before = [e.novelty for e in arc.entries]
    d.tell(d.ask(4), moving(4, base=100))
    assert [e.novelty for e in arc.entries[:4]] != before


def test_the_refresh_count_is_a_fraction_of_the_archive():
    """A fixed count means the sweep period grows with the archive, and at
    grid 8 the old 64/generation matched the ADMISSION rate exactly - a sweep
    took a full turnover."""
    d, arc, _ = make(seed_n=0)
    d.refresh_sweep_gens = 10

    class _N:
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

    real = d.archive
    for n, want in ((0, 0), (5, 1), (100, 10), (4808, 481), (20000, 2000)):
        d.archive = _N(n)
        assert d._refresh_count() == want, f"n={n}"
    d.archive = real


def test_a_sweep_of_zero_generations_refreshes_nothing():
    """0 is off rather than a division by zero."""
    d, _, _ = make(seed_n=0)
    d.refresh_sweep_gens = 0
    assert d._refresh_count() == 0


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
    assert set(st) >= {"regime", "goal", "archive_size", "capacity",
                       "n_evicted", "admission_rate", "score_label", "sigma"}
    assert "threshold" not in st, "the adaptive threshold is gone"


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


def seeded_wide(**kw):
    """As seeded(), but past the PCA's minimum.

    A latent goal is built from a Projection fit at LATENT_DIMS components,
    which needs strictly more rows than components. seeded() stops at 8 and
    would silently produce no latent goal at all. The app cannot hit this -
    expeditions require len(archive) >= seed_n, which defaults to 256.
    """
    d, arc, ts = seeded(**kw)
    d.tell(d.ask(4), moving(4, base=150))
    assert len(arc) > LATENT_DIMS, "fixture must clear the projection's minimum"
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


def test_an_expedition_seeds_near_the_archive_entry_matching_the_goal():
    """Sampled with p proportional to fit^alpha rather than argmaxed, so this is
    a statement about the mode, not about one draw. The goal here IS entry 2's
    own descriptor, so entry 2 should dominate."""
    d, arc, _ = seeded(expansion_between=0)
    goal = arc.embeddings[2].copy()
    picks = []
    for _ in range(200):
        d.start_expedition_with(goal, kind="chase", text="")
        picks.append(d._x0_index)
    assert max(set(picks), key=picks.count) == 2
    assert picks.count(2) > 100, f"entry 2 must dominate, got {picks.count(2)}/200"


def test_expedition_fitness_ranks_the_tile_that_matches_the_goal_first():
    d, arc, _ = seeded(expansion_between=0, expedition_gens=5)
    goal = np.zeros(DIM, dtype=np.float32)
    goal[3] = 1.0
    d.start_expedition_with(goal, kind="latent", text="")
    z = d.ask(4)
    # FakeScorer puts tile i on axis (mean brightness % DIM); brightness 3 -> axis 3
    out = d.tell(z, snaps(4, [[1, 2, 3, 4], [41, 42, 43, 44]]))
    assert d.status()["score_label"] == "goal match"
    assert int(np.argmax(out)) == 2, "the tile landing on the goal axis wins"
    assert np.all((out >= 0.0) & (out <= 1.0)), \
        "a contrastive probability, not a raw cosine"


def test_expedition_fitness_is_contrastive_not_raw_alignment():
    """Raw <b, g> was the old objective. It saturates: the archive's mean
    pairwise similarity is 0.897, so every reachable goal is already matched to
    0.96-0.99 and the usable range is a couple of percent."""
    from services.expedition_fitness import IMAGE_LOGIT_SCALE, contrastive

    d, arc, _ = seeded(expansion_between=0, expedition_gens=5)
    goal = np.zeros(DIM, dtype=np.float32)
    goal[3] = 1.0
    d.start_expedition_with(goal, kind="latent", text="")
    z = d.ask(4)
    out = d.tell(z, snaps(4, [[1, 2, 3, 4], [41, 42, 43, 44]]))

    refs, scale = d._references()
    assert scale == IMAGE_LOGIT_SCALE, "a latent goal is an IMAGE goal"
    assert refs is not None and len(refs) == 1, "the archive centroid, alone"
    assert np.allclose(refs[0], arc.centroid(), atol=1e-5)
    # Reproduce it independently from the descriptors the archive kept.
    assert out.shape == (4,)


def test_a_text_goal_scores_against_the_distractors_at_clips_own_scale():
    from services.expedition_fitness import TEXT_LOGIT_SCALE

    g = GoalList()
    g.add("coral")
    scorer = FakeScorer()
    g.ensure_embedded(scorer)
    d, _, _ = seeded(expansion_between=0, latent_share=0.0)
    d.goals = g
    d.start_expedition()

    refs, scale = d._references()
    assert d._goal.kind == "text"
    assert scale == TEXT_LOGIT_SCALE
    assert refs is not None and len(refs) > 1, "the distractor set, not a centroid"


def test_the_distractors_are_embedded_once_for_the_whole_run():
    """Re-embedding them every generation would spend a text-encoder pass on a
    constant."""
    g = GoalList()
    g.add("coral")
    scorer = FakeScorer()
    g.ensure_embedded(scorer)
    d, _, _ = seeded(expansion_between=0, latent_share=0.0)
    d.goals = g
    d.start_expedition()
    before = scorer.calls
    for _ in range(3):
        d._references()
    assert scorer.calls == before, "cached, not re-embedded"


def test_the_scorers_prompt_cache_is_never_touched():
    """set_prompt() writes scorer._text_emb, which the Auto (CLIP) tab owns.
    Explore must not clobber the other mode's prompt."""
    g = GoalList()
    g.add("coral")
    scorer = FakeScorer()
    g.ensure_embedded(scorer)
    d, _, _ = seeded(expansion_between=0, latent_share=0.0)
    d.goals = g
    d.start_expedition()
    d._references()
    assert getattr(scorer, "prompt_set_calls", 0) == 0


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
    d, _, _ = seeded_wide(expansion_between=0, latent_share=1.0)
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
    d, _, _ = seeded_wide(expansion_between=0, latent_share=0.0)
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


# ---- where an expedition STARTS ----------------------------------------

def test_the_seed_is_chosen_by_the_same_objective_the_expedition_is_scored_on():
    """Archive.nearest() is argmax(embeddings @ goal), and for a text goal that
    ranking is degenerate: measured over 15 unrelated prompts on a real 4784-
    entry archive it returned 6 distinct seeds, one cyan NOISE TEXTURE winning 7
    of them. Noise carries a decent cosine to every phrase.

    Sampled, so pin the DISTRIBUTION: seeds must land well above what picking
    uniformly would give.
    """
    from services.expedition_fitness import contrastive

    g = GoalList()
    g.add("coral")
    g.ensure_embedded(FakeScorer())
    d, arc, _ = seeded_wide(expansion_between=0, latent_share=0.0)
    d.goals = g
    assert d.start_expedition() is True
    assert d._goal.kind == "text"

    refs, scale = d._references("text")
    fit = contrastive(arc.embeddings[None, :, :], d._goal.embedding, refs,
                      logit_scale=scale)
    picked = [fit[d._seed_index(d._goal.embedding, "text")] for _ in range(200)]
    assert float(np.mean(picked)) > float(fit.mean()),         "sampling must prefer entries that match the goal"


def test_the_seed_is_sampled_rather_than_argmaxed():
    """E&E picks a parent with p proportional to NOV^alpha; the seed follows the
    same rule and the same alpha. An argmax would send every expedition toward a
    given goal from the identical entry, so repeating a goal could only ever
    retrace one trajectory."""
    d, arc, _ = seeded_wide(expansion_between=0)
    goal = arc.embeddings[2].copy()
    seen = {d._seed_index(goal, "latent") for _ in range(200)}
    assert len(seen) > 1, "a deterministic seed makes a repeated goal pointless"


def test_alpha_zero_makes_seeding_uniform():
    """The Random-GA ablation, reachable from the UI without a second code path
    - exactly as it is for parent sampling."""
    d, arc, _ = seeded_wide(expansion_between=0)
    d.alpha = 0.0
    goal = arc.embeddings[2].copy()
    seen = {d._seed_index(goal, "latent") for _ in range(300)}
    assert len(seen) == len(arc), "every entry must be reachable at alpha 0"


def test_a_noise_magnet_loses_the_seed_to_the_distractor_set():
    """The failure in one line, and the mechanism that fixes it.

    High-frequency noise carries a middling cosine to EVERY phrase, so it wins
    argmax(e @ goal) for all of them - measured on the real archive, one cyan
    static tile was the seed for 7 of 15 unrelated prompts. What rejects it is
    that DEFAULT_DISTRACTORS contains "random noise" and "an abstract texture":
    the decoy scores higher against those than against the goal, so its softmax
    probability for the target collapses.

    Real entries never match a prompt the way this fixture's one-hot vectors
    match an axis, so the decoy has to out-align them on the goal too - that is
    exactly the situation raw cosine loses in.
    """
    d, arc, _ = seeded_wide(expansion_between=0)
    goal = np.eye(DIM, dtype=np.float32)[1]
    noise_distractor = np.eye(DIM, dtype=np.float32)[7]
    d._distractors = noise_distractor[None, :]

    ordinary = np.zeros(DIM, np.float32); ordinary[0] = 1.0; ordinary[1] = 0.2
    ordinary /= np.linalg.norm(ordinary)
    decoy = np.zeros(DIM, np.float32); decoy[1] = 0.4; decoy[7] = 0.9
    decoy /= np.linalg.norm(decoy)
    for j in range(len(arc)):
        arc._emb[j] = ordinary
    arc._emb[0] = decoy

    assert float(decoy @ goal) > float(ordinary @ goal),         "the decoy really does out-align everything on the goal"
    assert arc.nearest(goal) == 0, "so raw cosine seeds on it"
    # Sampled, so state it as a rate: the decoy floors to ~0 fitness and must
    # essentially never be drawn, rather than merely losing an argmax.
    picks = [d._seed_index(goal, "text") for _ in range(200)]
    assert picks.count(0) == 0, "the distractor set has to see through it"


def test_the_seed_falls_back_to_nearest_when_there_is_nothing_to_contrast():
    d, arc, _ = seeded_wide(expansion_between=0)
    goal = arc.embeddings[2].copy()
    d.archive = type("A", (), {
        "embeddings": arc.embeddings,
        "centroid": staticmethod(lambda: None),
        "nearest": staticmethod(lambda g: 2),
    })()
    assert d._seed_index(goal, "latent") == 2


def test_an_empty_archive_has_no_seed():
    d, _, _ = make()
    assert d._seed_index(np.eye(DIM, dtype=np.float32)[0], "latent") is None


# ---- set_spec ----------------------------------------------------------

def test_a_scales_only_change_does_not_abandon_the_expedition():
    """set_spec ends the expedition, so comparing spec objects by identity made
    a decode-scale tweak throw away the goal the driver was climbing toward."""
    from services.brains import REGISTRY
    from services.genome_spec import spec_for

    m = REGISTRY["fourier"]
    d, _, _ = make()
    d.set_spec(spec_for(m.layout_from_settings({"freq_scale": 3.0})))
    d._remaining = 5
    scaled = spec_for(m.layout_from_settings({"freq_scale": 1.5}))
    d.set_spec(scaled)
    assert d._remaining == 5
    assert d.spec is scaled


def test_a_width_change_ends_the_expedition():
    from services.brains import REGISTRY
    from services.genome_spec import spec_for

    m = REGISTRY["fourier"]
    d, _, _ = make()
    d.set_spec(spec_for(m.layout_from_settings({"centers": 10})))
    d._remaining = 5
    d.set_spec(spec_for(m.layout_from_settings({"centers": 20})))
    assert d._remaining == 0
