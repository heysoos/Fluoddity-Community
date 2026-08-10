"""Three kinds of expedition goal, and the noise defence the latent one lacked.

Measured 2026-08-10 on two real archives: the entry a latent goal ranks FIRST
is in the archive's roughest decile 34.0% and 32.5% of the time, against the
10% a blind draw gives. The mechanism is that a latent or chase goal contrasts
against ONE reference, the archive centroid, which makes contrastive() a
monotone squash of raw cosine - and raw cosine to an arbitrary direction is
maximised by high-frequency noise. Text goals never had the problem because
DEFAULT_DISTRACTORS carries "random noise" and "an abstract texture".

Two answers, because the flaw had two halves that shared _references():
  the FITNESS  multiplied by capture_health.structure
  the SEED     latent goals now start at the anchor they were extrapolated from
and a third kind of goal that has no target to be unreachable at all.
"""
from __future__ import annotations

import numpy as np
import pytest

from services.archive_projection import Projection
from services.capture_health import structure
from services.goal_source import LATENT_DIMS, latent_goal, novelty_goal
from tests.test_imgep_driver import DIM, make, moving, snaps


# ---- the structure statistic --------------------------------------------

def test_white_noise_scores_zero_and_smooth_scores_one():
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 256, (1, 64, 64, 3), dtype=np.uint8)
    ramp = np.tile(np.linspace(0, 255, 64, dtype=np.uint8)[None, :, None],
                   (64, 1, 3))[None]
    s = structure(np.concatenate([noise, ramp]))
    assert s[0] < 0.05, "white noise should carry no spatial coherence"
    assert s[1] > 0.95, "a smooth ramp is all coherence"


def test_a_flat_tile_scores_one():
    """It has no high frequencies at all. Correct here, and not a hole:
    is_viable_tile already rejects a blank capture."""
    assert structure(np.full((1, 32, 32, 3), 40, dtype=np.uint8))[0] == 1.0


def test_structure_is_per_tile():
    rng = np.random.default_rng(1)
    a = np.zeros((3, 32, 32, 3), dtype=np.uint8)
    a[0] = 40
    a[1] = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    a[2] = np.tile(np.linspace(0, 255, 32, dtype=np.uint8)[:, None, None], (1, 32, 3))
    s = structure(a)
    assert s.shape == (3,)
    assert s[1] < s[2]


def test_an_empty_batch_is_not_an_error():
    assert structure(np.zeros((0, 8, 8, 3), dtype=np.uint8)).shape == (0,)


# ---- the seed ------------------------------------------------------------

def seeded(n=32, dim=DIM):
    """An archive big enough to FIT the projection - latent_goal declines
    otherwise, and 8 components need more than 8 samples."""
    from services.archive import Archive, Candidate
    from services.physics_genome import PHYSICS_DIM

    rng = np.random.default_rng(0)
    a = Archive(store=None, dim=dim, capacity=100, liveness_min=0.0,
                min_separation=0.0)
    for i in range(n):
        e = rng.normal(size=dim).astype(np.float32)
        e /= np.linalg.norm(e)
        a.consider(Candidate(brain=np.zeros((10, 8), np.float32),
                             physics=np.zeros(PHYSICS_DIM, np.float32), embedding=e,
                             liveness=0.5, viable=True, spec="brain:80",
                             gen=0, tile=i, run_id="", goal=""),
                   novelty=0.1 * (i + 1))
    return a


def test_a_latent_goal_does_not_pick_its_own_seed():
    """It looks like it should - the anchor is the archive's nearest point to
    the goal by construction. Measured on the fraction of seeds landing in the
    archive's roughest decile it is a regression (default 15.8% -> 27.0%),
    because _seed_index samples with banded_alpha rather than argmaxing while
    p ~ NOV^4 concentrates hard. The 34% that motivated it is the fitness
    ARGMAX, which the coherence factor answers instead."""
    a = seeded()
    g = latent_goal(a, np.random.default_rng(0), Projection(LATENT_DIMS))
    assert g is not None
    assert g.seed_index is None


def test_the_expedition_starts_at_the_goals_own_seed():
    d, arc, _ = make(seed_n=1)
    for i in range(3):
        d.tell(d.ask(4), moving(4, base=30 * i + 10))
    want = len(arc) - 1
    assert d.start_expedition_with(arc.embeddings[0].copy(), "latent", "",
                                   seed_index=want) is True
    assert d._x0_index == want


def test_an_out_of_range_seed_falls_back_rather_than_crashing():
    d, arc, _ = make(seed_n=1)
    d.tell(d.ask(4), moving(4))
    assert d.start_expedition_with(arc.embeddings[0].copy(), "latent", "",
                                   seed_index=9999) is True
    assert 0 <= d._x0_index < len(arc)


# ---- novelty goals -------------------------------------------------------

def test_a_novelty_goal_has_no_target_but_knows_where_to_start():
    a = seeded()
    g = novelty_goal(a, np.random.default_rng(0))
    assert g is not None and g.kind == "novelty"
    assert g.embedding is None, "there is nothing to point at"
    assert 0 <= g.seed_index < len(a)


def test_a_novelty_goal_needs_an_archive():
    from services.archive import Archive

    assert novelty_goal(Archive(store=None, dim=DIM), np.random.default_rng(0)) is None


def test_a_novelty_expedition_runs_and_scores_on_novelty():
    d, arc, _ = make(seed_n=1, expansion_between=0, expedition_gens=5)
    for i in range(3):
        d.tell(d.ask(4), moving(4, base=30 * i + 10))
    assert d.start_expedition_with(None, "novelty", "", seed_index=0) is True
    assert d.regime == "expedition"
    fit = d.tell(d.ask(4), moving(4, base=200))
    assert len(fit) == 4 and np.all(np.isfinite(fit))
    assert d.status()["score_label"] == "goal match"


def test_a_novelty_expedition_without_a_seed_cannot_start():
    """No embedding and no seed leaves nothing to point at and nowhere to
    begin, which must be a refusal rather than a crash."""
    d, _, _ = make(seed_n=1)
    assert d.start_expedition_with(None, "novelty", "") is False


def test_the_novelty_fitness_is_the_archive_novelty():
    """Asserted as an identity rather than as a spread: over a small one-hot
    archive every tile can legitimately share a kNN distance, so 'the numbers
    differ' would be a test of the fake, not of the driver."""
    d, _, _ = make(seed_n=1, expansion_between=0, expedition_gens=5)
    for i in range(3):
        d.tell(d.ask(4), moving(4, base=30 * i + 10))
    d.start_expedition_with(None, "novelty", "", seed_index=0)
    want = np.array([0.1, 0.9, 0.4, 0.2], dtype=np.float32)
    d.archive.novelty_of = lambda b: want
    # Flat crops have zero variance, so structure() is exactly 1.0 and the
    # coherence factor is a no-op here.
    fit = d.tell(d.ask(4), moving(4, base=200))
    assert np.allclose(fit, want)


# ---- the noise defence on the fitness ------------------------------------

def noisy_batch(n, rng, noisy_tile=0):
    """One generation where a single tile is static and the rest are smooth."""
    out = []
    for _ in range(2):
        a = np.zeros((n, 224, 224, 3), dtype=np.uint8)
        for i in range(n):
            if i == noisy_tile:
                a[i] = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
            else:
                a[i] = 60 + 10 * i
        out.append(a)
    return out


def test_a_static_tile_is_demoted_by_the_fitness():
    d, arc, _ = make(seed_n=1, expansion_between=0, expedition_gens=5)
    for i in range(3):
        d.tell(d.ask(4), moving(4, base=30 * i + 10))
    d.start_expedition_with(arc.embeddings[0].copy(), "latent", "", seed_index=0)
    rng = np.random.default_rng(3)
    fit = d.tell(d.ask(4), noisy_batch(4, rng, noisy_tile=0))
    assert fit[0] == pytest.approx(0.0, abs=1e-3), \
        "static should be multiplied to nothing whatever it matched"
    assert float(np.max(fit[1:])) > 0.0


# ---- the three-way draw --------------------------------------------------

def kinds(d, n=60):
    out = []
    for _ in range(n):
        g = d._draw_goal()
        out.append(g.kind if g is not None else None)
    return out


def test_all_novelty_share_gives_only_novelty_goals():
    d, _, _ = make(seed_n=1)
    d.tell(d.ask(4), moving(4))
    d.novelty_share, d.latent_share = 1.0, 0.0
    assert set(kinds(d)) == {"novelty"}


def test_all_latent_share_gives_only_latent_goals():
    d, _, _ = make(seed_n=1)
    for i in range(3):
        d.tell(d.ask(4), moving(4, base=30 * i + 10))
    d.novelty_share, d.latent_share = 0.0, 1.0
    assert set(kinds(d)) == {"latent"}


def test_text_takes_the_remainder():
    from services.goal_source import GoalList

    d, _, _ = make(seed_n=1)
    d.tell(d.ask(4), moving(4))
    g = GoalList()
    g.add("coral reef")
    d.goals = g
    d.novelty_share, d.latent_share = 0.0, 0.0
    assert set(kinds(d)) == {"text"}


def test_the_two_shares_cannot_add_up_to_more_than_everything():
    """Both sliders reach 1.0 independently. Latent is clamped to what novelty
    leaves rather than either being silently rescaled."""
    d, _, _ = make(seed_n=1)
    for i in range(3):
        d.tell(d.ask(4), moving(4, base=30 * i + 10))
    d.novelty_share, d.latent_share = 1.0, 1.0
    assert set(kinds(d)) == {"novelty"}


def test_a_mixed_split_produces_both():
    d, _, _ = make(seed_n=1)
    for i in range(3):
        d.tell(d.ask(4), moving(4, base=30 * i + 10))
    d.novelty_share, d.latent_share = 0.5, 0.5
    seen = set(kinds(d, n=120))
    assert seen == {"novelty", "latent"}


def test_a_kind_that_declines_falls_through_instead_of_wasting_the_interval():
    """An expedition that fails to start costs a whole cadence interval. With
    no text goals and an unfittable projection, a text draw must still yield
    the novelty goal rather than None."""
    d, _, _ = make(seed_n=1)
    d.tell(d.ask(4), moving(4))
    d.goals = None
    d.projection = None                  # latent declines
    d.novelty_share, d.latent_share = 0.0, 0.0
    assert set(kinds(d)) == {"novelty"}
