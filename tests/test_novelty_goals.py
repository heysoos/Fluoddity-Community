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


def lattice(period, n=96):
    """A fine, perfectly regular pattern - complex but not remotely noise."""
    y, x = np.mgrid[0:n, 0:n]
    v = 127 + 120 * np.sin(2 * np.pi * x / period) * np.sin(2 * np.pi * y / period)
    return np.repeat(np.clip(v, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)


@pytest.mark.parametrize("period", [3, 4, 6, 12])
def test_a_fine_regular_pattern_is_not_mistaken_for_noise(period):
    """The reason structure() looks at more than the neighbouring pixel.

    A lattice with a 3-pixel period decorrelates in one pixel exactly like
    static does - measured, lag 1 alone scored a 3px and a 4px lattice at
    0.000, the same as white noise, which would have annihilated the fitness
    of a perfectly good fine-grained creature. It re-correlates at its own
    period; noise never re-correlates at all.
    """
    assert float(structure(lattice(period)[None])[0]) > 0.9


def test_fine_detail_riding_on_a_large_envelope_survives():
    y, x = np.mgrid[0:96, 0:96]
    v = 127 + 60 * np.sin(2 * np.pi * x / 3) + 60 * np.sin(2 * np.pi * y / 40)
    img = np.repeat(np.clip(v, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
    assert float(structure(img[None])[0]) > 0.9


def test_slow_motion_is_not_penalised_at_all():
    """structure() is purely SPATIAL - it never compares frames. How fast a
    pattern moves is liveness, a different quantity, and the two must not be
    conflated: a slow complex creature is exactly what must survive."""
    img = lattice(6)
    still = np.stack([img, img])                  # identical frames
    moved = np.stack([img, np.roll(img, 17, axis=1)])
    assert float(structure(still)[0]) == pytest.approx(
        float(structure(moved)[0]), abs=1e-6)


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


# ---- a seed has to be a genome THIS brain can decode ----------------------
#
# An archive pools every layout, so most of what it holds may be another
# brain's. Expansion and _seed_index both filter through native_rows(); a
# novelty goal carries its seed instead, and that was the one path that did
# not. It crashed a 2.5-hour run inside Fourier's encode() with "cannot
# reshape array of size 71 into shape (10,8)" - 71 being an mlp-n3.4.4 genome.

def _mixed(n_native, n_foreign):
    """A driver on FOURIER whose archive also holds wider, foreign entries."""
    from services.archive import Candidate
    from services.brains import BrainLayout

    d, arc, _ = make(seed_n=1)
    other = BrainLayout("gabor", (12,), 168)
    # As a load does: the pooled column widens to the widest layout present,
    # and brain_at() trims each row back to its own.
    arc._widen(other.length)
    arc._widths[other.signature()] = other.length
    # A store-backed archive registers its own width at construction and a load
    # setdefaults every sibling's; this fake has no store, so brain_at() would
    # hand back the widened row for a NATIVE entry too.
    arc._widths[arc.signature] = arc.layout.length

    def add(k, layout, foreign):
        v = np.zeros(DIM, np.float32)
        v[k % DIM] = 1.0
        v[(k + 1) % DIM] = 0.5 if foreign else -0.5
        cand = Candidate(
            embedding=(v / np.linalg.norm(v)).astype(np.float32),
            brain=np.arange(layout.length, dtype=np.float32),
            physics=np.zeros(8, np.float32), liveness=0.5, spec="brain",
            viable=True)
        e = arc.consider(cand, novelty=1.0, force=True)
        if e is not None and foreign:
            e.layout = layout.signature()

    for j in range(n_native):
        add(j, arc.layout, False)
    for j in range(n_foreign):
        add(j, other, True)
    return d, arc


def test_a_novelty_goal_seeds_on_a_row_this_brain_can_decode():
    """The root cause. sample_by_novelty ran over every entry, so in a mixed
    archive it returned another brain's row and the expedition encoded that
    genome under the running layout."""
    d, arc = _mixed(n_native=2, n_foreign=8)
    native = set(int(i) for i in arc.native_rows())
    assert native and len(native) < len(arc), "precondition: the archive is mixed"

    rng = np.random.default_rng(0)
    for _ in range(60):
        g = novelty_goal(arc, rng)
        assert g is not None
        assert int(g.seed_index) in native, (
            f"row {g.seed_index} belongs to {arc.layout_at(int(g.seed_index))}")


def test_a_novelty_goal_declines_when_no_row_is_decodable():
    """Every entry belongs to another brain. Nothing to seed from, so the
    expedition must not start rather than encode a foreign genome."""
    d, arc = _mixed(n_native=0, n_foreign=6)
    for e in arc.entries:
        e.layout = "gabor-n12"
    assert not len(arc.native_rows()), "precondition: nothing native"
    assert novelty_goal(arc, np.random.default_rng(0)) is None


def test_a_foreign_seed_index_is_refused_rather_than_encoded():
    """The backstop. Whatever a goal claims, the seed becomes the optimizer's
    mean and gets re-encoded under the running layout - so a row this brain
    cannot decode has to be turned away at the boundary, not deep inside a
    modality's reshape."""
    d, arc = _mixed(n_native=2, n_foreign=6)
    foreign = [i for i in range(len(arc)) if not arc.is_native(i)]
    assert foreign, "precondition: there is a foreign row"

    # No embedding, as a novelty goal: there is no _seed_index to fall back on.
    assert d.start_expedition_with(None, "novelty", "",
                                   seed_index=foreign[0]) is False
    # With an embedding it falls through to _seed_index, which filters.
    ok = d.start_expedition_with(arc.embeddings[0].copy(), "latent", "",
                                 seed_index=foreign[0])
    assert not ok or arc.is_native(int(d._x0_index))


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
