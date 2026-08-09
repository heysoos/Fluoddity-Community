"""The seed pool is a BAND on effective sample size, not a target.

How concentrated a goal's matches are is real information about the archive.
Measured 2026-08-08 over 20 varied prompts against 4784 real descriptors, ESS
at alpha=4 ran 3.1 to 1973, and the ordering is semantic: the archive genuinely
holds almost nothing like "a photograph of a cat" and a great deal that could
pass for "circuit board traces". A target would tell the cat prompt it has 64
good seeds when it has three.
"""
import numpy as np
import pytest

from services.novelty import (
    alpha_for_ess,
    banded_alpha,
    effective_sample_size,
)


def scores(n=4000, decay=0.995, seed=0):
    """A plausible fitness vector: many poor entries, a few good ones."""
    rng = np.random.default_rng(seed)
    w = decay ** np.arange(n)
    return (w * rng.uniform(0.8, 1.2, size=n)).astype(np.float64)


# ---- the ESS statistic --------------------------------------------------

def test_uniform_weights_have_ess_equal_to_the_population():
    assert effective_sample_size(np.ones(500), 1.0) == pytest.approx(500.0)


def test_alpha_zero_is_uniform_whatever_the_weights():
    assert effective_sample_size(scores(300), 0.0) == pytest.approx(300.0)


def test_a_single_dominant_entry_drives_ess_to_one():
    w = np.zeros(200); w[7] = 1.0
    assert effective_sample_size(w, 1.0) == pytest.approx(1.0)


def test_all_zero_weights_are_uniform_by_convention():
    """Matching sample_by_novelty's own fallback - a goal every tile floors on
    must not divide by zero."""
    assert effective_sample_size(np.zeros(64), 4.0) == pytest.approx(64.0)


def test_ess_is_monotone_non_increasing_in_alpha():
    """The property bisection depends on. log ESS(a) = 2S(a) - S(2a) with S a
    log-sum-exp, which is convex, so the derivative is 2[S'(a) - S'(2a)] <= 0."""
    w = scores()
    vals = [effective_sample_size(w, a) for a in np.linspace(0.0, 32.0, 80)]
    assert all(b <= a + 1e-6 for a, b in zip(vals, vals[1:]))


# ---- solving ------------------------------------------------------------

def test_bisection_hits_the_requested_ess():
    w = scores()
    for target in (8.0, 64.0, 512.0):
        a = alpha_for_ess(w, target)
        assert effective_sample_size(w, a) == pytest.approx(target, rel=0.02)


def test_an_unreachable_target_returns_the_bracket_end_rather_than_diverging():
    w = np.ones(100)                       # ESS is 100 at every alpha
    assert alpha_for_ess(w, 10.0, lo=0.0, hi=8.0) == pytest.approx(8.0)
    assert alpha_for_ess(w, 500.0, lo=0.0, hi=8.0) == pytest.approx(0.0)


def test_a_large_alpha_does_not_overflow():
    """w**alpha leaves float64 well inside the bracket bisection explores -
    21**256 is already 1e338 - and an inf makes the ESS ratio NaN."""
    w = 1.0 + 20.0 * np.linspace(0.0, 1.0, 2000)
    for a in (64.0, 256.0, 512.0):
        e = effective_sample_size(w, a)
        assert np.isfinite(e) and 1.0 <= e <= len(w)


# ---- the band -----------------------------------------------------------

def test_an_alpha_already_inside_the_band_is_left_alone():
    """THE point. The spread of ESS across goals is signal, so the band must
    not touch anything that is not degenerate."""
    w = scores()
    a = 4.0
    e = effective_sample_size(w, a)
    assert 8.0 < e < 512.0, "fixture must start inside the band"
    assert banded_alpha(w, a, 8.0, 512.0) == pytest.approx(a)


def test_a_goal_the_archive_barely_matches_keeps_its_concentration():
    """"a photograph of a cat" scored ESS 3.1 on the real archive - the archive
    really does hold almost nothing cat-like. Three good seeds is the honest
    answer, and a floor of 8 must not invent five more."""
    n = 4000
    w = np.zeros(n); w[:3] = [1.0, 0.9, 0.8]
    before = effective_sample_size(w, 4.0)
    after = effective_sample_size(w, banded_alpha(w, 4.0, 8.0, 512.0))
    assert before < 8.0
    # the floor may lift it a little, but it must not reach a fabricated 8
    assert after < 8.0, f"floor invented candidates that do not exist: {after}"


def test_a_goal_almost_everything_matches_is_pulled_back():
    """"circuit board traces" scored ESS 1973 of 4808 - 41% of the archive
    effectively in play, which means the goal is barely influencing the seed."""
    w = 1.0 + 0.001 * np.arange(4000)[::-1]
    assert effective_sample_size(w, 4.0) > 1500.0, "over a third of the archive"
    a = banded_alpha(w, 4.0, 8.0, 512.0)
    assert effective_sample_size(w, a) == pytest.approx(512.0, rel=0.05)
    assert a > 4.0, "flattening cannot reduce ESS; it has to sharpen"


def test_the_floor_never_demands_more_candidates_than_an_eighth_of_the_archive():
    """A floor is a demand for entries that may not exist. Asking for 8 of a
    12-entry archive would force two thirds of it into the pool."""
    w = np.zeros(12); w[0] = 1.0; w[1] = 0.5
    a = banded_alpha(w, 4.0, 8.0, 512.0)
    assert effective_sample_size(w, a) <= 12 / 8.0 + 1e-6


def test_the_band_holds_ess_steady_as_the_archive_grows():
    """Without it, ESS/N is roughly constant per goal, so a diffuse goal drifts
    from 92 at 300 entries to 1602 at 4808 and heads for ~6600 at capacity -
    seeding converges on ignoring the goal entirely."""
    fixed, banded = [], []
    for n in (300, 1200, 4800, 20000):
        w = 1.0 + 0.001 * np.arange(n)[::-1]
        fixed.append(effective_sample_size(w, 4.0))
        banded.append(effective_sample_size(w, banded_alpha(w, 4.0, 8.0, 512.0)))
    assert fixed[-1] / fixed[0] > 20.0, "the drift this exists to stop is real"
    assert max(banded) <= 512.0 * 1.05
