"""The two calibration criteria, on synthetic populations.

Each encoder's value is chosen so it reproduces on the same population what
clip-b32's value does - equating on BEHAVIOUR rather than on a summary
statistic. See CLAUDE.md.
"""
import numpy as np

from tools.calibrate_encoder import (floored_fraction, latent_goal, retention,
                                     scale_matching, threshold_matching)


def _cone(rng, n=64, dim=64, spread=0.5):
    """A cone whose goal-versus-centroid gap is the size a real archive's is.

    The default spread matters: at 0.15 the gap is ~0.09 and nothing floors
    until scale 300, so a fixture that tight tests nothing about the range the
    encoders actually live in.
    """
    base = rng.normal(size=dim).astype(np.float32)
    e = base + spread * rng.normal(size=(n, dim)).astype(np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def test_a_sharper_scale_floors_more_of_the_population():
    """A floored tile carries no information to a rank-based optimizer."""
    rng = np.random.default_rng(3)
    e = _cone(rng)
    goal, refs = latent_goal(e)
    assert floored_fraction(e, goal, refs, 100.0) > \
           floored_fraction(e, goal, refs, 30.0)


def test_flooring_is_monotone_in_the_scale():
    """What makes the bisection in scale_matching valid."""
    rng = np.random.default_rng(4)
    e = _cone(rng)
    goal, refs = latent_goal(e)
    got = [floored_fraction(e, goal, refs, s) for s in (10, 30, 100, 300)]
    assert got == sorted(got)


def test_scale_matching_hits_the_target_fraction():
    rng = np.random.default_rng(5)
    e = _cone(rng)
    goal, refs = latent_goal(e)
    s = scale_matching(e, goal, refs, target_frac=0.10)
    assert abs(floored_fraction(e, goal, refs, s) - 0.10) < 0.03


def test_the_latent_goal_is_the_plus_three_sd_construction():
    """The construction is the one recorded in test_expedition_fitness."""
    rng = np.random.default_rng(9)
    e = _cone(rng)
    goal, refs = latent_goal(e)
    assert goal.shape == (e.shape[1],)
    assert refs.shape == (1, e.shape[1])
    assert np.allclose(np.linalg.norm(goal), 1.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(refs[0]), 1.0, atol=1e-5)
    # It leans towards entry 0 and away from the centroid.
    assert float(e[0] @ goal) > float(refs[0] @ goal)


def test_retention_falls_as_the_threshold_rises():
    nn = np.linspace(0.0, 0.1, 200).astype(np.float32)
    assert retention(nn, 0.01) > retention(nn, 0.05)


def test_retention_of_a_zero_bar_admits_everything():
    nn = np.linspace(0.0, 0.1, 50).astype(np.float32)
    assert retention(nn, 0.0) == 1.0


def test_threshold_matching_recovers_a_known_threshold():
    nn = np.linspace(0.0, 0.1, 500).astype(np.float32)
    target = retention(nn, 0.02)
    assert abs(threshold_matching(nn, target) - 0.02) < 0.002


def test_a_wider_spread_population_needs_a_larger_threshold():
    """The whole reason min_separation is per-encoder."""
    narrow = np.linspace(0.0, 0.05, 400).astype(np.float32)
    wide = np.linspace(0.0, 0.11, 400).astype(np.float32)
    target = retention(narrow, 0.02)
    assert threshold_matching(wide, target) > threshold_matching(narrow, target)
