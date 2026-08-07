import numpy as np

from tools.clip_spread_check import spread_report


def _unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def test_identical_embeddings_have_zero_spread_and_fail():
    e = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (32, 1))
    r = spread_report(e)
    assert r["n"] == 32
    assert r["mean_pairwise"] == 0.0
    assert r["passes"] is False


def test_equidistant_embeddings_fail_despite_maximal_mean_distance():
    """An orthogonal basis is the degenerate 'everything equally far' case.

    Mean pairwise distance is a perfect 1.0, but the variance is zero: no point
    is nearer than any other, so k-NN novelty is constant everywhere and there
    is nothing to climb. This is precisely what the std bar exists to catch, and
    why a mean-only check would wave this through.
    """
    e = np.eye(8, dtype=np.float32)
    r = spread_report(e)
    assert r["mean_pairwise"] == 1.0
    assert r["std_pairwise"] == 0.0
    assert r["passes"] is False


def test_a_structured_spread_passes():
    """Two well-separated clusters: some neighbours near, some far. This is the
    shape novelty search actually needs."""
    rng = np.random.default_rng(7)
    a = _unit(np.array([1.0, 0, 0, 0], np.float32) + rng.normal(0, 0.3, (40, 4)))
    b = _unit(np.array([0, 1.0, 0, 0], np.float32) + rng.normal(0, 0.3, (40, 4)))
    r = spread_report(np.concatenate([a, b]).astype(np.float32))
    assert r["mean_pairwise"] > 0.15
    assert r["std_pairwise"] > 0.05
    assert r["passes"] is True


def test_report_uses_only_the_upper_triangle():
    """Including the zero self-distances would halve every mean."""
    e = _unit(np.random.default_rng(0).normal(size=(64, 16)).astype(np.float32))
    r = spread_report(e)
    sim = e @ e.T
    iu = np.triu_indices(64, k=1)
    assert r["mean_pairwise"] == float(np.mean(1.0 - sim[iu]))


def test_pass_bar_is_mean_over_015_and_std_over_005():
    rng = np.random.default_rng(1)
    e = _unit(rng.normal(size=(128, 32)).astype(np.float32))
    r = spread_report(e)
    assert r["passes"] == (r["mean_pairwise"] > 0.15 and r["std_pairwise"] > 0.05)


def test_a_duplicate_dominated_sample_is_flagged():
    """One converged run's frames can be 57% of the pairs and drag the mean
    below the bar while the across-run spread is fine. Measured 2026-08-07."""
    rng = np.random.default_rng(11)
    clump = _unit(np.array([1.0, 0, 0, 0], np.float32) + rng.normal(0, 0.02, (60, 4)))
    outliers = _unit(rng.normal(size=(6, 4)).astype(np.float32))
    r = spread_report(np.concatenate([clump, outliers]).astype(np.float32))
    assert r["p50"] < 0.5 * r["mean_pairwise"]
    assert r["duplicate_dominated"] is True


def test_a_healthy_sample_is_not_flagged_as_duplicate_dominated():
    rng = np.random.default_rng(12)
    e = _unit(rng.normal(size=(80, 16)).astype(np.float32))
    assert spread_report(e)["duplicate_dominated"] is False


def test_fewer_than_two_embeddings_reports_zero_and_fails():
    r = spread_report(np.zeros((1, 4), dtype=np.float32))
    assert r["n"] == 1
    assert r["mean_pairwise"] == 0.0
    assert r["passes"] is False
