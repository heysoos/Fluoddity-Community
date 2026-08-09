import numpy as np
import pytest

from services.archive_projection import Projection


def _blob(n, dim, rng, scale):
    return (rng.normal(size=(n, dim)) * scale).astype(np.float32)


def _structured(n, dim, rng):
    """A cloud with two dominant axes, which is the regime real CLIP embeddings
    are in - measured 2026-08-07, presets spread at mean pairwise 0.16 with std
    0.056, so there is genuine structure rather than isotropic noise."""
    x = rng.normal(size=(n, dim)) * 0.3
    x[:, 0] *= 8.0
    x[:, 1] *= 4.0
    return x.astype(np.float32)


def test_transform_before_fit_returns_zeros():
    p = Projection()
    out = p.transform(np.zeros((5, 8), np.float32))
    assert out.shape == (5, 2)
    assert np.all(out == 0.0)
    assert p.fitted is False


def test_fit_needs_more_rows_than_components():
    p = Projection(n_components=2)
    assert p.fit(np.zeros((1, 8), np.float32)) is False
    assert p.fitted is False


def test_components_are_orthonormal():
    rng = np.random.default_rng(0)
    p = Projection()
    p.fit(_blob(200, 16, rng, 1.0))
    c = p.components
    assert c.shape == (2, 16)
    assert np.allclose(c @ c.T, np.eye(2), atol=1e-4)


def test_the_first_component_follows_the_direction_of_greatest_variance():
    rng = np.random.default_rng(1)
    x = _blob(300, 4, rng, 0.01)
    x[:, 2] += rng.normal(size=300) * 5.0          # variance dominates axis 2
    p = Projection()
    p.fit(x.astype(np.float32))
    assert abs(p.components[0, 2]) > 0.9


def test_transform_is_centred_and_deterministic():
    rng = np.random.default_rng(2)
    x = _blob(100, 8, rng, 1.0)
    p = Projection()
    p.fit(x)
    a = p.transform(x)
    b = p.transform(x)
    assert a.shape == (100, 2)
    assert np.array_equal(a, b)
    assert np.allclose(a.mean(axis=0), 0.0, atol=1e-4)


def test_a_refit_sign_aligns_to_the_previous_components():
    """Eigenvectors have arbitrary sign. Without alignment the map mirrors
    itself at every refit and reads as a bug."""
    rng = np.random.default_rng(3)
    x = _structured(200, 8, rng)
    p = Projection()
    p.fit(x)
    first = p.components.copy()
    before = p.transform(x[:20])

    p.fit(np.concatenate([x, _structured(20, 8, rng)]))
    assert np.all(np.sum(first * p.components, axis=1) > 0), "signs must agree"
    after = p.transform(x[:20])
    # a small data change must not flip the layout
    assert np.mean(np.sign(before) == np.sign(after)) > 0.8


def test_near_degenerate_axes_rotate_between_refits():
    """The honest limit of sign alignment, recorded rather than hidden.

    Sign alignment fixes the SIGN of an axis, not its direction. When the top
    eigenvalues are nearly equal the axes are free to rotate within their shared
    subspace, and a refit genuinely moves them - no amount of alignment helps.
    Measured on isotropic 8-d noise: lambda1/lambda2 = 1.12, and adding 10% more
    points swings component 2 to cos 0.27 (about 74 degrees) while every sign
    still agrees.

    This is why the map is a VIEW and never feeds the search, and why "Refit
    projection" is a button rather than something that fires continuously.
    """
    rng = np.random.default_rng(3)
    x = _blob(200, 8, rng, 1.0)
    p = Projection()
    p.fit(x)
    first = p.components.copy()
    p.fit(np.concatenate([x, _blob(20, 8, rng, 1.0)]))

    align = np.sum(first * p.components, axis=1)
    assert np.all(align > 0), "sign alignment still holds"
    assert align[1] < 0.5, "a near-degenerate second axis really does rotate"


def test_a_deliberately_flipped_refit_is_corrected():
    rng = np.random.default_rng(4)
    x = _blob(200, 6, rng, 1.0)
    p = Projection()
    p.fit(x)
    p._prev = -p.components.copy()       # pretend the last fit was the mirror
    p.fit(x)
    assert np.all(np.sum(p._prev * p.components, axis=1) > 0)


def test_fit_transform_matches_fit_then_transform():
    rng = np.random.default_rng(5)
    x = _blob(60, 8, rng, 1.0)
    a = Projection().fit_transform(x)
    p = Projection()
    p.fit(x)
    assert np.allclose(a, p.transform(x), atol=1e-6)


def test_degenerate_input_does_not_raise():
    """Every embedding identical - zero variance in every direction."""
    x = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], np.float32), (50, 1))
    p = Projection()
    p.fit(x)
    out = p.transform(x)
    assert out.shape == (50, 2)
    assert np.all(np.isfinite(out))


# ---- variances: the search whitens with these ---------------------------

def test_variances_are_the_retained_eigenvalues_descending():
    """latent_goal divides by their square root, so an extrapolation distance
    means the same thing along a dominant axis and a minor one."""
    rng = np.random.default_rng(11)
    x = _structured(300, 10, rng)
    p = Projection(n_components=4)
    p.fit(x)
    assert p.variances.shape == (4,)
    assert np.all(np.diff(p.variances) <= 0.0), "descending"
    # each equals the variance of the data projected on that component
    proj = (x - p.mean) @ p.components.T
    assert np.allclose(p.variances, proj.var(axis=0, ddof=1), rtol=0.02)


def test_variances_are_never_negative():
    """eigh returns a tiny negative for a near-zero eigenvalue, and a negative
    variance becomes a NaN standard deviation the moment it is whitened with."""
    x = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], np.float32), (50, 1))
    p = Projection(n_components=3)
    p.fit(x)
    assert np.all(p.variances >= 0.0)
    assert np.all(np.isfinite(np.sqrt(np.maximum(p.variances, 1e-12))))


def test_variances_are_absent_until_a_fit_succeeds():
    p = Projection(n_components=2)
    assert p.variances is None
    assert p.fit(np.zeros((2, 8), np.float32)) is False
    assert p.variances is None


def test_the_dominant_component_carries_the_most_variance():
    rng = np.random.default_rng(12)
    p = Projection(n_components=2)
    p.fit(_structured(400, 10, rng))
    assert p.variances[0] > 2.0 * p.variances[1]


def test_fitting_moves_the_version():
    """The map caches transform()'s output against this. A refit that leaves
    it alone would keep drawing the old components' layout."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 16)).astype(np.float32)
    p = Projection()
    assert p.version == 0
    assert p.fit(x) is True
    first = p.version
    assert first > 0
    assert p.fit(x) is True
    assert p.version > first


def test_a_failed_fit_leaves_the_version_alone():
    p = Projection()
    assert p.fit(np.zeros((2, 16), dtype=np.float32)) is False
    assert p.version == 0
