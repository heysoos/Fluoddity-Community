"""The plane the Inspector sweeps, including the random projection.

A 4D function drawn on a 2D tile is always a slice, and the four axis-aligned
slices are an arbitrary choice: a brain's units point wherever the search put
them, so structure lying diagonally is invisible in all four at once. A random
2-plane is the standard answer (it is how neural-net loss landscapes are drawn,
Li et al. 2018) and it is what `basis_for(None, seed)` returns.

Everything here is pure numpy - the basis is built on the CPU and handed to the
shader as two vec4 - so it needs no GL context. The GPU side, that the tile
really is x = U*p.x + V*p.y, is in test_brain_preview_gpu.py.
"""
import numpy as np
import pytest

from services.brain_preview import AXES, basis_for

RANDOM_AXES = [i for i, (_, a) in enumerate(AXES) if a is None]
FIXED_AXES = [a for _, a in AXES if a is not None]


def test_the_random_projection_is_offered():
    assert len(RANDOM_AXES) == 1, "exactly one entry should be the random one"
    assert AXES[RANDOM_AXES[0]][0] == "random projection"


@pytest.mark.parametrize("axes", FIXED_AXES)
def test_an_axis_aligned_slice_is_the_standard_basis(axes):
    """The unification must be exact. The shader used to set two components of
    x and leave the rest at zero; now every slice goes through U and V, so the
    fixed ones have to reproduce that identically or all four views changed
    silently when the random option was added."""
    u, v = basis_for(axes)
    x = u * 0.3 + v * -0.7
    want = np.zeros(4, dtype=np.float32)
    want[axes[0]], want[axes[1]] = 0.3, -0.7
    assert np.array_equal(x, want)


@pytest.mark.parametrize("seed", range(12))
def test_a_random_plane_is_orthonormal(seed):
    """Not merely random. Two arbitrary Gaussian directions span a skewed,
    unequally scaled plane, so Input Range would mean a different distance along
    each axis of the picture, and a near-parallel pair would collapse the whole
    tile to almost a line."""
    u, v = basis_for(None, seed)
    assert np.linalg.norm(u) == pytest.approx(1.0, abs=1e-6)
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-6)
    assert float(np.dot(u, v)) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("seed", range(4))
def test_the_same_seed_gives_the_same_plane(seed):
    """The basis is rebuilt EVERY frame. A plane that redrew itself each time
    would strobe rather than show anything, which is the whole reason the seed
    lives in BrainState instead of being drawn here."""
    assert all(np.array_equal(a, b) for a, b in
               zip(basis_for(None, seed), basis_for(None, seed)))


def test_reseeding_actually_moves_the_plane():
    """The Reseed button increments by one, so consecutive seeds - not just
    far-apart ones - have to give visibly different planes."""
    for s in range(8):
        u0, v0 = basis_for(None, s)
        u1, v1 = basis_for(None, s + 1)
        # |<u0,u1>| near 1 would mean the same direction over again.
        assert abs(float(np.dot(u0, u1))) < 0.98, f"seed {s} barely moved"
        assert not np.array_equal(v0, v1)


def test_random_planes_are_not_secretly_axis_aligned():
    """A QR of a Gaussian should almost never land on a coordinate axis. If it
    did, the random option would be showing the same four views."""
    for s in range(32):
        u, _ = basis_for(None, s)
        assert np.abs(u).max() < 0.99, f"seed {s} produced an axis"


def test_the_plane_distribution_is_not_biased_to_one_direction():
    """QR gives a uniform draw over 2-planes, so over many seeds the sweep
    directions should cover the space evenly rather than clustering - a bias
    here would mean the 'random' view is really one more fixed view.
    """
    us = np.stack([basis_for(None, s)[0] for s in range(400)])
    # Sign is arbitrary (a plane has no orientation), so compare the second
    # moment: for an isotropic direction it is the identity over 4.
    m = (us[:, :, None] * us[:, None, :]).mean(0)
    assert np.allclose(m, np.eye(4) / 4.0, atol=0.05), f"anisotropic:\n{m}"


def test_a_seed_far_outside_int32_still_works():
    """preview_seed only ever increments, and nothing clamps it."""
    u, v = basis_for(None, 2 ** 40 + 7)
    assert np.isfinite(u).all() and np.isfinite(v).all()
    assert float(np.dot(u, v)) == pytest.approx(0.0, abs=1e-6)


def test_the_state_carries_a_stable_seed():
    from state import BrainState

    assert BrainState().preview_seed == 0
