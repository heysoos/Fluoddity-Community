import numpy as np
import pytest

from services.descriptor import descriptor, liveness, stack_snapshots


def _basis(i, dim=4, n=1):
    e = np.zeros((n, dim), dtype=np.float32)
    e[:, i] = 1.0
    return e


def test_stack_snapshots_shape():
    s = stack_snapshots([np.zeros((3, 4), np.float32) for _ in range(5)])
    assert s.shape == (5, 3, 4)
    assert s.dtype == np.float32


def test_descriptor_is_unit_norm():
    rng = np.random.default_rng(0)
    snaps = rng.normal(size=(4, 6, 8)).astype(np.float32)
    snaps /= np.linalg.norm(snaps, axis=-1, keepdims=True)
    b = descriptor(snaps)
    assert b.shape == (6, 8)
    assert np.allclose(np.linalg.norm(b, axis=1), 1.0, atol=1e-5)


def test_descriptor_of_identical_snapshots_is_that_snapshot():
    e = _basis(1, n=3)
    snaps = np.stack([e, e, e, e])
    assert np.allclose(descriptor(snaps), e, atol=1e-6)


def test_descriptor_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"\(S, n, dim\)"):
        descriptor(np.zeros((4, 8), dtype=np.float32))


def test_liveness_is_zero_for_a_frozen_pattern():
    e = _basis(0, n=2)
    snaps = np.stack([e, e, e, e])
    assert np.allclose(liveness(snaps), 0.0, atol=1e-6)


def test_liveness_is_one_for_mutually_orthogonal_snapshots():
    snaps = np.stack([_basis(0), _basis(1), _basis(2), _basis(3)])
    assert np.allclose(liveness(snaps), 1.0, atol=1e-6)


def test_liveness_matches_the_asal_formula_by_hand():
    """ASAL Eq.3: L = 1 - mean_{s>0} max_{s'<s} <e_s, e_s'>.

    e0 == e1, e2 orthogonal to both:
      s=1 -> max(<e1,e0>)         = 1.0
      s=2 -> max(<e2,e0>,<e2,e1>) = 0.0
      mean = 0.5  ->  L = 0.5
    """
    snaps = np.stack([_basis(0), _basis(0), _basis(1)])
    assert liveness(snaps) == pytest.approx([0.5], abs=1e-6)


def test_liveness_uses_historical_max_not_consecutive_pairs():
    """A pattern that returns to an earlier state is not novel just because the
    step before it differed. e2 == e0 must score 0 on its own term."""
    snaps = np.stack([_basis(0), _basis(1), _basis(0)])
    # s=1 -> <e1,e0> = 0 ; s=2 -> max(<e2,e0>=1, <e2,e1>=0) = 1 ; mean 0.5
    assert liveness(snaps) == pytest.approx([0.5], abs=1e-6)


def test_liveness_is_zero_when_there_is_only_one_snapshot():
    """A single snapshot carries no evidence of change. The caller must disable
    the liveness gate at snapshots_per_gen == 1."""
    assert np.array_equal(liveness(_basis(0, n=5)[None]), np.zeros(5, np.float32))


def test_liveness_is_per_tile():
    frozen = np.stack([_basis(0), _basis(0), _basis(0)])[:, 0]        # (3, 4)
    moving = np.stack([_basis(0), _basis(1), _basis(2)])[:, 0]        # (3, 4)
    snaps = np.stack([frozen, moving], axis=1)                        # (3, 2, 4)
    out = liveness(snaps)
    assert out[0] == pytest.approx(0.0, abs=1e-6)
    assert out[1] == pytest.approx(1.0, abs=1e-6)
