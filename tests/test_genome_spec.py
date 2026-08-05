import numpy as np
import pytest

from services.genome_spec import (
    AMP_SCALE,
    BRAIN_SPEC,
    FREQ_SCALE,
    decode,
    encode,
)


def test_decode_shape_and_dtype():
    g = decode(np.zeros(80, dtype=np.float32))
    assert g.shape == (10, 8)
    assert g.dtype == np.float32


def test_decode_is_bounded_for_extreme_z():
    """tanh squash means the optimizer can never wander to freq=50."""
    for fill in (1e6, -1e6):
        g = decode(np.full(80, fill, dtype=np.float32))
        assert np.all(np.abs(g[:, :4]) <= FREQ_SCALE + 1e-5)
        assert np.all(np.abs(g[:, 4:]) <= AMP_SCALE + 1e-5)


def test_encode_decode_roundtrip_for_in_range_values():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 0.7, 80).astype(np.float32)
    g = decode(z)
    z2, clamped = encode(g)
    assert clamped == 0
    assert np.allclose(z, z2, atol=1e-3)


def test_encode_clamps_out_of_range_and_reports_count():
    g = np.zeros((10, 8), dtype=np.float32)
    g[0, 0] = 99.0     # frequency far beyond +3
    g[1, 4] = -5.0     # amplitude far beyond -1
    z, clamped = encode(g)
    assert clamped == 2
    assert np.all(np.isfinite(z)), "must never produce inf in x0"
    assert abs(z[0]) == pytest.approx(np.arctanh(1 - 1e-4), rel=1e-3)


def test_spec_dim_and_signature():
    assert BRAIN_SPEC.dim == 80
    assert BRAIN_SPEC.signature() == "brain:80"


def test_spec_decode_returns_named_parts():
    parts = BRAIN_SPEC.decode(np.zeros(80, dtype=np.float32))
    assert set(parts) == {"brain"}
    assert parts["brain"].shape == (10, 8)


def test_decoded_range_is_comparable_to_random_genome():
    """Generation 0 of Auto mode should look about as varied as generation 0 of
    Manual mode, so the decoded spread must resemble services.genome."""
    from services.genome import random_genome

    rng = np.random.default_rng(0)
    manual = np.stack([random_genome(rng) for _ in range(200)])
    auto = np.stack([decode(rng.normal(0, 0.5, 80)) for _ in range(200)])
    # amplitudes: manual is uniform[-1,1] (std ~0.577)
    assert 0.2 < auto[:, :, 4:].std() < manual[:, :, 4:].std() * 1.5
    # frequencies: same order of magnitude, not 10x off
    assert 0.25 < auto[:, :, :4].std() / manual[:, :, :4].std() < 4.0
