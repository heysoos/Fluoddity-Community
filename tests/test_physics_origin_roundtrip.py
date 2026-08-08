"""The archive stores DECODED phenotypes, not z.

With physics search on, z means 'this far from the preset that happened to be
loaded' (physics_genome: value = origin + span*tanh(z)). A z archived under one
preset and re-run under another is a different creature. These tests pin the
property that makes the archive origin-independent.
"""
import numpy as np
import pytest

from services.genome_spec import BRAIN_PHYSICS_SPEC, BRAIN_SPEC
from services.physics_genome import PHYSICS_PARAMS, decode_physics, encode_physics


def origin_a():
    return {n: (lo + hi) / 2.0 for n, _g, lo, hi in PHYSICS_PARAMS}


def origin_b():
    return {n: lo + 0.25 * (hi - lo) for n, _g, lo, hi in PHYSICS_PARAMS}


def test_the_same_z_means_different_physics_under_different_origins():
    """This is the trap the design avoids. If this test ever fails, storing z
    would have been safe - and it is not."""
    z = np.full(len(PHYSICS_PARAMS), 0.5, dtype=np.float32)
    a = decode_physics(z, origin_a())
    b = decode_physics(z, origin_b())
    assert any(not np.isclose(a[k], b[k]) for k in a)


def test_a_stored_phenotype_reproduces_under_any_origin():
    z = np.full(len(PHYSICS_PARAMS), 0.4, dtype=np.float32)
    phenotype = decode_physics(z, origin_a())          # what the archive stores

    z_under_b = encode_physics(phenotype, origin_b())  # re-encoded on reuse
    got = decode_physics(z_under_b, origin_b())

    for k, v in phenotype.items():
        assert got[k] == pytest.approx(v, abs=1e-3), k


def test_zero_physics_genes_reproduce_the_current_origin_exactly():
    """A brain-only archive entry used in a physics-enabled run gets z=0 for the
    physics block, which must be the preset as loaded - not a slider midpoint."""
    o = origin_b()
    got = decode_physics(np.zeros(len(PHYSICS_PARAMS), np.float32), o)
    for k, v in o.items():
        assert got[k] == pytest.approx(v, abs=1e-6), k


def test_a_brain_spec_entry_is_shorter_than_a_brain_physics_entry():
    """The signature difference is what tells the driver to pad with zeros."""
    assert BRAIN_SPEC.signature() == "brain:80"
    assert BRAIN_PHYSICS_SPEC.signature() == "brain:80,physics:8"
    assert BRAIN_PHYSICS_SPEC.dim == BRAIN_SPEC.dim + len(PHYSICS_PARAMS)
