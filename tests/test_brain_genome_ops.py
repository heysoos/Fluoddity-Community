"""CPU mutation and crossover, per modality.

A brain's floats are not interchangeable scalars. Some are WIDTHS and
DIRECTIONS, which must be scaled - offsetting a width walks it through zero, and
the envelope divides by its square. Others are amplitudes, biases and locations,
which must be offset - scaling a band centre near zero pins it at zero forever.

Every modality already declares that split on the GPU, in its own
`<modality>_param_at`. Nothing declared it on the host, so the interactive
tournament used Fourier's split (`services.genome.mutate`) for every modality -
and since it also generated Fourier-shaped genomes, that was moot in the worst
possible way.

`scale_mask` is the host-side copy of that split. It IS a second copy of a
mapping, which this codebase has been bitten by; the guard is
tests/test_brain_mutation_gpu.py::test_the_host_scale_mask_matches_the_shader,
which runs both and compares.
"""
import numpy as np
import pytest

from services.brains import REGISTRY, crossover, get, mutate

MODALITIES = sorted(REGISTRY)

# Which floats of ONE unit are SCALES, read off each modality's *_param_at.
#   fourier.glsl  frequency(4), amplitude(4)          -> k < 4
#   gabor.glsl    centre(4), freq(4), amp(4), sg, ph  -> 4 <= k < 8 or k == 12
#   lenia.glsl    projection(4), amp(4), mu, sigma    -> k < 4 or k == 9
#   mlp.glsl      every weight and bias is a scalar   -> none
EXPECTED_SCALES = {
    "fourier": (8, {0, 1, 2, 3}),
    "gabor": (14, {4, 5, 6, 7, 12}),
    "lenia": (10, {0, 1, 2, 3, 9}),
    "mlp": (None, set()),
}


def layout(name, **s):
    return REGISTRY[name].layout_from_settings(s)


@pytest.mark.parametrize("name", MODALITIES)
def test_the_scale_mask_covers_the_whole_genome(name):
    lay = layout(name)
    assert get(name).scale_mask(lay).shape == (lay.length,)


@pytest.mark.parametrize("name", MODALITIES)
def test_the_scale_mask_marks_the_floats_the_shader_scales(name):
    lay = layout(name)
    stride, scales = EXPECTED_SCALES[name]
    mask = get(name).scale_mask(lay)
    if stride is None:
        assert not mask.any(), "mlp has no widths or directions"
        return
    for i in range(lay.length):
        assert bool(mask[i]) is (i % stride in scales), f"{name} float {i}"


@pytest.mark.parametrize("name", MODALITIES)
def test_the_unit_stride_matches_the_layout(name):
    lay = layout(name)
    stride, _ = EXPECTED_SCALES[name]
    assert get(name).unit_floats(lay) == stride
    if stride is not None:
        assert lay.length == stride * lay.shape[0]


@pytest.mark.parametrize("name", MODALITIES)
def test_mutation_keeps_the_width_and_moves_the_genome(name):
    lay = layout(name)
    rng = np.random.default_rng(0)
    g = np.asarray(get(name).random(rng, lay), np.float32).reshape(-1)
    out = mutate(g, 0.3, rng, lay).reshape(-1)
    assert out.shape == g.shape and out.dtype == np.float32
    assert not np.allclose(out, g)


@pytest.mark.parametrize("name", MODALITIES)
def test_zero_strength_is_the_identity(name):
    lay = layout(name)
    rng = np.random.default_rng(1)
    g = np.asarray(get(name).random(rng, lay), np.float32).reshape(-1)
    assert np.array_equal(mutate(g, 0.0, rng, lay).reshape(-1), g)


@pytest.mark.parametrize("name", MODALITIES)
def test_a_scaled_float_never_changes_sign(name):
    """That is the whole point of scaling it. At strength 1.0 the factor stays
    inside [0.75, 1.25], so a width cannot be walked through zero however many
    generations it survives."""
    lay = layout(name)
    stride, scales = EXPECTED_SCALES[name]
    if stride is None:
        pytest.skip("mlp scales nothing")
    rng = np.random.default_rng(2)
    g = np.full(lay.length, -1.0, np.float32)
    idx = [i for i in range(lay.length) if i % stride in scales]
    for _ in range(50):
        g = mutate(g, 1.0, rng, lay).reshape(-1)
    assert np.all(g[idx] < 0.0)


@pytest.mark.parametrize("name", MODALITIES)
def test_a_child_takes_every_float_from_one_parent_or_the_other(name):
    """Crossover recombines; it must never average. A blended value is one
    neither parent held, which is a mutation wearing a crossover's name."""
    lay = layout(name)
    rng = np.random.default_rng(3)
    a = np.full(lay.length, -5.0, np.float32)
    b = np.full(lay.length, 5.0, np.float32)
    child = crossover(a, b, rng, lay).reshape(-1)
    assert np.all(np.isin(child, [-5.0, 5.0]))


@pytest.mark.parametrize("name", ["fourier", "gabor", "lenia"])
def test_crossover_keeps_a_unit_whole(name):
    """A Fourier centre, a Gabor filter and a Lenia bump are each ONE feature.
    Splitting a filter's centre from its frequency makes a child that is neither
    parent's feature. MLP is excluded: a hidden unit is not contiguous."""
    lay = layout(name)
    stride, _ = EXPECTED_SCALES[name]
    rng = np.random.default_rng(4)
    a = np.full(lay.length, -5.0, np.float32)
    b = np.full(lay.length, 5.0, np.float32)
    child = crossover(a, b, rng, lay).reshape(-1)
    for u in range(lay.shape[0]):
        unit = child[u * stride:(u + 1) * stride]
        assert len(set(unit.tolist())) == 1, f"{name} unit {u} was split"


def test_mlp_crosses_over_per_gene():
    """With no contiguous unit, per-gene is the honest operator. Over a whole
    genome the chance of one parent supplying every float is 2^-148."""
    lay = layout("mlp")
    rng = np.random.default_rng(5)
    a = np.full(lay.length, -5.0, np.float32)
    b = np.full(lay.length, 5.0, np.float32)
    child = crossover(a, b, rng, lay).reshape(-1)
    assert set(child.tolist()) == {-5.0, 5.0}


def test_fourier_keeps_its_presentation():
    """genome_spec.decode hands Fourier back as (N, 8) and callers index it by
    centre, so these operators must not flatten it underneath them."""
    lay = layout("fourier")
    rng = np.random.default_rng(6)
    g = np.zeros((lay.shape[0], 8), np.float32)
    assert mutate(g, 0.2, rng, lay).shape == (lay.shape[0], 8)
    assert crossover(g, g, rng, lay).shape == (lay.shape[0], 8)
