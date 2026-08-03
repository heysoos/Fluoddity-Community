import numpy as np
from services.genome import random_genome, mutate, crossover, GENOME_SHAPE


def test_random_genome_shape_and_dtype():
    rng = np.random.default_rng(0)
    g = random_genome(rng)
    assert g.shape == GENOME_SHAPE
    assert g.dtype == np.float32


def test_random_genome_ranges():
    rng = np.random.default_rng(1)
    g = random_genome(rng)
    freq, amp = g[:, :4], g[:, 4:]
    # amplitude in [-1, 1]; frequency in [-3, 3] (freq_scale in [1,3])
    assert amp.min() >= -1.0 and amp.max() <= 1.0
    assert freq.min() >= -3.0 and freq.max() <= 3.0


def test_random_genome_is_deterministic_per_seed():
    a = random_genome(np.random.default_rng(42))
    b = random_genome(np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_mutate_zero_strength_is_noop():
    rng = np.random.default_rng(2)
    g = random_genome(rng)
    out = mutate(g, 0.0, np.random.default_rng(2))
    assert np.array_equal(out, g)
    assert out is not g  # must be a copy


def test_mutate_changes_values_and_scales_with_strength():
    base = random_genome(np.random.default_rng(3))
    small = mutate(base, 0.05, np.random.default_rng(7))
    big = mutate(base, 0.5, np.random.default_rng(7))
    assert not np.array_equal(small, base)
    # Larger strength => larger amplitude deviation on average
    assert np.abs(big[:, 4:] - base[:, 4:]).mean() > np.abs(small[:, 4:] - base[:, 4:]).mean()


def test_crossover_takes_each_center_from_one_parent():
    a = np.zeros((10, 8), dtype=np.float32)
    b = np.ones((10, 8), dtype=np.float32)
    child = crossover(a, b, np.random.default_rng(5))
    # Every row must be all-zeros (from a) or all-ones (from b)
    for row in child:
        assert np.all(row == 0.0) or np.all(row == 1.0)
    assert child.dtype == np.float32
