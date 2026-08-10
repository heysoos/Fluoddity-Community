"""The Fourier prior itself.

mutate/crossover used to live here and were Fourier-only; their tests moved to
tests/test_brain_genome_ops.py, which runs them against every modality and pins
the scale/offset split against each shader's own <modality>_param_at.
"""
import numpy as np
from services.genome import random_genome, GENOME_SHAPE


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
