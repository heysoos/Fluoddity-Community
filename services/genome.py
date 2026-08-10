"""The Fourier genome's shape and its generator.

A genome is a (10, 8) float32 array: columns 0-3 = frequency, 4-7 = amplitude,
which matches the flat brain buffer's byte layout exactly.

The mutate/crossover that lived here were Fourier-only and are gone. Anything
that breeds a brain now goes through services.brains, which asks the MODALITY
which of its floats scale and which offset - see FourierModality.SCALE_OFFSETS.

random_genome stays because it is the Fourier prior itself, still used by
tools/calibrate_imgep.py and by the GPU tests as a known-good rule.
FourierModality.random() is the same formula against a BrainLayout.
"""
import numpy as np

N_CENTERS = 10
GENOME_SHAPE = (N_CENTERS, 8)


def random_genome(rng: np.random.Generator) -> np.ndarray:
    """A fresh random brain. Frequencies biased toward low magnitude."""
    freq_scale = 1.0 + 2.0 * rng.random((N_CENTERS, 4)) ** 2      # [1, 3]
    freq = (rng.random((N_CENTERS, 4)) * 2.0 - 1.0) * freq_scale  # [-3, 3], low-biased
    amp = rng.random((N_CENTERS, 4)) * 2.0 - 1.0                  # [-1, 1]
    return np.concatenate([freq, amp], axis=1).astype(np.float32)
