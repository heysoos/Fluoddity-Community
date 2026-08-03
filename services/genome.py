"""Pure genome operations for the interactive tournament.

A genome is a (10, 8) float32 array: columns 0-3 = FourierCenter frequency,
columns 4-7 = amplitude. This matches the GLSL Rule struct byte layout exactly.
The distributions mirror generate_random_centers() in shaders/fourier4_4.glsl and
mutate_rule() in shaders/entity_update.glsl (distribution-level, not bit-exact —
genomes are generated on the CPU and uploaded verbatim).
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


def mutate(genome: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Additive jitter on amplitude, multiplicative jitter on frequency."""
    g = genome.astype(np.float32).copy()
    if strength == 0.0:
        return g
    freq = g[:, :4]
    amp = g[:, 4:]
    amp += strength * (-1.0 + 2.0 * rng.random((N_CENTERS, 4)))
    freq *= 1.0 + 0.5 * strength * (rng.random((N_CENTERS, 4)) - 0.5)
    return g.astype(np.float32)


def crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform per-center crossover: each of the 10 centers comes from a or b."""
    mask = rng.random(N_CENTERS) < 0.5           # True => take from a
    out = np.where(mask[:, None], a, b)
    return out.astype(np.float32)
