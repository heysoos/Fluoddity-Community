"""Which cohorts an audio mapping drives.

A mask over the NORMALISED cohort axis, not over cohort indices: a live cohort
reads the slot its position falls in, so a mask keeps its proportions when the
cohort count moves. Changing the count writes nothing, which is what makes it
lossless; painting at a coarse count writes wide spans, which is not.
"""
from __future__ import annotations

import numpy as np

from services.cohort_tiling import MAX_COHORTS

MASK_SLOTS = MAX_COHORTS

# The parameters a mask can reach, in the row order the SSBO uses. Mirrored as
# CA_* in shaders/cohort_audio.glsl; the two are compared by a test. A
# parameter belongs here only if entity_update.glsl evaluates it per particle.
COHORT_AUDIO_PARAMS: tuple[str, ...] = (
    "SENSOR_GAIN", "SENSOR_ANGLE", "SENSOR_DISTANCE", "MUTATION_SCALE",
    "GLOBAL_FORCE_MULT", "DRAG", "AXIAL_FORCE", "LATERAL_FORCE",
    "STRAFE_POWER", "HAZARD_RATE",
)


def full_mask() -> np.ndarray:
    return np.ones(MASK_SLOTS, dtype=bool)


def slot_of(cohort: int, n: int) -> int:
    if n <= 0:
        return 0
    return min(MASK_SLOTS - 1, int((cohort + 0.5) / n * MASK_SLOTS))


def paint_span(cell: int, n: int) -> tuple[int, int]:
    """Half-open slot range one strip cell owns. Never empty."""
    if n <= 0:
        return 0, MASK_SLOTS
    lo = min(MASK_SLOTS - 1, int(cell / n * MASK_SLOTS))
    hi = min(MASK_SLOTS, max(lo + 1, int((cell + 1) / n * MASK_SLOTS)))
    return lo, hi


def paint(mask: np.ndarray, cell: int, n: int, value: bool) -> None:
    lo, hi = paint_span(cell, n)
    mask[lo:hi] = value


def covers(mask: np.ndarray, cohort: int, n: int) -> bool:
    return bool(mask[slot_of(cohort, n)])


def cells_lit(mask: np.ndarray, n: int) -> np.ndarray:
    """One entry per live cohort - what the strip draws."""
    if n <= 0:
        return np.zeros(0, dtype=bool)
    idx = np.minimum(MASK_SLOTS - 1,
                     ((np.arange(n) + 0.5) / n * MASK_SLOTS).astype(np.int64))
    return mask[idx]


def is_full(mask: np.ndarray) -> bool:
    return bool(np.all(mask))


def is_empty(mask: np.ndarray) -> bool:
    return not bool(np.any(mask))
