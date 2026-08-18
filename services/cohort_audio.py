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

# Stands in for an unbounded clamp on a row nothing has masked, so such a row
# is exactly identity and cannot clip what a sweep or a jitter produced.
_OPEN = 1.0e30

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


def build_arrays(mappings, targets, signals, states, strengths,
                 global_strength: float, dt: float, deaf, n_cohorts: int,
                 held=(), rate_scale: float = 1.0,
                 apply_shapers: bool = True) -> tuple[np.ndarray, bool]:
    """Per-cohort gain and offset for every maskable parameter.

    The modulation chain is affine in the base value, so a cohort's whole
    contribution is one multiply and one add:

        gain   = 1 + S * (M - 1)
        offset = S * A * M

    with A the summed add/subtract terms, M the product of the multiply terms
    and S the combined strength. Nothing there reads the base, which is what
    lets this run per cohort on the CPU and land on top of a sweep.

    Indexed [row, COHORT], because a cohort is all the shader has. One extra
    entry per row, at MASK_SLOTS, carries that parameter's (lo, hi) - the same
    bounds modulate() clamps to. A row with no masked mapping keeps the open
    bounds, so it stays exactly identity and cannot clip a sweep.

    The second return says whether any mapping is masked; when it is False the
    array is identity and the caller may skip the upload.
    """
    from services.audio_shapers import ShaperState

    rows = len(COHORT_AUDIO_PARAMS)
    arr = np.empty((rows, MASK_SLOTS + 1, 2), dtype=np.float32)
    arr[..., 0] = 1.0
    arr[..., 1] = 0.0
    arr[:, MASK_SLOTS, 0] = -_OPEN
    arr[:, MASK_SLOTS, 1] = _OPEN

    by_target = {t.key: t for t in targets}
    live = []
    any_masked = False
    for m in mappings:
        if m.target not in COHORT_AUDIO_PARAMS:
            continue
        if not m.enabled or m.target in deaf or m.target not in by_target:
            continue
        if m.signal not in signals:
            continue
        if not is_full(m.cohorts):
            any_masked = True
        live.append(m)

    if not any_masked:
        return arr, False

    n = min(MASK_SLOTS, max(1, int(n_cohorts)))
    # The slot each live cohort reads. The OUTPUT is indexed by cohort, not by
    # slot: the shader has a cohort and no way to recover a slot from it.
    slots = np.minimum(MASK_SLOTS - 1,
                       ((np.arange(n) + 0.5) / n * MASK_SLOTS).astype(np.int64))

    shaped: dict[int, float] = {}
    for m in live:
        s = min(1.0, max(0.0, signals[m.signal] * m.gain))
        if apply_shapers:
            s = states.setdefault(m.uid, ShaperState()).apply(
                s, dt, m.shaper, m.signal not in held, rate_scale)
        shaped[m.uid] = s

    for row, key in enumerate(COHORT_AUDIO_PARAMS):
        bound = [m for m in live if m.target == key]
        if not bound:
            continue
        t = by_target[key]
        span = t.hi - t.lo
        strength = float(strengths.get(key, 1.0)) * float(global_strength)

        a = np.zeros(n, dtype=np.float64)
        mul = np.ones(n, dtype=np.float64)
        for m in bound:
            covered = m.cohorts[slots]
            s = shaped[m.uid]
            if m.mode == "multiply":
                mul[covered] *= 1.0 + s * m.depth
            else:
                sign = -1.0 if m.mode == "subtract" else 1.0
                a[covered] += sign * s * m.depth * span

        arr[row, :n, 0] = (1.0 + strength * (mul - 1.0)).astype(np.float32)
        arr[row, :n, 1] = (strength * a * mul).astype(np.float32)
        arr[row, MASK_SLOTS, 0] = t.hard_lo if t.hard_lo is not None else t.lo
        arr[row, MASK_SLOTS, 1] = t.hard_hi if t.hard_hi is not None else t.hi

    return arr, True
