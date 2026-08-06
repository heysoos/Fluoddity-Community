"""Physics as part of the search space.

Measured motivation: with the brain alone, whether a run improves is decided by
the (starting preset x prompt) pair. On the default config "glowing coral"
gained +0.225 mean fitness over 25 generations, but the same optimizer on
HungryHungryHippos gained +0.008 - its physics leaves the brain almost no
leverage over the image. The optimizer needs to move the physics too.

Parameters are searched RELATIVE TO THE LOADED PRESET:

    value = origin + span * tanh(z)

not as a position inside a fixed absolute range. The absolute version was
broken in two ways at once. A preset may hold a value outside a parameter's
nominal range - HungryHungryHippos stores global_force_mult = -0.341 against a
nominal (0.0, 2.0) - and the slider machinery accommodates that by expanding the
stored range to *exactly* that value. Either way the origin lands on a boundary:

  - it did not round-trip: -0.341 encoded to z = -4.605 and decoded back to
    0.0002, so merely enabling physics search switched the forces off before
    generation 1
  - the tanh gradient there is 4e-4, so CMA-ES could never move the gene again

Origin-relative fixes both: z = 0 reproduces the preset exactly, and the
gradient at the origin is `span`, its maximum. Still bounded, so no clipping
repair bias in the covariance estimate.

MUTATION_SCALE is deliberately absent: tournament mode owns it. HAZARD_RATE is
absent because respawning is a global aesthetic, not a morphology knob.
"""
from __future__ import annotations

import numpy as np

# (SimState field, GLSL config field, nominal low, nominal high).
# The nominal range no longer positions the value - it only sets how far the
# optimizer may roam from the preset (see SPAN_FRACTION).
PHYSICS_PARAMS: list[tuple[str, str, float, float]] = [
    ("SENSOR_DISTANCE", "sensor_distance", 0.0, 4.0),
    ("SENSOR_ANGLE", "sensor_angle", -1.0, 1.0),
    ("SENSOR_GAIN", "sensor_gain", 0.0, 5.0),
    ("AXIAL_FORCE", "axial_force", -1.0, 1.0),
    ("LATERAL_FORCE", "lateral_force", -1.0, 1.0),
    ("STRAFE_POWER", "strafe_power", 0.0, 0.5),
    ("GLOBAL_FORCE_MULT", "global_force_mult", 0.0, 2.0),
    ("DRAG", "drag", -1.0, 1.0),
]

PHYSICS_DIM = len(PHYSICS_PARAMS)

# How far from the preset a gene may travel, as a fraction of the nominal range.
# 0.5 lets a parameter move half its nominal span either way, so the reachable
# interval is the nominal width centred on wherever the preset happens to sit.
SPAN_FRACTION = 0.5

EPS = 1e-4


def spans() -> np.ndarray:
    return np.array([SPAN_FRACTION * (hi - lo) for _n, _g, lo, hi in PHYSICS_PARAMS],
                    dtype=np.float64)


def default_origin() -> dict[str, float]:
    """Nominal midpoints, used only when no preset has been supplied."""
    return {n: (lo + hi) / 2.0 for n, _g, lo, hi in PHYSICS_PARAMS}


def _origin_vec(origin: dict[str, float] | None) -> np.ndarray:
    src = default_origin() if not origin else origin
    fallback = default_origin()
    return np.array([float(src.get(n, fallback[n]))
                     for n, _g, _lo, _hi in PHYSICS_PARAMS], dtype=np.float64)


def decode_physics(z: np.ndarray,
                   origin: dict[str, float] | None = None) -> dict[str, float]:
    """(8,) unbounded -> {SimState field: value}, centred on `origin`.

    z = 0 reproduces `origin` exactly, which is what makes enabling the toggle
    a no-op until the optimizer actually moves.
    """
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if z.size != PHYSICS_DIM:
        raise ValueError(f"expected {PHYSICS_DIM} physics genes, got {z.size}")
    values = _origin_vec(origin) + spans() * np.tanh(z)
    return {n: float(v) for (n, _g, _lo, _hi), v in zip(PHYSICS_PARAMS, values)}


def encode_physics(values: dict[str, float],
                   origin: dict[str, float] | None = None) -> np.ndarray:
    """Inverse of decode_physics. encode(origin, origin) is exactly zero."""
    o = _origin_vec(origin)
    s = spans()
    v = np.array([float(values.get(n, o[i]))
                  for i, (n, _g, _lo, _hi) in enumerate(PHYSICS_PARAMS)],
                 dtype=np.float64)
    t = np.clip((v - o) / s, -1.0 + EPS, 1.0 - EPS)
    return np.arctanh(t).astype(np.float32)


def midpoint_z() -> np.ndarray:
    """z = 0 decodes to the origin; the neutral starting point."""
    return np.zeros(PHYSICS_DIM, dtype=np.float32)
