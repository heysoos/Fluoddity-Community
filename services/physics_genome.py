"""Physics as part of the search space.

With the brain alone, whether a run improves is decided by the (starting
preset x prompt) pair - some presets' physics leave the brain almost no
leverage over the image, so the optimizer needs to move the physics too.

Parameters are searched RELATIVE TO THE LOADED PRESET:

    value = origin + span * tanh(z)

not as a position inside a fixed absolute range. A preset can hold a value
outside a parameter's nominal range (the slider machinery expands the stored
range to fit it), which puts an absolute encoding's origin on a boundary where
it neither round-trips nor has a usable gradient. Origin-relative fixes both:
z = 0 reproduces the preset exactly, and the gradient at the origin is the
reach, its maximum. Still bounded, so no clipping repair bias in the covariance
estimate.

The nominal range sets the roam DISTANCE and never the value, so it cannot on
its own keep a gene inside a hard limit - see `reach()`.

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
    """Nominal roam distance per parameter. What a gene may ACTUALLY travel is
    `reach()`, which is this bounded by the parameter's hard limits."""
    return np.array([SPAN_FRACTION * (hi - lo) for _n, _g, lo, hi in PHYSICS_PARAMS],
                    dtype=np.float64)


_HARD: tuple[np.ndarray, np.ndarray] | None = None


def _hard_vecs() -> tuple[np.ndarray, np.ndarray]:
    """(hard_lo, hard_hi) in PHYSICS_PARAMS order, +-inf where unbounded.

    Read from ui.physics_params, which is the one home for a parameter's
    limits; a copy here is how the two tables drift apart. Imported lazily and
    cached because `ui` pulls in the whole UI package.
    """
    global _HARD
    if _HARD is None:
        from ui.physics_params import PARAM_BY_NAME

        lo, hi = [], []
        for n, _g, _l, _h in PHYSICS_PARAMS:
            p = PARAM_BY_NAME.get(n)
            lo.append(-np.inf if p is None or p.hard_min is None else p.hard_min)
            hi.append(np.inf if p is None or p.hard_max is None else p.hard_max)
        _HARD = (np.array(lo, dtype=np.float64), np.array(hi, dtype=np.float64))
    return _HARD


def reach(origin: dict[str, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(up, down): how far each gene may travel from `origin` in each direction.

    The span alone bounds the DISTANCE and not the value, so a preset near a
    hard limit could be searched straight past it. The reach is shrunk to fit
    rather than the decoded value being clamped: clamping would map a band of z
    onto one value and hand the optimizer a plateau, while shrinking keeps the
    map smooth and keeps z = 0 on the preset. See docs/imgep.md.

    Asymmetric by construction, and zero on a side whose limit the preset
    already sits on - which is correct, since the only way from a ceiling is
    down.
    """
    o = _origin_vec(origin)
    s = spans()
    hlo, hhi = _hard_vecs()
    return (np.maximum(np.minimum(s, hhi - o), 0.0),
            np.maximum(np.minimum(s, o - hlo), 0.0))


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
    t = np.tanh(z)
    up, dn = reach(origin)
    values = _origin_vec(origin) + np.where(t >= 0.0, up, dn) * t
    return {n: float(v) for (n, _g, _lo, _hi), v in zip(PHYSICS_PARAMS, values)}


def encode_physics(values: dict[str, float],
                   origin: dict[str, float] | None = None
                   ) -> tuple[np.ndarray, int]:
    """Inverse of decode_physics -> ((8,) float32, n_clamped).

    encode(origin, origin) is exactly zero. This is the ONLY lossy direction:
    decode is bounded by tanh, so it cannot leave `origin +- span`, while a
    value further than one span from `origin` has no z at all and comes back as
    the nearest one that does. The count is what a caller warns on, the same
    contract as genome_spec.encode's.
    """
    o = _origin_vec(origin)
    v = np.array([float(values.get(n, o[i]))
                  for i, (n, _g, _lo, _hi) in enumerate(PHYSICS_PARAMS)],
                 dtype=np.float64)
    d = v - o
    up, dn = reach(origin)
    scale = np.where(d >= 0.0, up, dn)
    raw = np.zeros_like(d)
    live = scale > 0.0
    raw[live] = d[live] / scale[live]
    # No reach at all on that side, so any offset is infinitely far - which
    # clips, and must count as clipped rather than silently reading as z = 0.
    stuck = ~live & (d != 0.0)
    raw[stuck] = np.sign(d[stuck]) * np.inf
    t = np.clip(raw, -1.0 + EPS, 1.0 - EPS)
    return np.arctanh(t).astype(np.float32), int(np.count_nonzero(raw != t))


def midpoint_z() -> np.ndarray:
    """z = 0 decodes to the origin; the neutral starting point."""
    return np.zeros(PHYSICS_DIM, dtype=np.float32)
