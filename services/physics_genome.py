"""Physics as part of the search space.

Measured motivation: with the brain alone, whether a run improves is decided by
the (starting preset x prompt) pair. On the default config "glowing coral"
gained +0.225 mean fitness over 25 generations, but the same optimizer on
HungryHungryHippos gained +0.008 - its physics leaves the brain almost no
leverage over the image. The optimizer needs to move the physics too.

Each parameter is decoded from an unbounded z with the same bounded tanh squash
the brain uses, mapped onto that slider's real range. Ranges match the defaults
in Sim._assign_physics_setting.

MUTATION_SCALE is deliberately absent: tournament mode owns it. HAZARD_RATE is
absent because respawning is a global aesthetic, not a morphology knob.
"""
from __future__ import annotations

import numpy as np

# (SimState field, GLSL config field, low, high)
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


def decode_physics(z: np.ndarray) -> dict[str, float]:
    """(8,) unbounded -> {SimState field: value} inside each slider's range.

    tanh keeps the map smooth and bounded, so CMA-ES never has to be repaired at
    an edge - clipping would collapse many distinct z onto one genome and skew
    the covariance estimate.
    """
    z = np.asarray(z, dtype=np.float32).reshape(-1)
    if z.size != PHYSICS_DIM:
        raise ValueError(f"expected {PHYSICS_DIM} physics genes, got {z.size}")
    out: dict[str, float] = {}
    for zi, (name, _glsl, lo, hi) in zip(z, PHYSICS_PARAMS):
        t = (np.tanh(float(zi)) + 1.0) * 0.5      # 0..1
        out[name] = float(lo + (hi - lo) * t)
    return out


def encode_physics(values: dict[str, float]) -> np.ndarray:
    """Inverse of decode_physics, for seeding a search from a real config."""
    eps = 1e-4
    z = np.zeros(PHYSICS_DIM, dtype=np.float32)
    for i, (name, _glsl, lo, hi) in enumerate(PHYSICS_PARAMS):
        v = float(values.get(name, (lo + hi) * 0.5))
        t = (v - lo) / (hi - lo) if hi > lo else 0.5
        t = min(max(t, eps), 1.0 - eps)
        z[i] = np.arctanh(2.0 * t - 1.0)
    return z


def midpoint_z() -> np.ndarray:
    """z = 0 decodes to the middle of every range; the neutral starting point."""
    return np.zeros(PHYSICS_DIM, dtype=np.float32)
