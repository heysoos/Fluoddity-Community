"""Cheap validity checks on the capture and the physics configuration.

Both failure modes here are silent: a black capture and a confounded physics
sweep each produce a fitness landscape that looks plausible and is meaningless.
"""
from __future__ import annotations

import numpy as np

# Tournament mode takes ownership of this one (it is driven in sim.py), so a
# sweep on it is neutralised rather than warned about.
_OWNED = {"MUTATION_SCALE"}


def check_capture(crops: np.ndarray) -> str | None:
    """Return a warning string, or None when the capture looks usable."""
    if crops.size == 0:
        return "Capture was empty - the offscreen framebuffer produced no pixels."
    if int(crops.max()) == 0:
        return (
            "Capture is entirely black. Fitness will be flat and meaningless - "
            "check that the offscreen framebuffer is being drawn into."
        )
    mean = float(crops.mean())
    if mean < 2.0:
        return f"Capture is nearly black (mean {mean:.1f}/255); fitness will be flat."
    if mean > 253.0:
        return f"Capture is blown out (mean {mean:.1f}/255); fitness will be flat."
    return None


def is_viable_tile(crop: np.ndarray) -> bool:
    """One tile's crop -> is it worth measuring at all?

    Same thresholds as check_capture, applied per tile rather than per batch.
    Novelty search must reject one dead tile on its own: discarding the whole
    generation because a single genome died would throw away the other 15
    useful samples with it.
    """
    if crop.size == 0:
        return False
    if int(crop.max()) == 0:
        return False
    return 2.0 <= float(crop.mean()) <= 253.0


COHERENCE_LAGS = (1, 2, 3, 4, 6, 8, 12, 16)


def structure(crops: np.ndarray, lags=COHERENCE_LAGS) -> np.ndarray:
    """(n, H, W, 3) uint8 -> (n,) in [0, 1]. 1 is coherent, 0 is white noise.

    Best |spatial autocorrelation| over several lags and both axes: is this
    image self-similar at ANY offset? Multiple lags because lag 1 alone scores
    a fine regular lattice the same as static - see CLAUDE.md.

    Multiplies expedition fitness because a single-reference contrastive score
    (latent/chase goals) is a monotone squash of raw cosine, which noise
    maximises. Not a CLIP term: image-image and image-text similarity sit at
    different scales, so distractors can't just join the reference set. This
    runs on crops already in memory instead.

    A flat tile scores 1.0 (no high frequencies) - not a hole, since
    is_viable_tile already rejects blank captures.

    Absolute value, axes taken separately: a period-2 stripe is
    anti-correlated at lag 1, which is still structure, and a pattern can be
    organised along one axis and not the other.
    """
    a = np.asarray(crops)
    if a.ndim != 4 or a.shape[0] == 0:
        return np.zeros(max(0, a.shape[0] if a.ndim else 0), dtype=np.float32)
    # float32 is enough and half the traffic of float64 at 64x224x224.
    g = a.astype(np.float32).mean(axis=3)
    g -= g.mean(axis=(1, 2), keepdims=True)
    energy = (g * g).mean(axis=(1, 2))
    var = np.maximum(energy, 1e-12)
    best = np.zeros(len(g), dtype=np.float32)
    for lag in lags:
        if lag >= min(g.shape[1], g.shape[2]):
            break
        rx = np.abs((g[:, :, :-lag] * g[:, :, lag:]).mean(axis=(1, 2)) / var)
        ry = np.abs((g[:, :-lag, :] * g[:, lag:, :]).mean(axis=(1, 2)) / var)
        best = np.maximum(best, np.maximum(rx, ry))
    return np.clip(np.where(energy > 1e-9, best, 1.0), 0.0, 1.0).astype(np.float32)


def sweeping_parameters(sim_state) -> list[str]:
    """Parameters whose value differs across tiles.

    Tiles partition position space and cohort index, so a non-zero x/y/cohort
    sweep confounds any comparison of genomes across tiles.

    Gated on parameter_sweeps_enabled, matching what reaches the GPU
    (_assign_physics_setting zeroes every sweep while the toggle is off, so a
    preset can carry inert sweep values without warning). Mirrors
    Sim.has_active_xy_sweep / has_active_cohort_sweep.

    Jitter is not counted: per-particle noise, not a systematic gradient, so
    it doesn't bias one tile against another.
    """
    if not getattr(sim_state, "parameter_sweeps_enabled", False):
        return []

    names: set[str] = set()
    for attr in ("x_sweeps", "y_sweeps", "cohort_sweeps"):
        for key, value in (getattr(sim_state, attr, None) or {}).items():
            if value and key not in _OWNED:
                names.add(key)
    return sorted(names)
