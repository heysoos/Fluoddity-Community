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


def structure(crops: np.ndarray) -> np.ndarray:
    """(n, H, W, 3) uint8 -> (n,) in [0, 1]. 1 is coherent, 0 is white noise.

    Lag-1 spatial autocorrelation of the luminance, averaged over both axes.
    Neighbouring pixels of a trail pattern agree; neighbouring pixels of noise
    do not. Validated on synthetics: white noise scores 0.00, a smooth ramp
    0.99. Measured over the real archives, entries run 0.93-0.98.

    This exists because a contrastive fitness with ONE reference - which is
    what a latent or chase goal has, the archive centroid - is a monotone
    squash of raw cosine to the goal, and raw cosine to an arbitrary direction
    is maximised by high-frequency noise, which carries energy everywhere.
    Text goals never had this problem: DEFAULT_DISTRACTORS contains "random
    noise" and "an abstract texture" precisely to reject it. Measured
    2026-08-10 on two real archives, the entry a latent goal ranks FIRST is in
    the archive's roughest decile 34.0% and 32.5% of the time, against the 10%
    a blind draw gives.

    Deliberately not a CLIP term. The distractors cannot simply be added to a
    latent goal's reference set: image-image similarity sits near 0.9 and
    image-text near 0.2, so at one logit scale the text references contribute
    nothing but a constant, which a softmax is invariant to. This is free, runs
    on crops already in memory, and cannot inherit CLIP's own blind spots.

    A flat tile scores 1.0 - it has no high frequencies. That is correct here
    and not a hole: is_viable_tile already rejects blank captures.
    """
    a = np.asarray(crops)
    if a.ndim != 4 or a.shape[0] == 0:
        return np.zeros(max(0, a.shape[0] if a.ndim else 0), dtype=np.float32)
    # float32 is enough and half the traffic of float64 at 64x224x224.
    g = a.astype(np.float32).mean(axis=3)
    g -= g.mean(axis=(1, 2), keepdims=True)
    var = (g * g).mean(axis=(1, 2))
    rx = (g[:, :, :-1] * g[:, :, 1:]).mean(axis=(1, 2))
    ry = (g[:, :-1, :] * g[:, 1:, :]).mean(axis=(1, 2))
    rho = np.where(var > 1e-9, 0.5 * (rx + ry) / np.maximum(var, 1e-12), 1.0)
    return np.clip(rho, 0.0, 1.0).astype(np.float32)


def sweeping_parameters(sim_state) -> list[str]:
    """Parameters whose value differs across tiles.

    Tiles partition both position space and cohort index, so any parameter with
    a non-zero x, y or cohort sweep takes different values in different tiles.
    Neither human selection nor a CLIP score is then comparing genomes on equal
    terms - the comparison is confounded by position.

    Gated on parameter_sweeps_enabled, matching what actually reaches the GPU:
    _assign_physics_setting sends 0.0 for every sweep while the master toggle is
    off, so the stored values are inert. Presets routinely carry sweep values
    with the toggle off, and warning about those cries wolf. This mirrors
    Sim.has_active_xy_sweep / has_active_cohort_sweep.

    Jitter is deliberately not counted: it is per-particle noise, not a
    systematic gradient, so it does not bias one tile against another.
    """
    if not getattr(sim_state, "parameter_sweeps_enabled", False):
        return []

    names: set[str] = set()
    for attr in ("x_sweeps", "y_sweeps", "cohort_sweeps"):
        for key, value in (getattr(sim_state, attr, None) or {}).items():
            if value and key not in _OWNED:
                names.add(key)
    return sorted(names)
