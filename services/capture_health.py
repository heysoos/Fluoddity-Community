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
