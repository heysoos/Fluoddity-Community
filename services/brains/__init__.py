"""Swappable particle brains.

A modality is one GLSL function plus one Python class. `BrainLayout` is the key
everything downstream derives from: the search dimension, the checkpoint
signature, and the archive directory all come from it.

See docs/superpowers/specs/2026-08-08-brain-modalities-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# The flat parameter buffer's stride. Sized so every UI-reachable layout fits:
# Fourier 48*8=384, Gabor 36*14=504, Lenia 48*10=480, MLP H=48 -> 9*48+4=436.
MAX_BRAIN_FLOATS = 512


@dataclass(frozen=True)
class BrainLayout:
    modality: str
    shape: tuple[int, ...]
    length: int
    # Decode SCALES: the settings that change what a z MEANS without changing
    # how many floats it has - Fourier's freq scale, Gabor's envelope width,
    # Lenia's band centre.
    #
    # compare=False, and absent from signature(), on purpose. They must not
    # split the archive or force a search reset, and they safely cannot: the
    # archive stores DECODED brains, so a creature already in it is unaffected
    # by a later change here. Only how new z decode moves.
    scales: tuple[tuple[str, float], ...] = field(default=(), compare=False)

    def scale(self, key: str, default: float) -> float:
        for k, v in self.scales:
            if k == key:
                return float(v)
        return float(default)

    def __post_init__(self):
        if self.length > MAX_BRAIN_FLOATS:
            raise ValueError(
                f"{self.modality} layout needs {self.length} floats, which "
                f"exceeds MAX_BRAIN_FLOATS ({MAX_BRAIN_FLOATS})"
            )

    def signature(self) -> str:
        """Archive directory name and checkpoint guard. Must be stable across
        runs - a change here silently merges two archives."""
        parts = "-".join(f"{c}{v}" for c, v in zip("nabc", self.shape))
        return f"{self.modality}-{parts}"


@dataclass(frozen=True)
class Setting:
    """One UI knob a modality declares. Lives here, not in a modality module,
    so every modality imports it from the same place."""
    key: str
    label: str
    kind: str            # "int" | "float" | "choice"
    lo: float
    hi: float
    default: float
    choices: tuple = ()


REGISTRY: dict = {}


def register(modality) -> None:
    if modality.name in REGISTRY:
        raise ValueError(f"duplicate modality name {modality.name!r}")
    ids = {m.modality_id for m in REGISTRY.values()}
    if modality.modality_id in ids:
        raise ValueError(f"duplicate modality_id {modality.modality_id}")
    REGISTRY[modality.name] = modality


def get(name: str):
    """The named modality, or Fourier if the name is unknown.

    Never raises: an unknown name comes from a config file written by a newer
    build, and the app must keep running.
    """
    return REGISTRY.get(name) or REGISTRY["fourier"]


def default_layout() -> BrainLayout:
    return REGISTRY["fourier"].layout_from_settings({})


# The eight floats entity_update.glsl probes to decide "no brain uploaded":
# centre 0's four frequencies, and the four amplitudes of centre 5 (or the last
# centre, for a layout shorter than six). Kept in one place because the Brain
# Inspector has to reach the SAME verdict as the particles - it renders the
# generated rule when this is true, and the stored buffer when it is not.
_BLANK_PROBE_CENTRE = 5


def is_fallback(params, layout: BrainLayout) -> bool:
    """Will the shader generate a rule instead of reading these params?

    Only Fourier has a fallback: it builds FourierCenters, which is meaningless
    for any other modality, and every other modality's all-zero brain is
    SILENCE rather than a neutral start. See Sim._blank_brain.
    """
    import numpy as np

    if layout.modality != "fourier" or params is None:
        return False
    p = np.asarray(params, dtype=np.float32).reshape(-1)
    if p.size < layout.length:
        return False
    n = max(int(layout.shape[0]), 1)
    o = min(_BLANK_PROBE_CENTRE, n - 1) * 8
    return bool(not p[0:4].any() and not p[o + 4:o + 8].any())


# Registration happens on package import, so `import services.brains` is enough
# to populate REGISTRY. Placed at the bottom because each module imports
# BrainLayout/Setting/register from this one.
from services.brains import fourier as _fourier  # noqa: E402,F401
from services.brains import gabor as _gabor      # noqa: E402,F401
from services.brains import lenia as _lenia      # noqa: E402,F401
from services.brains import mlp as _mlp          # noqa: E402,F401
