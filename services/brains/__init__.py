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

# The cohort brains live in the slots after the multi-load configs, one per
# cohort, and are what "no rule loaded" means. MAX_COHORT_BRAINS must cover
# SimState.num_cohorts' maximum (144). Both are mirrored in
# shaders/brains/_header.glsl and must agree with it exactly.
COHORT_BRAIN_SLOT0 = 64
MAX_COHORT_BRAINS = 144


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


def generated_brains(layout: BrainLayout, seed: float, count: int):
    """`count` independent brains of `layout`, deterministic in `seed`.

    What "no rule loaded" means, for EVERY modality. It used to mean two
    different things: Fourier answered an all-zero buffer with a GPU-generated
    rule per cohort, and the other three got one CPU brain shared by every
    cohort. At the default MUTATION_SCALE of 0.0 that is the difference between
    64 cohorts doing 64 different things and 64 cohorts doing one thing.

    Fourier loses nothing by moving to the host: generate_random_centers() in
    fourier4_4.glsl and FourierModality.random() are the same formula -
    frequency scaled by 1+2h^2, amplitudes 2h-1 - so the family of rules is
    identical and only the particular draws differ.
    """
    import numpy as np

    m = get(layout.modality)
    rng = np.random.default_rng(int(abs(float(seed)) * 1e9) % (2 ** 32))
    return [np.asarray(m.random(rng, layout), dtype=np.float32).reshape(-1)
            for _ in range(max(int(count), 1))]


# Registration happens on package import, so `import services.brains` is enough
# to populate REGISTRY. Placed at the bottom because each module imports
# BrainLayout/Setting/register from this one.
from services.brains import fourier as _fourier  # noqa: E402,F401
from services.brains import gabor as _gabor      # noqa: E402,F401
from services.brains import lenia as _lenia      # noqa: E402,F401
from services.brains import mlp as _mlp          # noqa: E402,F401
