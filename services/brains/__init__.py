"""Swappable particle brains.

A modality is one GLSL function plus one Python class. `BrainLayout` is the key
everything downstream derives from: the search dimension, the checkpoint
signature, and the archive directory all come from it.

See docs/superpowers/specs/2026-08-08-brain-modalities-design.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# The flat parameter buffer's stride. Sized so every UI-reachable layout fits:
# Fourier 48*8=384, Gabor 36*14=504, Lenia 48*10=480, and an MLP layer stack,
# which is what needs the room - see the variable-depth MLP caveat in CLAUDE.md.
MAX_BRAIN_FLOATS = 1024

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


def layout_from_signature(sig: str, settings: dict | None = None):
    """The layout a signature names, or None if it cannot be rebuilt.

    The inverse of BrainLayout.signature(), and VERIFIED rather than assumed:
    the shape numbers are handed to the modality's own integer settings in
    schema order, the layout is rebuilt, and its signature is compared. A
    mismatch returns None instead of a plausible-looking layout of the wrong
    width, which is the one outcome worse than refusing.

    Needed because a modality's DEFAULTS are not its only shape: an archive
    entry saved under gabor-n7 cannot be reached by asking gabor for a layout,
    which answers gabor-n12.

    `settings` supplies the decode SCALES, which a signature deliberately
    leaves out. Without them the modality's defaults apply - the stored genome
    still plays back, since the archive holds it decoded, but re-encoding it
    would use different scales.
    """
    modality = str(sig).split("-")[0]
    m = REGISTRY.get(modality)
    if m is None:
        return None
    nums = []
    for part in str(sig).split("-")[1:]:
        body = part[1:]
        if not body.lstrip("-").isdigit():
            return None
        nums.append(int(body))
    keys = [s.key for s in m.settings_schema() if s.kind == "int"]
    merged = dict(settings or {})
    merged.update(dict(zip(keys, nums)))
    try:
        layout = m.layout_from_settings(merged)
    except (TypeError, ValueError, KeyError):
        return None
    return layout if layout.signature() == sig else None


def settings_of(layout: BrainLayout) -> dict:
    """The settings dict that reproduces `layout`.

    The signature carries only what changes the parameter COUNT - Fourier's
    centres, Gabor's filters, MLP's hidden width and activation. Everything
    else a modality declares is a decode SCALE, and those are deliberately
    absent from it: they change what a z means, not how many there are, so
    they must not split an archive or reset a search.

    That is right for identity and wrong for provenance. A saved rule is
    DECODED, so it plays back the same whatever the scales say - but anything
    that re-encodes it divides by them, and encode() clips at the rails. So a
    file needs this as well as the signature.

    Derived from the modality's own settings_schema rather than a hand-written
    map, because the hand-written map is what let two declared-but-never-read
    settings through before.
    """
    m = get(layout.modality)
    out: dict = {k: float(v) for k, v in layout.scales}
    structural = [s for s in m.settings_schema() if s.kind in ("int", "choice")]
    for i, s in enumerate(structural):
        if i < len(layout.shape):
            out[s.key] = int(layout.shape[i])
    return out


def brain_rng(seed: float) -> np.random.Generator:
    """The generator every "draw brains for this seed" path shares.

    One home for the seed mapping, so the cohort brains and the tournament's
    tiles agree about what a seed means. Two copies of this would drift, and the
    symptom - a modality round trip landing somewhere new - is invisible until
    someone goes back and forth looking for the brain they had.
    """
    return np.random.default_rng(int(abs(float(seed)) * 1e9) % (2 ** 32))


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
    m = get(layout.modality)
    rng = np.random.default_rng(int(abs(float(seed)) * 1e9) % (2 ** 32))
    return [np.asarray(m.random(rng, layout), dtype=np.float32).reshape(-1)
            for _ in range(max(int(count), 1))]


def unit_scale_mask(layout: BrainLayout, stride, scale_offsets) -> np.ndarray:
    """Which floats SCALE rather than offset, given one unit's field map.

    A modality calls this from scale_mask() with the same split its GLSL
    `<modality>_param_at` uses. Widths and directions scale; amplitudes, biases
    and locations offset. See services/brains/gabor.py for why that is not
    cosmetic.
    """
    if stride is None:
        return np.zeros(layout.length, dtype=bool)
    return np.isin(np.arange(layout.length) % int(stride),
                   sorted(scale_offsets))


def mutate(genome, strength: float, rng, layout: BrainLayout) -> np.ndarray:
    """Jitter a DECODED brain, respecting which of its floats are widths.

    Scales take a multiplicative jitter and offsets an additive one, which is
    what `<modality>_param_at` does per particle on the GPU. The magnitudes are
    the Fourier operator's, generalised: a scale stays inside
    [1 - strength/4, 1 + strength/4] and an offset moves by at most `strength`.

    Sign is preserved on every scaled float by construction - at strength 1.0
    the factor floor is 0.75 - so a width cannot be walked through zero however
    many generations it survives, and the envelope that divides by its square
    cannot blow up.

    The input shape is preserved: Fourier arrives as (N, 8) and must leave that
    way, because callers index it by centre.
    """
    g = np.asarray(genome, dtype=np.float32)
    if strength == 0.0:
        return g.copy()
    flat = g.reshape(-1)
    mask = get(layout.modality).scale_mask(layout)
    u = rng.random(flat.shape)
    out = np.where(mask,
                   flat * (1.0 + 0.5 * strength * (u - 0.5)),
                   flat + strength * (2.0 * u - 1.0))
    return out.astype(np.float32).reshape(g.shape)


def crossover(a, b, rng, layout: BrainLayout) -> np.ndarray:
    """Uniform crossover at the modality's UNIT.

    A Fourier centre, a Gabor filter and a Lenia bump are each one feature, so
    they cross whole - splitting a filter's centre from its frequency makes a
    child that is neither parent's feature. MLP crosses per gene: a hidden unit
    is three separate regions of the buffer (input weights, bias, output
    column), so there is no contiguous unit to keep together.

    Never blends. An averaged float is a value neither parent held, which is a
    mutation wearing a crossover's name.
    """
    a = np.asarray(a, dtype=np.float32)
    fa = a.reshape(-1)
    fb = np.asarray(b, dtype=np.float32).reshape(-1)
    stride = get(layout.modality).unit_floats(layout)
    if stride:
        units = -(-fa.size // int(stride))          # ceil, for a ragged tail
        take = np.repeat(rng.random(units) < 0.5, int(stride))[:fa.size]
    else:
        take = rng.random(fa.size) < 0.5
    return np.where(take, fa, fb).astype(np.float32).reshape(a.shape)


# Registration happens on package import, so `import services.brains` is enough
# to populate REGISTRY. Placed at the bottom because each module imports
# BrainLayout/Setting/register from this one.
from services.brains import fourier as _fourier  # noqa: E402,F401
from services.brains import gabor as _gabor      # noqa: E402,F401
from services.brains import lenia as _lenia      # noqa: E402,F401
from services.brains import mlp as _mlp          # noqa: E402,F401
