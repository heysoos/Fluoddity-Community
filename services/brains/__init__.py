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
    # Audio channels fed to the brain beside the four sensor taps. Structural:
    # every unit gains this many weights. Kept out of `shape` so positional
    # indexing and every signature parser are untouched, and 0 is a layout
    # that is character-identical to one written before it existed.
    audio_inputs: int = 0

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
        runs - a change here silently merges two archives.

        A modality may own its own format by declaring `signature_of`, which
        `layout_from_signature` then parses back through its
        `settings_from_signature`. MLP does, because a variable-length layer
        stack does not fit four positional ints - and a modality that formats
        also parses, so the two cannot disagree.
        """
        m = REGISTRY.get(self.modality)
        fn = getattr(m, "signature_of", None)
        base = fn(self) if fn is not None else default_signature(self)
        return f"{base}+a{self.audio_inputs}" if self.audio_inputs else base


def default_signature(layout: BrainLayout) -> str:
    """`modality-n<shape0>-a<shape1>-...`, up to four positional ints."""
    parts = "-".join(f"{c}{v}" for c, v in zip("nabc", layout.shape))
    return f"{layout.modality}-{parts}"


@dataclass(frozen=True)
class Setting:
    """One UI knob a modality declares. Lives here, not in a modality module,
    so every modality imports it from the same place.

    "layers" is a REPEATED structural group rather than one value: `lo`/`hi`
    bound each entry's width, `default` is a new entry's width, and `choices`
    names the activations. Its value is a list of [width, activation] pairs.
    """
    key: str
    label: str
    kind: str            # "int" | "float" | "choice" | "layers"
    lo: float
    hi: float
    default: float
    choices: tuple = ()


# The kinds that change the PARAMETER COUNT, and so the genome's meaning. A
# change to one of these resets the optimizer and switches archive; everything
# else is a decode SCALE and is free to change mid-run.
STRUCTURAL_KINDS = ("int", "choice", "layers")

# Mirrored in shaders/brains/_header.glsl as MAX_AUDIO_INPUTS.
MAX_AUDIO_INPUTS = 8

# Declared ONCE and appended by every modality's schema, so the count and the
# scale mean the same thing under every brain. The count is structural; the
# scale is a decode scale, so it moves live and reaches zero, where the brain
# is deaf without a weight being touched.
AUDIO_INPUTS_SETTING = Setting("audio_inputs", "Audio Inputs", "int",
                               0, MAX_AUDIO_INPUTS, 0)
AUDIO_SCALE_SETTING = Setting("audio_scale", "Audio Scale", "float",
                              0.0, 4.0, 1.0)


def audio_inputs_of(s: dict) -> int:
    """The count a settings dict asks for, bounded. Raises past the maximum,
    as a width past MAX_BRAIN_FLOATS does."""
    k = int(s.get("audio_inputs", 0))
    if not 0 <= k <= MAX_AUDIO_INPUTS:
        raise ValueError(f"audio_inputs {k} is outside 0..{MAX_AUDIO_INPUTS}")
    return k


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
    base, _sep, suffix = str(sig).partition("+a")
    if _sep and not suffix.isdigit():
        return None
    modality = base.split("-")[0]
    m = REGISTRY.get(modality)
    if m is None:
        return None
    parse = getattr(m, "settings_from_signature", None)
    structural = (parse(base) if parse is not None
                  else default_settings_from_signature(m, base))
    if structural is None:
        return None
    merged = dict(settings or {})
    merged.update(structural)
    merged["audio_inputs"] = int(suffix) if _sep else 0
    try:
        layout = m.layout_from_settings(merged)
    except (TypeError, ValueError, KeyError):
        return None
    return layout if layout.signature() == sig else None


def default_settings_from_signature(m, sig: str):
    """The structural settings `sig` names, in schema order, or None.

    The inverse of default_signature(), and it must consume the same kinds
    settings_of() emits. It used to take only `kind == "int"`, which silently
    dropped a `choice`: an mlp-n16-a1 signature rebuilt as activation 0 and the
    round-trip check refused it, so no sin or gelu entry could be adopted or
    previewed at all.
    """
    nums = []
    for part in sig.split("-")[1:]:
        body = part[1:]
        if not body.lstrip("-").isdigit():
            return None
        nums.append(int(body))
    keys = [s.key for s in positional_structure(m)]
    return dict(zip(keys, nums))


def positional_structure(m) -> list:
    """The structural settings a signature carries POSITIONALLY. The audio
    input count is structural too, but it rides as a suffix rather than a
    position, so every parser that zips shape numbers against the schema has
    to leave it out."""
    return [s for s in m.settings_schema()
            if s.kind in STRUCTURAL_KINDS and s.key != "audio_inputs"]


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
    settings through before. A modality whose structure is not four positional
    ints declares `settings_of` and takes over that half.
    """
    m = get(layout.modality)
    out: dict = {k: float(v) for k, v in layout.scales}
    out["audio_inputs"] = int(layout.audio_inputs)
    fn = getattr(m, "settings_of", None)
    if fn is not None:
        out.update(fn(layout))
        return out
    for i, s in enumerate(positional_structure(m)):
        if i < len(layout.shape):
            out[s.key] = int(layout.shape[i])
    return out


def audio_weight_index(layout: BrainLayout) -> np.ndarray:
    """Flat indices, into the DECODED brain, of every audio weight.

    A modality owns this because it owns its packing; everything generic -
    the transfer, the shrink, the Inspector's readout - is written against
    these indices rather than against a reading of each modality's table.
    Empty at zero inputs.
    """
    if not layout.audio_inputs:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(get(layout.modality).audio_weight_index(layout),
                      dtype=np.int64).reshape(-1)


def audio_z_index(layout: BrainLayout) -> np.ndarray:
    """The same weights' coordinates in the SEARCH vector. Identical to the
    decoded indices unless a modality orders z by block rather than by unit,
    which Fourier does."""
    if not layout.audio_inputs:
        return np.zeros(0, dtype=np.int64)
    m = get(layout.modality)
    fn = getattr(m, "audio_z_index", None)
    if fn is None:
        return audio_weight_index(layout)
    return np.asarray(fn(layout), dtype=np.int64).reshape(-1)


def unit_count(layout: BrainLayout) -> int:
    """Units this brain decomposes into - centres, filters, bumps, or the
    hidden units the Inspector can draw. shape[0] unless the modality says
    otherwise; MLP does, because a deep stack's drawable units are its LAST
    hidden layer's."""
    fn = getattr(get(layout.modality), "unit_count", None)
    if fn is not None:
        return int(fn(layout))
    return int(layout.shape[0]) if layout.shape else 0


def layout_defines(layout: BrainLayout) -> dict:
    """Preprocessor constants the brain shaders must be COMPILED with.

    Separate from layout_uniforms because a uniform cannot size a local array.
    Anything in here changes the program, so it is also the cache key sim.py
    keeps one compiled entity-update shader per.

    EVERY modality is asked, not just the running one: one program carries all
    four shaders, so MLP's scratch arrays are allocated per invocation while a
    Fourier brain runs. A modality answers for its own constants given whatever
    layout is live, and the answer for someone else's layout is its floor.
    """
    out: dict = {}
    for m in REGISTRY.values():
        fn = getattr(m, "shader_defines", None)
        if fn is not None:
            out.update(fn(layout))
    return out


def with_audio_inputs(layout: BrainLayout, k: int) -> BrainLayout:
    """The same layout with `k` audio inputs. Scales and structure kept."""
    k = int(k)
    if not 0 <= k <= MAX_AUDIO_INPUTS:
        raise ValueError(f"audio inputs {k} is outside 0..{MAX_AUDIO_INPUTS}")
    return get(layout.modality).layout_from_settings(
        {**settings_of(layout), "audio_inputs": k})


def channel_rng(seed: float, k: int) -> np.random.Generator:
    """The generator channel k's audio weights are drawn from.

    Keyed by (seed, k) and nothing else, so a channel's weights are the same
    whether it was the first added or the fifth, and typing a seed back in
    gives the weights it gave before.
    """
    return np.random.default_rng(
        [int(abs(float(seed)) * 1e9) % (2 ** 32), int(k)])


def audio_aware_normal(rng, layout: BrainLayout, sigma: float,
                       deaf_transform=None) -> np.ndarray:
    """A normal z of `layout.length`: the DEAF layout's draw first, in its own
    order, then the audio weights.

    So the non-audio floats of a wide brain are exactly the deaf brain's from
    the same generator, which is what keeps a "no rule loaded" creature
    unchanged when the count moves. `deaf_transform` is applied to the deaf
    draw before placement, for a modality that normalises part of it.
    """
    deaf = with_audio_inputs(layout, 0)
    z0 = rng.normal(0.0, sigma, deaf.length).astype(np.float32)
    if deaf_transform is not None:
        z0 = deaf_transform(z0)
    if not layout.audio_inputs:
        return z0
    z = np.empty(layout.length, dtype=np.float32)
    audio = audio_z_index(layout)
    keep = np.setdiff1d(np.arange(layout.length), audio)
    z[keep] = z0
    z[audio] = rng.normal(0.0, sigma, audio.size).astype(np.float32)
    return z


def brain_rng(seed: float) -> np.random.Generator:
    """The generator every "draw brains for this seed" path shares.

    One home for the seed mapping, so the cohort brains and the tournament's
    tiles agree about what a seed means. Two copies of this would drift, and the
    symptom - a modality round trip landing somewhere new - is invisible until
    someone goes back and forth looking for the brain they had.
    """
    return np.random.default_rng(int(abs(float(seed)) * 1e9) % (2 ** 32))


def generated_brains(layout: BrainLayout, seed: float, count: int,
                     audio_seed: float | None = None):
    """`count` independent brains of `layout`, deterministic in `seed`.

    What "no rule loaded" means, for EVERY modality, via the registry. The
    DEAF brains are drawn first from one stream, exactly as they always were;
    the audio weights come after, from `audio_seed` (or `seed`) per channel,
    the same on every cohort - so the count moving leaves every cohort's
    creature where it was. See the brain caveats in CLAUDE.md.
    """
    from services.brains.layout_moves import transfer_audio_inputs

    m = get(layout.modality)
    deaf = with_audio_inputs(layout, 0)
    rng = np.random.default_rng(int(abs(float(seed)) * 1e9) % (2 ** 32))
    brains = [np.asarray(m.random(rng, deaf), dtype=np.float32).reshape(-1)
              for _ in range(max(int(count), 1))]
    if not layout.audio_inputs:
        return brains
    aseed = seed if audio_seed is None else audio_seed
    return [transfer_audio_inputs(b, deaf, layout, aseed) for b in brains]


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
