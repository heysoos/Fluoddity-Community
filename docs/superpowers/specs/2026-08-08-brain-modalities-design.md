# Swappable brain modalities

**Date:** 2026-08-08
**Status:** design approved, not yet implemented

## Problem

The brain is one fixed function. `fourier_noise()` maps four egocentric sensor
readings to a force and a strafe vector, and its 80 parameters — 10 centers of
4 frequencies and 4 amplitudes — are the only genome the search stack knows how
to handle. Exploring a different *kind* of controller currently means editing
the shader and breaking every archive.

This design makes the brain a swappable component with a per-modality parameter
layout, selectable from the UI, searchable by the existing optimizers, and
inspectable as a rendered response field.

## Motivating measurements

Taken from `runs/` on 2026-08-08 over five `driver=prompt` runs. They shape
several decisions below and are recorded so the reasoning survives.

**Frequencies drift to roughly twice the hand-tuned prior.** `random_genome()`
biases frequencies low on purpose ("Bias towards lower frequencies for smoother
base behaviors") and produces mean `|freq| = 0.832`. `genome_spec.decode()`
applies a flat `3*tanh(z)` with no such bias. Measured mean `|freq|` of the
best genome, first checkpoint to last:

| run | prompt | start | end |
|---|---|---|---|
| 205941 | a smiley face | 0.89 | 1.90 |
| 000739 | a network of neurons | 0.74 | 1.44 |
| 025441 | a smiley face | 1.07 | 1.68 |
| 030020 | purple | 1.01 | 1.71 |
| 030656 | purple | 1.00 | 1.46 |

**Saturation is real but mild.** Hard saturation (decoded value within 1% of its
rail) stays under 10%. Mean `sech²(z)` — the phenotype sensitivity — falls from
~0.88 to 0.51–0.68, so the effective step in behaviour space drops to 0.46–0.84x
of its gen-10 value while the reported sigma barely moves. A drag, not a wall.

**The dominant problem is evaluation noise, not the parameterisation.** On the
`purple` runs the generation-to-generation swing in `fit_best` (0.11–0.12)
exceeds the entire within-generation spread across 16 tiles (0.059–0.071). A
rank-based optimizer given a ranking that unstable is ranking noise. Both runs
peak near 0.7 and end below 0.37.

**Therefore: new modalities are not expected to fix CMA-ES.** They are worth
building for expressive range. The optimizer's problem lives in the objective,
and is out of scope here. See "Non-goals".

## Scope

In scope: modality selection reaching the full stack — shader, UI, genome
encoding, optimizers, checkpoints, and the novelty archive. Plus a Brain
Inspector that renders each unit's response.

### Non-goals

- Fixing evaluation noise in the tournament objective. Separate work.
- Changing the sensor model, the symmetrisation, or the integrator.
- Migrating existing archives. They remain valid as the Fourier archive.

## The modality contract

A modality is two files plus one registry entry. Nothing else in the codebase
learns its name.

### GPU side — `shaders/brains/<name>.glsl`

One pure function per modality:

```glsl
vec4 brain_gabor(uint base, vec4 x);
```

`base` is the index of this brain's first float in the flat parameter array,
i.e. `slot * MAX_BRAIN_FLOATS` where slot is 0 in manual mode and the tile index
in tournament mode. `x` is `vec4(L.axial, L.lateral, R.axial, R.lateral)`. The
return is `vec4(force.axial, force.lateral, strafe.axial, strafe.lateral)`.

The function must depend on nothing but its parameters and `x` — no entity
state, no trail texture, no globals beyond `BRAIN_SHAPE`. This is what lets the
same function be called from the inspector's preview pass (see below), and it
is a hard requirement, not a stylistic preference.

### CPU side — `services/brains/<name>.py`

```python
class BrainModality(Protocol):
    name: str                  # "gabor" — UI label and archive signature
    modality_id: int           # 1 — the GLSL branch value
    glsl_file: str             # "shaders/brains/gabor.glsl"

    def settings_schema(self) -> list[Setting]: ...
    def layout_from_settings(self, s: dict) -> BrainLayout: ...
    def decode(self, z, layout) -> np.ndarray: ...
    def encode(self, params, layout) -> tuple[np.ndarray, int]: ...
    def random(self, rng, layout) -> np.ndarray: ...
```

`decode`/`encode` replace the module-level functions in `genome_spec.py`, which
become thin delegations to the active modality. `random()` gives each modality a
CPU-side prior; today `services/genome.py:random_genome` mirrors
`generate_random_centers()` in GLSL by hand, and that duplication moves into one
place per modality.

Moving `decode()` into the modality is also the fix for the frequency-prior
mismatch measured above: the squash stops being a global constant and becomes a
per-modality decision, so Fourier's can carry the low-frequency bias its own
random generator already uses.

### BrainLayout

```python
@dataclass(frozen=True)
class BrainLayout:
    modality: str            # "gabor"
    shape: tuple[int, ...]   # structural ints, e.g. (12,) filters
    length: int              # active float count == the search dimension

    def signature(self) -> str:   # "gabor-n12"
```

`signature()` is the key everything downstream derives from.

## The four modalities

### Fourier (existing, `modality_id = 0`)

```
out = Σᵢ ampᵢ ⊙ (sin(φᵢ + oᵢ), cos(φᵢ + 0.7oᵢ), sin(2φᵢ + 1.3oᵢ), cos(2φᵢ + 0.5oᵢ))
φᵢ = dot(x, freqᵢ)     oᵢ = i·(2π/5) + ampᵢ.w·π
```

Global and periodic — every reading excites every unit in all directions, with
no "off" state. 8 floats per center. Behaviour is preserved bit-for-bit at the
default layout so existing genomes and archives keep working.

Settings: Centers (4–48), Freq Scale, Low-Freq Bias.

### Gabor (`modality_id = 1`)

A Gaussian envelope around the oscillation:

```
gᵢ(x) = exp(−‖x − cᵢ‖² / 2σᵢ²) · cos(dot(x, fᵢ) + φᵢ)
out   = Σᵢ aᵢ · gᵢ(x)
```

The envelope makes the unit *local*: it responds near its center and is silent
elsewhere, so a brain becomes a set of "when I see roughly this, do that" rules
rather than one global interference pattern. This also damps the drift measured
above — a high frequency inside a narrow envelope produces locally intricate
response, not global chaos.

14 floats per filter: center (4), frequency (4), amplitude (4), width, phase.

Settings: Filters (4–36), Freq Scale, Envelope Width, Phase Spread.

### Lenia bumps (`modality_id = 2`)

Lenia's growth function — a Gaussian band, not an oscillation:

```
uᵢ    = dot(x, wᵢ)
Gᵢ    = 2·exp(−(uᵢ − μᵢ)² / 2σᵢ²) − 1
out   = Σᵢ aᵢ · Gᵢ
```

The `−1` is essential: response is positive inside a narrow band of sensor
values and negative everywhere else, which is the "thrive at this density, die
away from it" rule that gives Lenia its membranes. Non-oscillatory and smooth;
expected to give the best-conditioned landscape of the four.

10 floats per bump: projection (4), amplitude (4), band center μ, band width σ.

Settings: Bumps (4–48), Band Center μ, Band Width σ.

Note on priors: canonical Lenia parameters (orbium and similar) are tuned for a
continuous CA whose input is a kernel-weighted neighbourhood sum over a grid.
Here the input is two sensor taps scaled by `√ws · 38.855 · SENSOR_GAIN`. The
numbers do not transfer and must not be copied in. What does transfer is
scale-free: Lenia growth bands are consistently narrow, roughly σ/μ ≈ 0.1. The
σ slider's default range encodes that ratio. No hand-tuned `random()` prior for
Gabor or Lenia in this iteration.

### MLP (`modality_id = 3`)

```
out = W₂ · act(W₁x + b₁) + b₂
```

`act` selectable: tanh (smoothest), sin (a SIREN, closest in character to
Fourier), GELU. `9H + 4` floats at hidden width H.

Settings: Hidden Width (4–48), Activation, Layers (1–2).

### Parameter budget

`MAX_BRAIN_FLOATS = 512`. Every UI range above fits at its maximum: Fourier
48×8 = 384, Gabor 36×14 = 504, Lenia 48×10 = 480, MLP at H=48 = 9(48)+4 = 436.

## Data flow

### Dispatch

A uniform branch inside one compiled program:

```glsl
vec4 black_box(vec2 L, vec2 R) {
    vec4 x = vec4(L, R);
    if (BRAIN_MODALITY == 0) return brain_fourier(base, x);
    if (BRAIN_MODALITY == 1) return brain_gabor(base, x);
    if (BRAIN_MODALITY == 2) return brain_lenia(base, x);
    return brain_mlp(base, x);
}
```

The branch is uniform across the whole dispatch — every particle takes the same
path — so there is no warp divergence, and switching modality is one uniform
write with no recompile. Rejected alternatives: one program per modality (would
require re-pushing ~40 uniforms and re-binding four SSBOs on every switch, and
restructuring `sim.py`, which is user-owned); a separate compute pass (a
permanent per-frame cost for isolation not needed here).

Modality GLSL files are spliced in by the existing `shader_prepend` mechanism
(`sim.py:140`), the same way `fourier4_4.glsl` is today.

### Buffers

The current `target_rule` uniform and `target_rules[64]` SSBO are two paths for
the same data. They collapse into one:

```glsl
#define MAX_BRAIN_FLOATS 512
layout(std430, binding = 4) buffer BrainBuffer { float brain_params[]; };
uniform int   BRAIN_MODALITY;
uniform int   BRAIN_LEN;      // active floats per brain; bounds every loop
uniform ivec4 BRAIN_SHAPE;    // structural ints (n_centers, hidden width, ...)
```

Manual mode writes slot 0; tournament writes slots 0..15. A flat `float` array
in std430 has 4-byte stride and no padding, so packing is
`params.astype(np.float32).tobytes()` with no alignment to get wrong. This
removes `set_rule_uniform()` (`utilities/gl_helpers.py:67`) and its 20
hand-written uniform assignments.

**The per-particle buffer must be sized to `BRAIN_LEN`, never to
`MAX_BRAIN_FLOATS`.** The `rules[]` SSBO exists so clicking a particle adopts
its mutated brain. At 320 bytes per particle it is already 192 MB; widening it
to 512 floats would make it ~1 KB per particle, about 600 MB. It is reallocated
on layout change — which already resets the optimizer and switches archive, so
this adds nothing to that path.

### Downstream keys

| consumer | today | after |
|---|---|---|
| archive dir | `archive/` | `archive/<signature>/` |
| `Archive._brain` | `(0, 10, 8)` | `(0, layout.length)` flat |
| `GenomeSpec` | `Block("brain", 80)` | `Block("brain", layout.length)` |
| checkpoint signature | `"brain:80"` | `"gabor-n12:168"` |
| `GAOptimizer` | reshapes to `(-1, 8)` | reshapes per layout |

An archive is keyed by `(modality, layout)` together, so `gabor-n12` and
`gabor-n8` are separate archives. The alternative — padding every entry to 512
floats with a stored length — would allow loading a genome whose parameters mean
something different from when it was saved. The split is deliberate and explicit.

Existing `archive/` contents become the Fourier archive at its default layout:
on first run the directory is renamed to `archive/fourier-n10/`, matching the
current 10-center genome. Entry contents are untouched — only the path changes,
and the stored `(N, 10, 8)` brains are read as flat 80-float vectors.

## Brain Inspector

A UI panel rendering each unit's response field, updating live.

It renders **through the same GLSL the particles use**. Because a brain function
is pure, a preview fragment shader can include the same `shaders/brains/*.glsl`
and bind the same parameter buffer. What is shown is what the particles compute,
and cannot drift out of sync — this is the reason for the purity requirement in
the contract.

**Layout.** A grid of thumbnails, one per unit (center / filter / bump / hidden
neuron), each showing that unit's response over a 2D slice of the 4D sensor
space as a heatmap. One 512×512 offscreen texture holds them as an atlas, filled
in a single draw; ImGui blits sub-rects. 48 units costs one draw call.

**Controls:**

- **Slice** — which two of the four inputs to sweep. Default `L.axial × R.axial`;
  the other two held at zero.
- **Output channel** — force-axial, force-lateral, strafe-axial, strafe-lateral,
  or all four as RGBA.
- **Range** — the sensor magnitude the slice spans. Defaults to the actual
  scaled range so the view covers the region particles really visit.
- **Mode** — 2D heatmap or 1D curve. 2D is the default for Fourier and Gabor,
  which *are* 2D images in the usual depiction. 1D is the default for Lenia,
  because `G(u)` against `u` is how a growth function is drawn and a heatmap
  would be a worse picture of the same thing. Both available for all four.

**Aggregate tile.** The summed output of the whole brain — the actual force
field as a function of sensor reading. This is the view that shows whether a
creature will be smooth or frantic.

**Live updates** on modality switch, seed reroll, parameter edit, and — during a
tournament — following the current best tile, so the response field can be
watched deforming as evolution runs.

## UI

A **Brain** section, following the existing `ui/` mixin pattern
(`ui/brain_window.py`, added to the `UI` class in `ui/core.py`).

- Modality dropdown.
- Settings rendered from the active modality's `settings_schema()`.
- Status line: `search dim: 168 · archive: gabor-n12 (1,204 entries)`.
- **Saturation meter**: the fraction of evolved parameters within 1% of their
  rails. This exists because the drift measured above was invisible until it was
  dug out of checkpoint files by hand. It belongs on screen.
- Any change to a parameter *count* warns before applying — it resets the
  optimizer and switches archive.

Naming follows the existing conventions: `ALL_CAPS_UNDERSCORE` for uniforms,
Title Case for UI labels, `_snake_case` for private UI state.

## Error handling

- **Unknown modality id in a saved config** — fall back to Fourier at its
  default layout, print a warning, keep running. Never crash on load.
- **Checkpoint signature mismatch** — already raises `CheckpointError`
  (`services/run_checkpoint.py:100`). Loading a Fourier checkpoint into a Gabor
  session fails loudly for free.
- **Layout change mid-tournament** — end the generation, reset the optimizer,
  switch archive, reallocate the per-particle buffer. Never partially applied.
- **`length > MAX_BRAIN_FLOATS`** — rejected by `layout_from_settings`, which
  clamps the UI range; the guard is a `ValueError` so a bad programmatic caller
  is not silent.
- **Modality GLSL fails to compile** — the existing hot-reload path prints and
  keeps the previous program. Must not take the app down.

## Testing

No GPU in CI, so tests target the CPU half and the contracts.

- **Round-trip per modality**: `encode(decode(z)) ≈ z` within the unsaturated
  range, for every modality at several layouts. Mirrors
  `tests/test_physics_origin_roundtrip.py`.
- **Layout signature stability**: `signature()` is stable across runs and
  distinct for distinct layouts. A regression here silently merges archives.
- **Length arithmetic**: `layout.length` matches the packed array length for
  every modality at every UI-reachable setting, and never exceeds 512.
- **Registry completeness**: every registered modality has its GLSL file present,
  a unique `modality_id`, and implements the full protocol.
- **Archive width**: an archive created at one layout rejects entries of another.
- **Fourier parity**: at the default layout, the new `decode()` reproduces the
  current `genome_spec.decode()` exactly, so existing genomes are unchanged.
- **Shader source check**: the modality GLSL files declare the expected function
  signature. Extends `tests/test_shader_source.py`.

Manual verification per `docs/testing_checklist.md`: each modality produces
motion, the inspector matches observed behaviour, switching modality mid-run
does not crash, and the saturation meter moves during a tournament.

## Open question deferred

Whether the saturation meter should also *act* — e.g. re-centring the search
when saturation crosses a threshold — is left out. Measure first; the meter is
the instrument that makes that decision possible later.
