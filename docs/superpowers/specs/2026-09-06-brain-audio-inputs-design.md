# Audio inputs to the brain

Date: 2026-09-06

Status: EXPERIMENTAL. Built to be removed without disturbing anything it sits
beside; see "How to remove this" at the end. Mockup:
https://claude.ai/code/artifact/946e8472-21a6-4123-895b-f979a5405dec

## The problem

Every brain, whatever its modality, is a function of four numbers: the axial
and lateral trail velocity at the left and right sensors. Audio reaches the
sim only by turning knobs AROUND that function - a physics slider through
`modulate()`, a per-cohort gain and offset through `cohort_audio()`, a decode
scale through `BrainModulator`. The brain itself never hears anything, so a
rig can make a creature faster or wider on a kick but cannot make it a
different creature on a kick.

## What this adds

1. **K audio channels as brain INPUTS.** The input vector grows from the four
   sensor taps to `4 + K`, K in 0..8, and K is part of the brain layout.
   Each unit of every modality gains K weights, so a brain can evolve - or
   be seeded with - a genuine response to sound.
2. **A channel is a mapping TARGET.** The rows, shapers, modes, cohort masks,
   mutes and strengths a channel carries are the ones the rig already has.
   "A combination of shaped signals" is several rows summing into one target,
   which `modulate()` already computes.
3. **Any preset takes audio inputs, deterministically.** Widen the layout,
   copy the existing weights unchanged, draw the audio weights from a seed.
   Same preset, same seed, same brain.
4. **A Brain Inputs tab** beside the existing rig in the Audio Reactive
   window, holding the layout line, the seed, the channel rows and a readout
   of the input vector the brain actually reads.

Explore and Auto are deliberately deaf: a search runs with every channel at
zero and judges the brain as its original preset. Scoring under a recorded
programme is a later design.

## The invariant: zero in, zero change

**With every channel at exactly 0.0, a K-input brain computes bit for bit
what its K=0 ancestor computed.** Feed off, capture stopped, a search
running, or a rig with no channel rows all produce exactly 0.0, so the
preset is its original self the moment audio is off.

This is a rule about WHERE an audio weight may live: only in a term that is
multiplied by the input. `0.0 * w` is `±0.0`, and `phase + (±0.0)` is `phase`
exactly - and an FMA of `(0.0, w, phase)` is `phase` exactly too - so the sum
is unchanged provided the original four-term dot product is formed FIRST and
the audio terms added after it. Per modality:

| Modality | Audio weights extend | Never |
|---|---|---|
| Fourier | the frequency vector: `phase = dot(x, f) + Σ a_k·fa_k` | the amplitude, the index-derived phase offset |
| Gabor | the frequency only: `cos(dot(x, f) + Σ a_k·fa_k + φ)` | the CENTRE - the envelope measures distance from it, and an audio centre of 0.3 makes silence read as "0.3 away" |
| Lenia | the projection: `u = dot(x, w) + Σ a_k·wa_k - μ` | μ, σ |
| MLP | the first layer's fan-in: `W1` is `w1 × (4+K)` | later layers, biases |

The mirror term needs nothing: `calculate_entity_behavior` swaps L and R and
reflects the lateral component; the audio components ride through both terms
unchanged, so no handedness bias enters.

While capture runs through a QUIET passage, a `sample_hold` or `phase` row
holds its last value - the rig's existing rule for those shapers, and the
reason they exist. Turning Feed or capture off still zeroes them. Nothing
here changes `audio_shapers.py`.

## The layout

`BrainLayout` gains one field, shared by every modality:

```python
audio_inputs: int = 0     # compare=True; part of the signature
```

It is NOT in `shape`, so every modality's positional indexing and every
existing signature parser are untouched. Each modality's
`layout_from_settings` reads `s.get("audio_inputs", 0)` into the field and
into its `length`, which is the only place a modality has to know K to build
a layout. `signature()` appends `+a{K}` only
when K > 0, so a K=0 layout's signature is character-identical to today's:
`fourier-n10` stays `fourier-n10`, and `fourier-n10+a4` is its four-input
sibling. `layout_from_signature` splits the suffix off before handing the rest
to the modality's parser, and re-verifies by rebuilding, as it does now.

`audio_inputs` is an `int` setting, so it is STRUCTURAL: changing it resets
the optimizer and re-points the archive, exactly as a centre count does. It
is declared ONCE, `AUDIO_INPUTS_SETTING` in `services/brains/__init__.py`,
and appended by each modality's `settings_schema()`, so `settings_of` emits
it, the Brain window draws it and the schema-derived tests cover it. It is
EXCLUDED from `_generic_proposals` in `layout_moves.py`: the layout search
must not wander into K on its own, since a search hears nothing.

### Where the floats go

The audio weights sit at the END of each unit, so every existing offset is
unchanged and a K=0 brain is BYTE-identical on disk and in VRAM:

| Modality | Unit today | Unit at K | Type |
|---|---|---|---|
| Fourier | freq(4) amp(4) = 8 | freq(4) amp(4) audio(K) | scale, like freq |
| Gabor | centre(4) freq(4) amp(4) σ φ = 14 | … + audio(K) | scale, like freq |
| Lenia | proj(4) amp(4) μ σ = 10 | … + audio(K) | scale, like proj |
| MLP | `W1` rows of 4 | `W1` rows of 4+K, audio columns last | offset, like W1 |

`unit_floats(layout)` returns the widened stride; `SCALE_OFFSETS` and
`AMPLITUDE_SLICE` keep their values because the audio block comes after them.
MLP's `IN_DIM` becomes `4 + layout.audio_inputs` inside `layer_spans`, which
is the one definition `mlp.glsl` is checked against.

Each modality gains one hook, `audio_weight_index(layout) -> np.ndarray`, the
flat indices of its audio weights. Everything generic - transfer, shrink,
the Inspector's readout - is written against that hook rather than against a
per-modality reading of the table above.

### Decode and encode

Audio weights decode exactly as the modality's own frequency or projection
components do, through the same `tanh` squash and the same scale, times one
new decode SCALE:

```
audio_scale   Setting(kind="float"), default to be measured (see Open questions)
```

so `w_audio = audio_scale * freq_scale * tanh(z)` for Fourier, and the
matching expression for the others. Being a decode scale it is free to move
mid-run, it re-decodes `z` rather than re-rolling anything, it is offered by
`brain_targets` so a rig can modulate it, and at 0 the brain is deaf without
touching a weight. `encode` divides by the same scales and floors
`audio_scale` the way `b_scale` is floored, so the modulator's
encode-once/decode-many holds. Guarded by `tests/test_brain_scales.py`, which
derives its cases from `settings_schema()` and is what proves the scale is
READ.

Fourier's structured mutation scales the whole frequency vector by ONE
scalar; the audio components join that vector and take the same scalar, so
the direction of the widened vector is preserved as it is today. The other
modalities mutate per float through `brain_mul`/`brain_add` as their table
row says.

## Adding inputs to a preset

Two operations in `layout_moves.py`, both pure:

```python
grow_inputs(layout, k)            -> BrainLayout      # k in 1..MAX_AUDIO_INPUTS
shrink_inputs(layout, k)          -> BrainLayout
transfer_audio_inputs(params, parent, child, rng) -> np.ndarray
```

`transfer_audio_inputs` copies every NON-audio float of the parent into its
position in the child, unit by unit, and fills the child's audio indices from
`random(rng, child)`. The generic `transfer_genome` cannot do this: it copies
by CHILD stride, which is wrong the moment the stride changes, and MLP's own
transfer zeroes a widened fan-in's new columns, which is right for a grown
layer and wrong here - these columns are meant to hear. A shrink is the same
copy with no fill, and is lossy: the audio weights are gone, and the stepper
says so.

The rng is `brain_rng(audio_seed)`, the path `generated_brains` already uses.
`audio_seed` is a `SimState` field beside `rule_seed`, saved in
`PhysicsConfig` like it, and undoable. **Reroll** draws a fresh seed and
re-runs the transfer FROM THE CURRENT RULE - which copies its non-audio
floats and redraws the rest - so a reroll never disturbs the deaf brain
underneath.

The saved config holds the widened brain itself, the archive's discipline of
storing decoded floats rather than a recipe, with `brain_layout` naming the
`+aK` signature and `audio_seed` beside it as provenance. A build without
this feature refuses the file on width and says so, exactly as it refuses
any foreign brain today; `_config_signature` reads the suffix and needs no
new special case, since a file naming no brain still carries an 80-float
Fourier rule.

### The apply path

The count stepper - in the Brain Inputs tab AND in the Brain window, both
writing `ui_state.brain.settings["audio_inputs"]`, one field - reaches
`_apply_brain_layout` like any structural change. One addition there: when
the ONLY structural difference between the current layout and the requested
one is `audio_inputs`, and a rule is loaded, the rule is transferred and
pushed rather than dropped. Every other structural change keeps its current
behaviour. The realloc, the retarget and the settings write all still
happen, because the signature moved.

A layout edit has an owner, and the same two states have none: under a
tournament grid slot 0 is tile 0 of a running grid, and during a hover borrow
slot 0 holds someone else's brain. The stepper and Reroll are disabled in
both, with the reason shown, exactly as `_handle_brain_source` refuses a
layer edit.

## The channels

### State

```python
@dataclass
class Channel:
    name: str = ""            # a label for the rig; "" shows as A<k+1>
    range: float = 0.25       # what a full-scale sum reads as, in brain-input units
    uid: int                  # as Mapping.uid, in-session only

AudioInState.channels: list[Channel]            # persisted
AudioInState.channel_mappings: list[Mapping]    # persisted; target = "AUDIO_IN_<k>"
AudioInState.feed: bool = True                  # persisted; the tab's master switch
```

A channel's identity is its INDEX - the weight column it drives - and the
name is a label, saved in the rig, nothing more. The list is sized to the
LAYOUT's K, not the other way round: a rig with four channels loaded under a
K=2 brain shows two rows and keeps the other two waiting, the way
`brain_mappings` waits for its modality.

### Targets and maths

`channel_targets(layout) -> list[TargetDef]` sits beside `physics_targets`
and `brain_targets`, group `"channel"`, key `AUDIO_IN_<k>`, label the
channel's name, `lo = -range`, `hi = +range`, hard-clamped to the same. It is
derived from the layout, so there is no second list of targets.

A channel's value is the existing modulation chain over a base of ZERO:

```
value = strength * range * A * M        clamped to [-range, +range]
A = Σ_add s·depth − Σ_sub s·depth
M = Π_mul (1 + s·depth)
```

with `s` each row's shaped signal. `range` is the span, so a channel is a
SIGNED sum that is 0 when its rows are 0, and a subtract row is how it goes
negative. There is no polarity control and no idle value: idle is 0 by the
invariant. Per cohort, `channel_values(...)` in `cohort_audio.py` mirrors
`build_arrays`: one float per (channel, cohort), masked rows contributing to
their cohorts only, returning `None` when Feed is off, capture is stopped, a
search is running, or no row is enabled - and `None` is what the sim uploads
as inactive.

`range` is in the brain's OWN input units, because the four sensor inputs
vary about 900x across the preset library (`tools/brain_input_scale.py`):
a channel reading 1.0 would swamp a preset whose sensors read 0.002.
**Match sensors** sets `range` to the preset's median `|input|` off that
tool's measurement, and the readout beside the input-vector meters shows the
preset's p50 and p90 so the choice can be made by eye.

### The GPU side

A new SSBO, binding 6, declared in `shaders/brains/_header.glsl` beside the
brain buffers:

```glsl
#define MAX_AUDIO_INPUTS 8                   // mirrored in services/brains
layout(std430, binding = 6) buffer AudioInputBuffer {
    float audio_in[];                        // [k * CA_SLOTS + cohort]
};
uniform int  BRAIN_AUDIO_IN;                 // layout.audio_inputs
uniform bool AUDIO_IN_ACTIVE;
float g_audio[MAX_AUDIO_INPUTS];             // invocation-local, like g_brain_mut
```

`entity_update.glsl` fills `g_audio` once per particle from its cohort - all
zeros when inactive - and every `brain_*()` keeps its `vec4 x` signature,
reading the audio terms from `g_audio` inside a loop bounded by
`BRAIN_AUDIO_IN`. That is the pattern `g_brain_mut` already uses, and it is
what lets the Brain Inspector call the same functions unchanged. The loop
body is `acc += g_audio[k] * unit_audio_weight(k)`, placed AFTER the
`dot(x, f)` so the invariant's evaluation order holds; at K=0 the loop does
not run and the shader is the one that ships today.

MLP is the exception in form, not in rule. `mlp_hidden` seeds its scratch
array with the four taps; it now seeds `cur[4..4+K)` from `g_audio` as well
and the first layer's fan-in loop runs to `4 + BRAIN_AUDIO_IN`, sensor
columns first, so the accumulation order is the invariant's. The scratch
array must therefore hold `4 + K` floats: `scratch_width` takes
`max(want, 4 + K)`, which at K up to 4 stays inside the free `8` bucket and
at K up to 8 lands in `16` - a cost the Brain window names beside the count,
as it names a width bucket today, and one to measure with
`tools/measure_brain_depth.py` before the ceiling is taken as free.

The Inspector evaluates at the LIVE channel values, so its tiles move with
the music - that is the readout - and its header lists them. `MAX_AUDIO_INPUTS`
is mirrored in Python and compared by the shader-source test, as
`COHORT_BRAIN_SLOT0` is.

`sim.py` gains `set_audio_inputs(arr)` beside `set_cohort_audio`, the buffer
reserve, and the per-step `AUDIO_IN_ACTIVE` write; `BRAIN_AUDIO_IN` rides
with the other brain uniforms, which are per-step because
`set_brain_scales` changes the layout without changing the program. `sim.py`
is user-owned: these are additions in the shape of the cohort audio ones,
not a restructuring.

`AudioRuntime.update` computes the channel array after `build_arrays` and
exposes it as `audio_inputs`, cleared before every early return for the same
reason `cohort_audio` is; `main.py` hands it to the sim beside
`set_cohort_audio`.

## The panel

The matrix section of the Audio Reactive window becomes two tabs, **Modulate**
(everything drawn today, unchanged) and **Brain Inputs**:

- **Layout line**: the signature with the `+aK` suffix highlighted, a stepper
  for the count, and the Feed switch. Under it, one line saying the count is
  a layout change and that zero channels is the original preset.
- **Weights line**: the seed, Reroll, and the `audio_scale` slider.
- **What the brain sees**: 4 + K small bipolar meters. The sensor four are
  grey, drawn with the preset's p10..p90 band; the channels are coloured by
  their first mapped signal. This is the instrument for judging `range`.
- **Channel rows**, shaped exactly like a target row: on-switch, `A<k>` and
  the name, sparkline, six signal dots, range readout. A dot toggles a row,
  the name opens the drawer, as today.
- **The drawer**: one tab per mapped signal - the existing shaper editor,
  untouched - plus a **Channel** tab with Name, Range and Match sensors, the
  summed formula in words, and the channel's own trace.
- **+ Add channel** is disabled at K of K, with the stepper named as the way
  to raise it.

Every widget has a `##suffix` where its text repeats a Modulate label
(`test_archive_window_render.py::id_clashes` discipline), the tab's labels
fit `WIDEST_LABEL` (`tests/test_label_widths.py`), and the drawer's combo
entries are opened by the forced-popup helper in the render test, since a
popup body never runs in an ordinary pass. Nothing in the window body touches
GL.

## Persistence and undo

- Rig (`Documents/Fluoddity/audio_rig.json`): `channels`,
  `channel_mappings`, `feed` join `PERSISTED_FIELDS` and `to_dict`/
  `apply_dict`. A malformed channel keeps its defaults; a malformed row is
  dropped, never the rig.
- Config: `audio_seed` joins `PhysicsConfig` beside `rule_seed`;
  `brain_layout` and `brain_settings` carry the rest for free.
- Undo: `audio_seed` is undoable; `channels`, `channel_mappings` and `feed`
  are classified in `tests/test_undo_fields.py` with the rest of the audio
  state, which is not undoable today.
- Archive: a `+aK` signature is its own directory. Nothing migrates.

## Testing

- `tests/test_brain_audio_inputs.py`, cases derived from `REGISTRY`: for
  every modality, K=0 gives the same signature, length, decode and encode as
  before; `grow_inputs`/`shrink_inputs` round-trip the signature through
  `layout_from_signature`; `transfer_audio_inputs` preserves every non-audio
  float and fills every audio index; a NumPy reference of each modality's
  evaluation at K=4 with zero inputs equals its K=0 parent's.
- `tests/test_brain_audio_inputs_gpu.py`, the `purefn` pattern: `eval_brain`
  over a fixed input grid, K=4 brain with `g_audio` zero against its K=0
  parent, `np.array_equal` - bit for bit, since the function is pure - and a
  control proving non-zero `g_audio` changes the output. A Gabor case
  asserts the envelope does not move with audio.
- `tests/test_cohort_audio.py` extension: `channel_values` maths, masks,
  clamp, and `None` for each of the four ways to be inactive.
- `tests/test_audio_rig_io.py` extension: channels and names round-trip; a
  rig with more channels than K keeps the extras.
- `tests/test_brain_layout_apply.py` extension: the audio-only apply path
  transfers and pushes the rule; any other structural change behaves as
  before; the stepper is refused under a grid and during a borrow.
- Render test for the tab: labels drawn, no id clashes, forced popups.
- `tests/test_brain_shader_source.py`: `MAX_AUDIO_INPUTS` mirrored.

## Open questions

- **`audio_scale`'s default.** The draw should land the audio terms at "about
  as strong as a sensor term" for a typical preset, which needs measuring
  across the library the way `w_scale` was, with `tools/brain_input_scale.py`
  as the instrument. Until then the default is 1.0 and the mockup's 0.60 is a
  guess.
- **Match sensors as the default.** Whether a new channel should start at
  the preset's median sensor input rather than a fixed 0.25 needs a listen.
- **Explore under sound.** Out of scope. When it comes, the search needs a
  fixed programme rather than a microphone, or its entries are
  unreproducible.

## How to remove this

At K=0 everything is byte-identical, which is what makes removal safe. Delete
the Brain Inputs tab, `channel_targets`, `channel_values`, the audio-input
buffer and its uniforms, the `audio_inputs` field and `AUDIO_INPUTS_SETTING`,
the three functions in `layout_moves.py`, and `audio_seed`. A config or
archive directory written with a `+aK` signature then refuses to load on
width - the same refusal as any brain this build cannot rebuild - and
everything written at K=0 opens untouched.
