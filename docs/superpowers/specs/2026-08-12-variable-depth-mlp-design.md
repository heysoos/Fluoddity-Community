# Variable-depth MLP

**Date:** 2026-08-12
**Status:** phases 0 and 1 implemented; phase 2 not started.

One decision the measurement overturned: the deep path's width cap is paid by
the depth-1 path too, so `MAX_MLP_WIDTH` had to fall to 8 and a lone layer keeps
a separate cap of 48. See the MLP layer stack caveats in CLAUDE.md, which are
the one home for those numbers.

## Problem

`MLPModality` is one hidden layer. Its width and activation are settings, but
its depth is a constant baked into both `services/brains/mlp.py` and
`shaders/brains/mlp.glsl`. A deeper controller is not reachable from the UI at
all.

This design makes the layer stack editable — add and remove layers, set each
one's width and activation independently — and gives each layer a right-click
menu for rerolling, rescaling and redistributing its weights, in the idiom the
physics panel already uses.

The hard part is not the arithmetic. It is that a brain's *shape* is the key
every downstream system derives from: the search dimension, the checkpoint
signature, the archive directory, the config file's stamp, and the layout a
hover preview borrows. A depth that varies has to travel all of those without
orphaning anything already on disk.

## Verified findings

Gathered while designing this, on 2026-08-12. Each is checkable, and each
shapes a decision below.

**`layout_from_signature` cannot rebuild a non-default MLP activation.** It
builds its key list from `kind == "int"` only (`services/brains/__init__.py`),
but `activation` is `kind == "choice"`. The parser drops the number, rebuilds
with activation 0, and the round-trip check correctly refuses the mismatch:

| signature | `layout_from_signature` |
|---|---|
| `mlp-n16-a0` | `BrainLayout(shape=(16, 0), length=148)` |
| `mlp-n16-a1` | `None` |
| `mlp-n16-a2` | `None` |

`signature()` and `settings_of()` both use the full shape; only the parser
disagrees. Consequence: `_adopt_foreign_entry` takes its `layout is None`
branch, so clicking a sin or gelu MLP entry in the gallery only warns. The
hover paths (`_preview_layout`, `_config_borrow_layout`) return `None` for
them too — and a `mlp-n16-a1` genome is the *same width* as `mlp-n16-a0`, so a
refused borrow is not caught by `apply_rule`'s width check either. There are
`mlp-n16-a1` and `mlp-n16-a2` directories in `archives/debug10-gabor` and
`archives/debug11-mlp`.

**`rule_buffer` is sized for a path nothing calls.** It reserves
`entity_count * BRAIN_LEN * 4` bytes — at the default world size, ~240k
particles, so ~490 MB at a 512-float brain and ~4.9 GB at world size 4.0. The
write is scoped to one particle by `WRITE_RULES_INDEX`, and the `-1` write-all
branch is unreachable: the sole setter, `Sim.request_rule_buffer_update`,
always sets `_pending_entity_id` alongside `_pending_rule_buffer_update`. Every
occurrence of `particle_brains` across `shaders/` is an assignment target or
the declaration — nothing reads it. The per-particle sizing is correct history,
not a mistake: the older version wrote every particle every frame, and the
single-index optimisation shrank the work without shrinking the allocation.

**Editing a loaded MLP rule translates the whole cohort cloud rigidly.**
`brain_jit(i)` depends on `(rule_seed + cohort, float index)` and nothing else,
and `mlp_param_at` is plain `brain_add`. So every cohort keeps its exact offset
when slot 0 moves. Fourier is the exception — `fourier_mut_seed` is hashed from
the rule's own content, so editing a Fourier rule reshuffles its cohort
offsets. This is why the layer operations below are coherent for MLP.

**The repo's shipped presets carry no `brain_layout` at all.** Every file under
`physics_configs/` predates modalities and is Fourier by history. The only
legacy MLP data is user data: the archive directories above, and 8 user configs
stamped `mlp-n16-a0`.

## Decisions

Taken with the user during design, recorded so the reasoning survives.

1. **Backward-compatible signature, no migration.** A depth-1 net keeps
   emitting `mlp-n16-a0` verbatim; deeper stacks grow parts. Nothing on disk is
   renamed or rewritten.
2. **`rule_buffer` is shrunk first**, as its own commit, and `MAX_BRAIN_FLOATS`
   then rises 512 → 1024.
3. **The caps are measured, not guessed.** Build at a generous compile-time
   cap, price it, then set the UI limits from the result.
4. **The Inspector draws the final hidden layer's units**, which are the only
   ones that decompose additively into the output.
5. **The Inspector gains an RGB random projection**, and it becomes the default
   output view.
6. **The layer editor is per-row `×` with `+ Add layer` at the bottom.**
7. **A Source selector** names which brain the Inspector draws and the layer
   operations edit.
8. **The Scale operation is unclamped and warns at the rail.**
9. **`decode` and `encode` stay flat; fan-in normalisation is initialisation
   only.** A gain folded into `decode` would be a cap on hidden→hidden weights,
   not a normalisation, and would put a whole region out of reach of live
   editing.
10. **Two phases.** The layer stack ships first; the per-layer operations menu
    and the Source selector follow. See "Phasing".

## Scope

In scope: the layer stack reaching the full stack — layout, signature, config
files, archive, optimizer, shader, Inspector, hover-borrow — plus the per-layer
operations menu, the Source selector, and a measurement tool that sets the
caps.

### Non-goals

- Changing Fourier, Gabor or Lenia. Each keeps its current signature, settings
  and shader byte for byte.
- Migrating anything on disk. See "Compatibility invariant".
- Making `W_SCALE` adjustable. See "Follow-ups".
- Fixing evaluation noise in the objective. Separate work.

## Phasing

Two phases, each shippable and verifiable alone. The split is along a real
seam: phase 1 changes what a brain *is*, phase 2 changes what you can *do* to
one. Phase 2 touches no layout, no signature, no archive and no optimizer.

**Phase 0 — the float budget.** Section 0. Its own commit, verified alone.

**Phase 1 — the layer stack.** Sections 1, 2, 3, 4 and 7, plus the RGB
projection from section 6 and the measurement tool. At the end of it the stack
is editable, every existing preset and archive still loads, sin and gelu
entries can be rebuilt for the first time, and the Inspector shows the whole
4-D response of a deep net. Tests 1–7, 9 and 10.

**Phase 2 — editing a layer in place.** Section 5 in full, the Source selector
from section 4, and `PREVIEW_BASE` from section 6. Test 8.

Phase 2 depends on phase 1 only for the layer rows it hangs a context menu off.
Nothing in phase 1 depends on phase 2 — the Inspector keeps drawing slot 0 with
its existing labels until the Source selector arrives, exactly as it does
today.

## Compatibility invariant

**A depth-1 net is bit-identical to today's in every respect**: same parameter
order, same length `9h + 4`, same decode, same initial draw for a given seed,
same signature string, same
shader path. Every existing genome, archive directory, config and checkpoint
therefore keeps working by construction, and there is nothing to migrate.

This is not a convenience. It is the property that makes the rest of the design
safe to reason about, and it is the first thing the test suite asserts.

## Design

### 0. Prerequisite: the float budget

Shipped as its own commit, verified alone, before any brain work.

`rule_buffer` shrinks to one brain. The shader writes at offset 0 rather than
`index * BRAIN_LEN`; `readback_rule` reads at offset 0; the unreachable `-1`
write-all branch is removed. Two tests pin the current contract and are updated
to assert the new one: `tests/test_brain_switch_gpu.py` (buffer size) and
`tests/test_brain_shader_source.py` (the `WRITE_RULES_INDEX < 0` branch).

Then `MAX_BRAIN_FLOATS` 512 → 1024, mirrored in `shaders/brains/_header.glsl`.
The flat brain buffer grows to 832 KB; nothing else scales with it.

Parameter count for hidden widths `w1..wk` with 4 inputs and 4 outputs:

```
n = (4·w1 + w1) + Σ(l=2..k) (w_{l-1}·w_l + w_l) + (4·w_k + 4)
```

At depth 1 that is `9h + 4`. Reachable stacks inside 1024:

| stack | floats |
|---|---|
| `[16]` | 148 |
| `[48]` | 436 |
| `[24, 24]` | 820 |
| `[16, 16, 16]` | 692 |
| `[16, 16, 16, 16]` | 964 |
| `[8] × 8` | 580 |
| `[32, 32]` | 1348 — does not fit |

This table is the float budget alone. The reachable set is further narrowed by
`MAX_MLP_WIDTH` and `MAX_MLP_DEPTH`, which are measured — see "Measurement".

### 1. Data model

`shape` stays a flat `tuple[int, ...]`, interleaved:

```
shape = (w1, a1, w2, a2, ..., wk, ak)      depth = len(shape) // 2
```

At depth 1 this is `(16, 0)` — exactly what MLP produces today, which is what
makes the compatibility invariant free rather than special-cased. `BrainLayout`
stays a frozen dataclass that hashes and compares as it does now, so
`same_space_as`, the checkpoint guard and `_borrow`'s `layout != current`
comparison are untouched.

**Signature format:**

```
mlp-n<w1>[.<w2>...]-a<a1>[.<a2>...]

  depth 1:  mlp-n16-a0                    (identical to today)
  depth 3:  mlp-n16.16.8-a0.1.2
```

Dots are legal in directory names on both platforms, the string always ends in
a digit so there is no trailing-dot hazard, and at the maximum stack it is well
under any path limit.

**Four optional hooks on the modality protocol**, each defaulting to today's
behaviour so the other three modalities are unchanged:

| hook | default | MLP override |
|---|---|---|
| `signature_of(layout)` | `zip("nabc", shape)` | the dotted form above |
| `settings_from_signature(sig)` | ints in schema order | parses the dotted lists |
| `settings_of(layout)` | positional structural map | `{"layers": [[16, 0], ...]}` |
| `unit_count(layout)` | `shape[0]` | `shape[-2]`, the final hidden width |

`layout_from_signature` keeps its parse → rebuild → **compare** verification
exactly as it is. That is what makes a modality-owned parser safe: a parser
that mis-reads returns `None` rather than a plausible layout of the wrong
width. It also fixes the `a1`/`a2` bug, because MLP's parser reads the
activation its own formatter wrote.

**Settings carry `layers` as a list of `[width, activation]` pairs.** Pairs,
not two parallel lists: two lists can desync, and then one edit means two
different stacks. It serialises straight into `brain_settings` JSON.
`layout_from_settings` reads `layers` when present and falls back to
`hidden` / `activation` when absent, so the 8 existing `mlp-n16-a0` configs load
as a one-layer stack with no migration step. `hidden` and `activation` leave
`settings_schema()` — replaced by one `Setting(kind="layers")` — but remain
accepted as input forever.

`ui/brain_window._COUNT_KEYS` stops being hand-written and derives from the
schema. A hand-written list is what let two declared-but-never-read settings
through before.

### 2. Packing and initialisation

Packing generalises today's order without changing it:

```
for l in 1..k:
    W_l   :  w_l rows of fan_in_l      unit-major, each unit's inputs contiguous
    b_l   :  w_l
W_out     :  4 rows of w_k             OUTPUT-major
b_out     :  4
```

At depth 1 this is W1 (h rows of 4), b1 (h), W2 (4 rows of h), b2 (4) —
identical. `W_out` stays output-major for the reason it already is: it lets a
final-layer unit's four output weights be read at stride, which is what keeps
`mlp_unit` and the Inspector working without holding the hidden vector.

This offset formula now lives in both `mlp.py` and `mlp.glsl` — the
`tile_lo_texel()` situation. The guard is a GPU-versus-NumPy parity test, not a
second reading of the same comment.

**`decode` and `encode` stay flat.** Every weight is `W_SCALE · tanh(z)` and
every bias `B_SCALE · tanh(z)`, in every layer, exactly as today. The full
±`W_SCALE` range is therefore reachable in every layer, by the search and by
hand alike, and the rail means one thing everywhere rather than a different
thing per layer.

The obvious alternative — folding a fan-in gain into `decode` — was rejected.
It is not a normalisation but a *cap*: it would put hidden→hidden weights out
of reach at `W_SCALE · sqrt(4 / 16) = 1.0` and make the parameterisation
layer-dependent, which is the fence this design elsewhere refuses to build.

**Fan-in normalisation belongs to initialisation.** A hidden→hidden layer sums
`w_{l-1}` terms rather than 4, so a fresh deep brain drawn at today's sigma has
pre-activations growing like `sqrt(w_{l-1})` — past the width of `tanh`'s
useful range by width 16 — and would be born as a sign function with a flat
landscape around it. `random()` therefore draws `z` in one call as it does now
and then multiplies **only the hidden→hidden slices** by `sqrt(4 / fan_in)`.

Two consequences worth stating. Depth 1 has no hidden→hidden slice, so its draw
is untouched and bit-identical — the compatibility invariant needs no carve-out
in the parameterisation. And this is a starting point, not a limit: the search,
a reroll at any distribution, and the Scale operation can all take a layer
anywhere inside the rails afterwards.

The cost is conditioning. With a flat decode, a step in `z` moves a wide
layer's contribution more than a narrow one's. CMA-ES adapts its covariance,
which is precisely what that adaptation is for, so this is accepted rather than
designed around.

### 3. Shader

`BRAIN_SHAPE` is an `ivec4` and cannot carry a variable stack. Two new
uniforms:

```glsl
uniform int BRAIN_DEPTH;
uniform int BRAIN_LAYERS[2 * MAX_MLP_DEPTH];   // layout.shape verbatim, zero-padded
```

`BRAIN_LAYERS` is `shape` itself rather than split into widths and activations,
so there is exactly one encoding and the GPU never re-derives what the host
already knows — the failure `ACTIVE_COUNT` and `MultiLoadConfig` both taught.
Accessors `mlp_width(l)` and `mlp_act_of(l)` keep the loop readable.

Both are set by **one shared host helper**, `set_brain_layout_uniforms(program,
layout)` in `utilities/gl_helpers.py`, called by `sim.py` and
`services/brain_preview.py`. It owns the layout's structure only; `BRAIN_LEN`
and `BRAIN_PER_COHORT` stay with `sim.py`, because the preview deliberately
omits `BRAIN_LEN` and setting it there only prints a warning.

This extends the purity allowlist in `tests/test_brain_shader_source.py` by the
two names. Deliberate, and it preserves the rule's purpose: both uniforms are
available in the fragment pass, so the Inspector still runs the identical code
the particles run.

**`brain_mlp` keeps today's path verbatim.** `if (BRAIN_DEPTH <= 1)` falls into
the existing stream-one-hidden-unit loop, unchanged. A uniform branch, so no
divergence, and depth 1 cannot regress. The deep path ping-pongs
`float[MAX_MLP_WIDTH]` locals — unavoidable, since any depth above 1 must
materialise a layer's activations, and it is exactly the cost the measurement
tool exists to price.

`mlp_param_at` stays `brain_add` for every index, so mutation, crossover and
the click-to-adopt writeback are untouched.

### 4. Brain window

```
Modality  [ mlp                    v ]
Source    [ Loaded rule            v ]         [ Adopt as loaded rule ]
Layers
   L1  [======16======]  [ tanh v ]  [x]        right-click: layer menu
   L2  [======16======]  [ sin  v ]  [x]
   L3  [=======8======]  [ gelu v ]  [x]
   [ + Add layer ]
   692 / 1024 floats  [######----]
   on release: resets the search, switches archive
search dim: 692     archive: mlp-n16.16.8-a0.1.2 (0 entries)
```

The window must make one distinction obvious, because both live on the same
row: **a layer's count, width or activation is a *layout* edit** — it resets the
search and switches archive — **while a layer's weights are a *genome* edit**,
which is free, undoable and touches nothing structural. The row's own controls
carry the warning; the right-click menu carries none.

- Width drags commit **on release**, through the existing `_brain_draft`, keyed
  `(modality, "layers", i)`. Activation combos, `+` and `×` commit immediately —
  one click, one layout change, which is how `activation` already behaves.
- `+` greys out when even a width-1 layer would not fit, and each width's max is
  clamped to the remaining budget given its neighbours. An over-budget layout
  must be unconstructable: `BrainLayout` raises over budget and `layout_for()`
  catches it, so an unclamped control would make the window silently snap back
  to defaults.
- Rows use hidden `##` labels with a text prefix, and `layout.wrap_row()` for
  the `×`, per the label-clipping and duplicate-ID rules.

**Source selector — phase 2.** It names which brain the Inspector draws and
which the layer operations edit, which is what dissolves the ambiguity of
editing "the" brain when 64 cohorts each have their own. Until it lands, the
Inspector keeps drawing slot 0 and labelling what that means, as it does today.
Which entries it offers is decided by the sim's state, not by the user, because
only one of them exists at a time:

| sim state | Source offers | editable |
|---|---|---|
| rule loaded | `Loaded rule` only — every particle reads slot 0, so no cohort has a brain of its own | yes |
| no rule loaded | `Cohort 0 … N-1` — the cohort slots each hold an independent generated brain | yes |
| tournament | `Tile 0 … N-1` | no, read-only |

**Adopt as loaded rule** promotes the named brain into slot 0 — the same
collapse click-to-adopt performs, as an explicit button rather than a side
effect.

Editing a cohort slot is visible immediately but is not saved by File > Save,
which writes slot 0. That is what the Adopt button is for, and the UI says so.

New `BrainState` fields: `source_kind`, `source_index`, `adopt_requested`
(one-shot), `layer_op` (one-shot), `layer_dist` (per-layer preference), and
`borrow_active` (pushed in by the orchestrator each frame, like
`preview_tile0`).

### 5. Layer operations — phase 2

Right-clicking a row opens a `begin_popup_context_item` menu in the physics
panel's idiom, including its distance-based auto-close from
`preferences.menu_close_threshold`:

```
Layer 2 - 16 units, 272 floats
  Distribution  [ normal  v ]
  Scale         [====1.00====]
  ------------------------------
  [ Reroll weights ]
  [ Reroll biases  ]
  ------------------------------
  [ Reset layer to loaded value ]
```

- **Scale is relative to a snapshot taken when the popup opens**, and pushed to
  rule history once on close. Applied per frame it would compound over a drag
  and the layer would explode — the same shape as the draft/commit the width
  slider needs.
- **Reroll draws `z` from the layer's distribution and then decodes**, so a
  reroll is always representable. `normal` is `N(0, 0.5)`, today's `random()`.
  `uniform` is `U(-1, 1)`. `sparse` is `N(0, 1)` with 80% of entries set to
  zero — `z = 0` decodes to exactly 0, so sparsity in `z` is sparsity in the
  weights. `heavy-tail` is `Cauchy(0, 0.25)`, which the decode bounds.
- **Reset restores the layer's slice of the current `RuleManager` entry** — the
  rule as it was applied, before any operations in this session. When Source is
  a cohort, it restores that cohort's generated brain instead.
- **Every op goes through `apply_rule` and pushes onto `RuleManager`**, so `Z`
  undoes it like anything else.
- **Ops are refused while a hover borrow is held**, because slot 0 then holds
  someone else's brain and the commit path would keep the edit. They are also
  greyed in tournament mode, where slot 0 is tile 0 of a running grid.

The distribution is a per-layer UI preference in `BrainState`, not part of the
genome, the layout or the signature. The weights are what a preset saves, so
there is no round trip to break.

**Scale is unclamped, and warns at the rail.** The archive stores decoded
params, so a rescaled layer saves and plays back exactly as it looks. What
breaks is only *seeding a search from that creature*: `imgep_driver.py` and
`command_handler.py` call `encode(archive.brain_at(i))` to start CMA-ES from an
entry, and `encode` is `arctanh(params / W_SCALE)`, which clips. So an
expedition seeded from a past-the-rail creature starts from a clipped version
of it. The row shows an inline warning naming that, in the same voice as the
existing "the search is losing dimensions".

The warning is an instrument, not a fence. See "Follow-ups".

### 6. Inspector

- **Unit tiles are the final hidden layer's units**, unchanged in meaning:
  `activation × output column`, an additive vec4 contribution, so every output
  channel stays valid and depth 1 renders exactly as it does now. Earlier layers
  are not drawn — a unit in layer 1 of a deep net reaches the output through
  further nonlinearities and has no additive contribution to show.
- **New `random projection (RGB)` channel**, appended to `CHANNELS` and made the
  default. Three orthonormal directions from one seed — QR of a 4×3 Gaussian,
  the same construction `basis_for` already uses — into R, G and B, with
  mid-grey at zero so sign still reads. The monochrome random projection stays;
  it is the honest single-direction view the existing docs describe. The fixed
  slices stay scalar, where RGB has no meaning.
  Trap: `CHANNEL_RANDOM = len(CHANNELS) - 1` today. Both indices must be named
  explicitly rather than derived from the length.
- **New `PREVIEW_BASE` uniform — phase 2**, replacing the hardcoded
  `eval_brain(0u, x)`, so the Inspector can draw whichever slot `Source` names.

Per-unit tiles re-run the forward pass, so the Inspector's redraw cost grows
with depth. The measurement tool reports it.

### 7. Integration surfaces

Each falls out of the data model rather than needing its own handling. Listed
because these are the systems that have broken before.

| surface | what happens |
|---|---|
| Config save/load | `brain_settings` gains `layers`; the signature is stamped from `sim.brain_layout` as now. Files without `layers` load unchanged. |
| Archive | New signature directories appear naturally. `brain_at`'s padding to the widest layout already copes with differing widths. |
| Optimizer | Dimension is `layout.length`, as always. Any layer edit resets the search and switches archive exactly as `hidden` does today. `same_space_as` untouched. |
| Hover-borrow | `_config_borrow_layout` and `_preview_layout` work through the new parser — and now succeed for sin/gelu entries, which they do not today. |
| Adopt | `_adopt_foreign_entry` uses `settings_of`, which now returns `layers`, so the Brain window shows the adopted stack. |
| Multi-load / tournament | Slots are `MAX_BRAIN_FLOATS` wide whatever the layout; only the constant changed. |

## Measurement

`tools/measure_brain_depth.py`. Reports ms per sim step for a set of stacks,
plus the Inspector's redraw cost, and its output sets `MAX_MLP_WIDTH`,
`MAX_MLP_DEPTH` and the UI limits.

Two traps it must respect, both already paid for elsewhere in this codebase:
the sim does not reproduce itself run to run, so every timed block restores
from **one** entity-buffer snapshot taken before the first of them; and two
runs in separate processes measure the laptop's thermal state rather than the
shader, so it stays in one process.

Build at `MAX_MLP_WIDTH = 32`, `MAX_MLP_DEPTH = 8` and narrow from the result.
The resulting numbers get one home, in the CLAUDE.md caveat for this feature.

## Tests

In priority order.

1. **Depth-1 is bit-identical.** Signature `mlp-n16-a0`, length 148, a golden
   parameter vector from a fixed seed, and a GPU output comparison over a fixed
   input grid using the `purefn` pattern — one invocation per point, no shared
   writes. A trajectory diff cannot validate a shader change here.
2. **Signature round-trip for every stack**, including the `mlp-n16-a1` and
   `-a2` regression, and rejection of malformed forms.
3. **GPU forward versus NumPy forward** on known nets at depths 1, 2 and 3 with
   ragged widths. This is the real guard on the packing formula living in two
   files.
4. **Config round-trip** — a deep-MLP config restores its stack and settings;
   an `mlp-n16-a0` file with no `layers` key still loads.
5. **Cross-brain hover, borrow and adopt** of a deep entry, extending
   `test_foreign_preview` and `test_menu_cross_brain_load` — the latter must run
   `_handle_archive_preview` between the hover and the click, because testing
   the menu alone missed this class of bug before.
6. **Budget unconstructability** — no sequence of UI actions produces a layout
   over `MAX_BRAIN_FLOATS`, and `+` is disabled at the limit.
7. **Label widths and ID clashes** for the new rows, extending
   `test_label_widths` and `test_brain_window_render`'s id-clash check.
8. **Layer ops** push to `RuleManager`, are refused during a borrow, and are
   greyed in tournament mode.
9. **encode/decode round trip** for multi-layer stacks — flat and symmetric, so
   every layer round-trips over the same range — plus that `random()`'s fan-in
   normalisation touches only the hidden→hidden slices and leaves a depth-1
   draw bit-identical.
10. `rule_buffer` step-0 tests, updated to the new contract.

## Follow-ups

Named, not implemented.

- **`W_SCALE` as a decode scale.** If interesting creatures turn out to live
  past ±`W_SCALE`, the conclusion is that `W_SCALE` is wrong, not that the
  region should be fenced off. The remedy is to promote it from a module
  constant to a `Setting(kind="float")`, which makes it a decode scale:
  excluded from the signature, `compare=False`, so raising it neither splits the
  archive nor resets a search, and existing entries are unaffected because the
  archive stores decoded brains. Not done now because there is no evidence yet
  that the region is interesting — the rail warning is what collects it.
- **`RuleManager.push_zero_rule` hardcodes `np.zeros((10, 8))`**, a Fourier
  shape. It currently works by accident under other modalities: `apply_rule`
  refuses it on width and substitutes a generated brain. Worth making explicit.
- **A cosmetic signature renamer.** Not needed — nothing is orphaned — but if
  the mixed `mlp-n16-a0` / `mlp-n16.8-a0.1` naming ever bothers, a `tools/`
  script could normalise archive directory names and the per-entry `layout`
  strings inside `index.jsonl`.

## Open items

None blocking. `MAX_MLP_WIDTH` and `MAX_MLP_DEPTH` were provisional at 32 and 8
until the measurement landed; it settled them at 8 and 8, with a separate
depth-1 cap of 48, for the reason recorded in CLAUDE.md.

Worth naming for later: a wider deep stack is only reachable by compiling the
deep path as a shader VARIANT, so a depth-1 brain never carries its locals. That
restructures how `sim.py` builds its program, which is user-owned, and there is
no evidence yet that stacks wider than 8 are interesting.
