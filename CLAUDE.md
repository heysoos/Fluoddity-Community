# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fluoddity is a GPU-accelerated 2D particle simulation for generative art. Thousands of particles follow neural-net-like "Rules" governing their response to trail density, producing emergent patterns. Physics runs entirely on the GPU via GLSL compute shaders. Built with Python 3.12, ModernGL (OpenGL 4.3), GLFW, imgui_bundle, NumPy, and FFmpeg.

## Commands

```bash
# Run (development)
pip install -r requirements.txt
python main.py

# Tests - the venv interpreter, NOT bare `python` (that is 3.10 with no pytest)
.venv/Scripts/python.exe -m pytest -q

# Build distributable (Windows, PowerShell)
.\build.ps1

# Manual build
python -m PyInstaller --clean --noconfirm Fluoddity.spec
Move-Item -Path "dist\Fluoddity\_internal\shaders" -Destination "dist\Fluoddity\shaders"
```

`tests/` covers the services, the GPU tiling (via a headless context) and UI
render smoke tests. The sim itself is verified by hand — see
`docs/testing_checklist.md`.

## Architecture

**Orchestrator pattern** — the `App` class in `main.py` coordinates all components. Components never talk to each other directly.

Each frame:
1. `ui.get_state()` — UI exposes its state as a snapshot
2. `CommandHandler.process_commands()` — handles one-shot flags (reset, save, load, etc.)
3. `process_camera_input()` — WASD/scroll camera
4. `sim.apply_state()` / `camera.apply_state()` — push state to GPU and renderer
5. `SimulationRunner.step_and_assemble()` — dispatch compute shaders, assemble frame
6. Render camera view, then UI on top

**Key design rules:**
- **UI is passive** — renders ImGui widgets and exposes state via `get_state()`, never runs sim logic
- **One-shot flags** — booleans set by UI (e.g. `request_reset`), read and cleared by orchestrator each frame
- **State containers** — all mutable state lives in dataclasses under `state/`
- **GPU-first** — physics runs in GLSL compute shaders; CPU just dispatches and reads back when needed

### UI Package (`ui/`)

Uses **mixin-based decomposition** (multiple inheritance). The `UI` class in `core.py` inherits its window mixins so all render methods share `self.*` — chosen because ImGui's immediate-mode paradigm requires shared widget state. Import via `from ui import UI`. See `ui/README.md`.

### GPU Pipeline

```
entity_update.glsl (compute) → fourier4_4.glsl (compute) → frame_assembly.frag (fragment)
```

Uniforms are set from Python via `tryset(program, 'UNIFORM_NAME', value)` which gracefully handles missing uniforms during shader development. Press `V` to hot-reload shaders.

## Writing Rules (comments, docstrings, tooltips)

**State the rule. Never the evidence, never the reasoning.**

Banned in comments, docstrings and UI strings: measured percentages, benchmark
timings, the word MEASURED, dates, before/after comparisons, "I tried X and it
failed", and multi-paragraph rationale.

- A **tooltip** is ONE sentence naming what the control does. Not why the
  default is what it is, not what happens at the extremes, not a number.
- A **comment** is one line naming a non-obvious constraint. If it is longer
  than the code beneath it, delete it.
- A **docstring** says what the function returns and what a caller must know to
  call it. Not the history of the design.

Measured facts have exactly one home: the caveats in this file, or a document
under `docs/`. Everywhere else points at that home instead of restating it — a
number in two places drifts, and then neither is evidence.

## Naming Conventions

- **SimState fields / shader uniforms**: `ALL_CAPS_UNDERSCORE` (e.g. `SENSOR_DISTANCE`)
- **UI labels**: Title Case with spaces (e.g. "Sensor Distance")
- **Private UI state**: `_snake_case` (e.g. `_request_reset`)
- Names must match exactly between SimState fields and GLSL uniforms

## Adding a New Physics Parameter

Four-step process (detailed in `docs/adding_ui_shader_params.md`):

1. **`state/sim_state.py`** — add field to `SimState` dataclass
2. **`shaders/*.glsl`** — declare `uniform` and use it
3. **`ui/physics_window.py`** — add ImGui widget (use `slider_float_with_range_menu` for full context menu support)
4. **`sim.py`** — add `tryset()` call in `entity_update()`

No additional wiring needed — the orchestrator pattern handles the rest.

## Important Caveats

Each of these exists because the obvious change was tried and measured. Read
`docs/imgep.md` before the search and archive sections — it explains the
mechanics these caveats assume.

### GPU, tiling and the tournament grid

- **Tile and cohort indices both derive from the particle index**, so the
  cohort count must be a multiple of the tile count — otherwise each tile holds
  exactly one cohort and stays a monoculture. See `services/cohort_tiling.py`.

- **`world_size` cannot change particle density.** It scales entity count by
  `ws` and canvas area by `ws`, so particles-per-texel is a constant 0.572 at
  every world size — it buys *more world*, not a more crowded one.
  `particle_density` is the multiplier that moves density, and lowering it is
  *cheaper*: at ws=4.0, density 0.1 runs 4.7x faster than 1.0.

- **`ACTIVE_COUNT` in `entity_update.glsl` is the denominator for every
  index-derived slice** — cohorts (`get_cohort`) and tournament tiles
  (`tournament_home_tile`). It must be `float(ENTITY_COUNT)` and never
  re-derived from `WORLD_SIZE`: a denominator larger than the buffer compresses
  every slice into the bottom of the range, so at density 0.5 the top half of
  the tournament grid renders empty.

- **A tournament tile is a SMALL WORLD under the world's own
  `BOUNDARY_CONDITIONS_MODE`, not a box with walls.** Under wrap it is a TORUS,
  with no edge for anything to pile against. Three places enforce the tile edge
  and must agree: the particle boundary block and the sensor confinement in
  `entity_update.glsl`, and `getBlur` in `canvas.frag`. The diffusion's guard
  must be the tile's own BOX, never an index recovered by clamping — a clamped
  probe re-enters the same tile, skips the zero-flux substitution and falls
  through to a `repeat`-wrapped sampler, which returns the far side of the
  canvas. Guarded by `tests/test_tile_isolation_gl.py`.

- **A tile owns a whole number of TEXELS, in INTEGER arithmetic, and `%` is
  banned near a seam.** The canvas is `int(1024*sqrt(world_size))` — 647 at the
  default 0.40 — and the grid is 2..8, so no canvas size divides evenly for
  every setting. `tile_lo_texel()` is integer ceil-division, defined once and
  repeated in `entity_update.glsl`, `canvas.frag`, `brush.vert` and
  `services/tile_geometry.py` (the capture crop uses it too). Two traps: GLSL
  division is not required to be correctly rounded, so a float seam test picks
  the wrong tile on exact halves; and `%` is undefined for negative operands,
  which the south and west probes at `lo - 1` always are. **Never test tiling
  at a power-of-two canvas** — at 1024 all of this passes and at 647 it costs
  84% of a tile's trail. `tests/test_tile_isolation_gl.py` runs 647/8, 647/3
  and 641/7 on purpose.

- **The trail is ONE moving average SPLIT ACROSS TWO SHADERS, and removing the
  brush texture bought NOTHING.** `canvas.frag` keeps `trail_persistence` and
  `brush.frag` adds `1 - trail_persistence` of this step's deposit straight
  into the same framebuffer, so `update()` binds it once and owns the
  double-buffer swap; neither pass may bind a target of its own. Measured over
  ten alternating runs the step is **0.799 ms against 0.793** - no change, with
  a slightly heavier upper tail, because the step is 70% brush RASTERISATION
  and the middleman only ever cost one full-resolution read (4.3% of the step
  is the whole canvas pass) plus one clear. It was adopted for parity with
  upstream, not for speed; do not re-attempt it expecting a win.
  **The deposit's weight is computed PER FRAGMENT in `brush.frag`, from its own
  copy of `calculate_setting`.** Upstream passes one `trail_persistence`
  uniform for the whole canvas, which is wrong wherever a sweep or jitter is
  on - the setting is position-dependent, and the two halves of one average
  must agree at every texel. `frag_world` IS `canvas.frag`'s
  `entity_space_pos`, so the two copies are evaluated at the same point;
  `tests/test_trail_deposit.py` asserts they stay character-identical.
  Three traps. The struct must be pushed to BOTH programs - feeding only the
  canvas leaves the brush reading an all-zero struct, which is a valid one
  meaning `slider_value` 0, so `(1 - p)` is 1 and every deposit lands 30x to
  1000x over with nothing raising anywhere; it took comparing canvas energy
  against the old pipeline to see it, and every source-level test passed while
  it was live. The blend is `ONE, ONE` now, so the kernel that `SRC_ALPHA` used
  to apply a second time is squared explicitly. And the particles now MOVE
  BEFORE they paint, where they used to paint before they moved - a one-step
  shift in the whole preset library that CANNOT be measured, because the
  splatting races and two runs of identical code already differ by more than
  the effect.

- **The entity struct is 32 bytes, and slimming it bought NO MEASURABLE
  SPEED.** `cohort` is a pure function of the index that `get_cohort()` already
  computes, and `color` was a vec4 whose brightness (1.0) and alpha (0.045)
  were the same constants for 600k particles every frame - so both went, and
  the vertex shaders rebuild the vec4. That is a third off entity-buffer
  traffic exactly as upstream claims, and the buffer is NOT the bottleneck:
  1.377 -> 1.354 ms/step at the default world size (inside noise), nothing at
  density 4.0, 1.2% at 8.0. The step is dominated by the two `get_can()` taps
  and the brush splat. Keep it for the 16 bytes per particle of VRAM and the
  two redundant fields, not for frame time; do not quote 33% as a speedup.
  **SATURATION is still stored, and that is not an oversight** - `reset()`
  writes a DESATURATED particle where the update writes 0.8, and `HAZARD_RATE`
  respawns particles continuously, so hardcoding 0.8 would change what a
  hazard-heavy preset looks like. It is free: the `vec2` members force an
  8-byte alignment, so the slot is padding either way. Two traps, both of which
  bit. Four files hardcoded a stride of `12`, and a stale stride does not fail
  - it reads other fields AS positions, so two GPU tests passed in isolation
  and failed in a full run on whatever the misaligned columns held; guarded by
  `tests/test_entity_struct.py::test_nobody_hardcodes_the_stride`. And
  `sim.reset()` clears the canvases and zeroes `frame_count` but never touches
  the entity buffer - the particles are reset by the shader on the
  `frame_count == 0` step - so a snapshot taken between the two reads
  uninitialised VRAM, which is a coin toss that the old 48-byte allocation
  happened to keep winning.

- **The frame-constant uniforms go up once per FRAME, and the cache covers
  EXACTLY what `apply_state` hands over.** The physics runs `speedmult` steps
  between two rendered frames and nothing in `SimState` moves across them, so
  uploading the setting structs per step was 138 `tryset` calls per step
  against 35.5 now. It buys **3.5x** where the CPU is the limit (0.444 ->
  0.128 ms/step at world size 0.05) and **nothing at all** at the default 0.40,
  where 240k particles make the step GPU-bound - the saving is real either way,
  it is just hidden behind the GPU. Three traps. The mark is the PROGRAM the
  uniforms went to, never a flag: `sim` keeps one entity-update program per set
  of shader defines and `realloc_brain_buffers` swaps between them, so a flag
  would hand a program pulled from that cache whatever it was last given.
  Anything reached by a path OTHER than `apply_state` must stay per-step -
  `reset()` moves `RESET_SEED` and `apply_tournament()` rewrites the grid, and
  caching either one is caught by `tests/test_reset_seed.py` and
  `tests/test_tournament_cohort_colour.py`, a long way from the code that broke
  them. Arguments stay per-step too, which is what keeps the rule statable. The
  multi-load trail values advance with progress every step and are excluded on
  those grounds, as are `frame_count`, `WRITE_RULES` and `fill_mode`. The brain
  uniforms are excluded because `set_brain_scales` changes the layout WITHOUT
  changing the program. Guarded by `tests/test_uniform_caching_gl.py`.

- **`V_MAX` caps the WHOLE STEP, because STRAFE IS ADDED STRAIGHT TO POSITION
  and never touches `e.vel`.** A cap on velocity alone caps nothing a strafing
  preset does: at `V_MAX = 1e-9` the default preset still covered 78% of its
  free distance per frame, and Karst moved *further* clamped than unclamped,
  the velocity it lost having partly opposed its hop. `entity_update.glsl`
  therefore forms `step_delta = e.vel + hop` and scales BOTH that and `e.vel`
  by the same factor — scaling only the step lets speed pile up behind the cap
  and lurch when it is raised. `hop` samples `STRAFE_POWER` at `e.pos + e.vel`,
  which is where the two-statement version read it, so a sweep on Strafe Power
  is unchanged. The advanced-drawing field is folded in before the cap for the
  same reason — its strafe half is a third straight-to-position term — and it is
  sampled at `e.pos + step_delta`, which is the point the two-statement version
  read it at, so an unbraked brush is unchanged. Guarded by
  `tests/test_v_max.py::test_a_zero_limit_actually_stops_a_strafing_preset`,
  which RUNS the sim — every source-level test passed while the bug was live.

- **`V_MAX`'s range is measured, and its units are CANVAS UNITS PER STEP — not
  a fraction of the screen.** Over the 131 presets, step distance runs 0.000036
  (Ooze, median) to 0.046 (Shrimp, p99) in slider units, a 1000x span; the
  canvas is 2 units wide, so even the fastest preset crosses under 4% of it per
  frame. The slider is `0..0.1` with a FOURTH-power curve, putting the slowest
  preset at 14% of the track and the fastest at 82%. A linear or too-wide range
  crowds the whole library into the bottom and reads as a control that does
  nothing: the first attempt at `0..1` let a particle cross 79% of the canvas
  in one frame. The default IS the range max, and the range max is an OFF
  SWITCH — the shader skips the clamp entirely at `vraw >= vm.max_value`, and
  the slider reads "Off" there (`PhysicsParamDef.off_at_max`). "Above every
  preset" is a measurement a hand-tuned config can beat; off is a fact, and a
  preset saved before this parameter existed must load unbrakeable rather than
  merely unbraked. The two must agree or the readout lies. Two traps when
  re-measuring:
  `particle_density` changes trail intensity and feeds back into speed — one
  preset runs 28x faster at 1.0 than at 0.25 — and a wrap, bounce or hazard
  respawn teleports a particle, so the peak `|dpos|` is a boundary event rather
  than a step and the range must be set off a percentile.
  `python -m tools.measure_speed`.

- **`TIME_SCALE` is compensated in TWO moving averages, and that is what makes
  it a slower creature rather than a different one.** Scaling the position
  alone gives a particle that turns as sharply per step while covering less
  ground, and a trail that takes in a whole step's deposit for half a step's
  travel — tighter and thicker, not slower. Both corrections are one line
  each, because both quantities are EMAs. The momentum filter `vel = vel*d + f`
  settles at `f/(1-d)`, so a step covering `ts` retains `d^ts` and takes in
  `(1-d^ts)/(1-d)` of the force, which holds that steady state exactly. The
  trail's decay and deposit are two halves of ONE average — it keeps `P` and
  takes in `1-P` — so `pow(P, ts)` corrects both at once, and two steps at
  half time land exactly where one step at full time did. The step is scaled
  AFTER the `V_MAX` clamp, or the cap would stop being a distance per unit
  time. **Every `pow` is guarded on `ts == 1.0`**, because GLSL does not
  promise a correctly rounded `pow` and an unguarded `pow(x, 1.0)` would
  perturb the whole preset library by a bit or two; multiplying by exactly 1.0
  needs no guard.

- **`TRAIL_DIFFUSION` is a THIRD thing on the clock, and "it is off in almost
  every preset" was never true.** The blur is one step of an explicit heat
  solve — `getBlur` returns `(1-a)*centre + a*neighbour_average` for
  `a = 4/(4+K)` — so an uncompensated clock ran `1/ts` of them per unit of
  simulated time and the trail smeared as `1/sqrt(ts)`. Measured on the shipped
  shader over a fixed span, the spread in VARIANCE went 8.0 at `ts=1` to 16.0
  at 0.5, 80 at 0.1 and 398 at 0.02 — **fifty times**, a seven-fold wider
  trail, and the sensors read that field. The slider's distribution over the
  131 presets is BIMODAL and the claim above had it backwards: **83 sit at
  exactly 1.0, the maximum**, 47 at 0, one at 0.36. At 1.0 a single step moves
  80% of a texel's mass into its neighbours.
  Variance ADDS, which is what makes the correction linear in `a` even though
  the slider's own mapping is not: `a_ts = ts*a`, solved back for `K` in
  `time_scaled_K`. It is CAPPED at `a = 0.8` — the strongest step the slider
  itself reaches — because the solve is explicit and `a` above 1 grows the
  checkerboard mode without bound; a clock above 1.0 at full diffusion
  therefore under-diffuses rather than blowing up, which is the price and not
  a bug. Guarded by `tests/test_time_scale_diffusion_gl.py`, which ping-pongs
  the WHOLE canvas.frag and measures the second moment — a source-level
  reading cannot see this, because what is wrong is a property of the
  recursion rather than of any one line.

- **Four things in the step are still NOT on the clock, and they are the rest
  of why a slower sim is a different creature.** `HAZARD_RATE` is a per-STEP
  probability (36x the respawn rate per unit time at `ts=0.02`, 15 presets);
  the advanced-drawing FORCE kick is added after the momentum filter with no
  `force_gain`, so it settles at `c/(1-d^ts)` — 39x at `ts=0.02` — while its
  strafe twin on the next line rides in `step_delta` and IS scaled, so the two
  halves of one feature disagree; jitter is re-drawn per step from
  `frame_count`; and negative drag (21 presets) alternates sign every step, so
  its ripple is at the step rate and has no continuous-time meaning at all —
  that last one is a limit, not a fix. The two look-ahead sample points
  (`e.pos+e.vel` for `STRAFE_POWER`, `e.pos+step_delta` for `get_field`) are
  also unscaled, which only bites under a sweep or a drawn field.

- **A `plain_uniform` parameter is in `PHYSICS_PARAMS` but not in
  `PHYSICS_PARAM_NAMES`.** It is one float for the whole canvas, so there is
  no sweep or jitter to offer and no entry in those dicts — declaring one
  would be the "declared but never read" defect the registry exists to stop.
  It stays in the table regardless, because the slider, the label, the range
  menu and the audio target must all come from one place: a second list of
  modulatable parameters is exactly what
  `tests/test_audio_mapping.py::test_physics_targets_come_from_the_registry_not_a_second_list`
  forbids, and it caught the first attempt at this. `TIME_SCALE` is also
  absent from `PhysicsConfig`, so loading someone else's preset leaves your
  tempo where you set it.

- **The `MultiLoadConfig` struct is written by OFFSET, and a physics
  parameter's label must be `title()` of its field name.** Two raw std430
  writers pack it in declaration order — `_write_multi_load_ssbo` and
  `write_tournament_physics` — so a member the shader has and Python does not
  shifts every field after it into the wrong slot. Both also look a custom
  slider range up by `NAME.replace('_', ' ').title()`, which does not raise
  when it misses: it silently hands that tile the default range. `V_MAX` is
  therefore labelled "V Max". `MULTI_LOAD_CONFIG_SIZE` in `sim.py` is the one
  definition of the struct's width, used by both the buffer reserve and the
  zero-fill. Guarded by `tests/test_v_max.py`.

- **The canvas and the brush are RG32F, because the trail is a VELOCITY FIELD
  and nothing reads a third channel.** The brain senses `ltap.xy`, the canvas
  view draws `atan(y,x)` and `length(xy)`, and the alpha the additive blend
  needs is the FRAGMENT's, not the target's — `SRC_ALPHA, ONE` into a
  two-component target is bit-identical to RGBA. Halving it is worth 6.7% of
  the sim step at world size 0.40, 10.9% at 1.0 and 18.9% at 4.0, over six
  presets. Two traps when re-measuring: the sim does not reproduce itself run
  to run, so both formats must be restored from ONE entity-buffer snapshot
  before each timed block — an uncontrolled run scored one preset 37% *slower*
  — and two runs in separate processes measure the laptop's thermal state, not
  the format. RGBA16F is the obvious alternative and it fails: density reaches
  47.19 over the preset library, where fp16 spacing is 0.031 against a
  particle's 0.01 deposit. RGB32F is not a required renderable format.
  Guarded by `tests/test_canvas_format.py`.

- **`canvas_view_rect` returns a GL texture coordinate (v=0 at the BOTTOM);
  `tex_to_screen` returns top-down window coordinates.** The flip cancels
  exactly when the canvas is centred, so a centred camera validates any
  orientation bug you like. Test capture crops with the camera *panned*.

- **The tournament capture does not crop the window view — it re-renders the
  canvas** at `grid*224` with an identity camera (`services/capture_view.py`),
  so pan, zoom and window size cannot affect what the optimizer scores.
  Anything needing the *displayed* image (screenshots, video) still uses
  `camera.assembled_texture`.

- **Bloom in the capture runs per tile, not over the grid.** It is a 5-level
  mip chain reaching 30–60px, so blooming the whole grid put a quarter of a
  tile's worth of neighbouring glow into every crop. `CaptureView.draw_grid`
  splits first and blooms each tile alone.

### The encoder and the capture

- **The encoder is a REGISTRY entry, and its key is written to disk.**
  `services/vision_models.py` is the one home for everything that differs
  between encoders — paths, preprocessing, tokenizer context, both logit scales
  and the separation bar. A key (`clip-b32`, `siglip2-b16`, …) names an
  archive's embedding space in `encoder.json` and can never be renamed, the
  same class of fact as `BrainLayout.signature()`.

- **Each encoder's scales and separation bar are MEASURED against `clip-b32`'s
  BEHAVIOUR, never scaled off a summary statistic.**
  `python -m tools.calibrate_encoder --entries 700`, over the three most
  recently worked archives:

  | key | image scale | text scale | min sep | ms/image | dim |
  |---|---|---|---|---|---|
  | `clip-b32` | 30.0 | 100 | 0.0200 | 4.3 | 512 |
  | `clip-b16` | 33.1 | 170.3 | 0.0195 | 8.0 | 512 |
  | `siglip2-b16` | 32.5 | 150.8 | 0.0195 | 11.0 | 768 |
  | `clip-l14` | 17.7 | 129.0 | 0.0519 | 33.0 | 768 |

  `clip-b32` keeps its historical values, which the calibration reproduces to
  4.7% and 0.5% — that agreement is the harness's own gate, and a run that
  misses it means the harness is wrong, not the registry. Two criteria carry
  the weight: the image scale is the one whose FLOORED FRACTION under a +3sd
  latent goal matches `clip-b32`'s, and the bar is the threshold that ADMITS
  the same fraction of a real archive. The text scale is the WEAKEST of the
  three — nothing floors at `clip-b32`'s trained 100, so it reads the onset of
  flooring and keeps the same margin below it.

  **The median is a bad predictor, which is why this is measured.** SigLIP 2's
  median nearest-neighbour distance is 19% above `clip-b32`'s, but its
  separation bar came out **0.0195** against the 0.024 that scaling predicted:
  admission sees the LOWER TAIL, not the middle. **And the bar must be read off
  a real sample size** — at a few dozen entries every stored entry clears it,
  retention saturates at 1.0 and the criterion returns the sample minimum.
  These ms/image figures include preprocessing; `tools/hue_nuisance` times
  `session.run` alone and reports lower ones.

- **A pooled embedding is chosen BY NAME, on BOTH towers.** SigLIP's exports
  put `last_hidden_state` first on the vision *and* text sessions, so
  `get_outputs()[0]` is an `(N, tokens, dim)` tensor. Nothing checks it: the
  mismatch surfaces as a matmul error deep inside `contrastive()`, far from the
  session that chose the wrong output. `pick_embedding_output` is one function
  used by both towers and by the measurement tools, because fixing the vision
  side alone is exactly the bug that shipped once already.

- **CLIP is strongly POSITION-dependent — use `embed_mean()`, never `embed()`.**
  Rolling a tile 16px on the torus moves its embedding 2.5–2.7x further than
  its nearest genuine neighbour (the dip at exactly 32px is ViT-B/32's patch
  stride, which is the mechanism). Averaging `n_views=3` random sub-crops cuts
  that nuisance to 0.032 while adding only 0.007 of new sampling noise, a third
  of the 0.02 separation bar. `embed()` returns `B*v` rows unreduced, so
  feeding those to the archive makes three sub-crops of one tile into three
  descriptors. Centring on the centre of mass was measured and rejected:
  26–34% of tiles have no well-posed centre, and it pushed real near-duplicates
  apart by 15–19%.

- **CLIP time is mostly CPU, and most of that was normalisation.** Per 64
  images the split was `preprocess` 112.6 ms / DirectML 58.4 ms / `augment`
  28 ms. `preprocess` now transposes while still uint8 and folds normalisation
  into one in-place multiply-add (112.6 → 48.7 ms), and `_embed_images`
  pipelines one chunk ahead on a worker thread — lookahead is exactly one
  chunk, since submitting all of them would hold 345 MB at grid 8. Only
  `preprocess` may leave the main thread: `augment` draws from `self._rng`, and
  only one thread may call `session.run`. Never pin `device_id` — DirectML
  device 0 is the discrete GPU (0.90 ms/image); device 1 is the iGPU at 63.

- **Hue IS the axial force term — colour is not an independent gene — and the
  gain that maps it to a hue is a PRESET parameter, not a constant.**
  `entity_update.glsl` forms `force = baseterm.xy + y_reflect(mirrorterm.xy)`
  and `color = baseterm.xy + mirrorterm.xy`, and `y_reflect` flips only the
  lateral component, so `color.x` and the axial force are the SAME expression,
  read before `AXIAL_FORCE` scaling. This holds for every modality — it is the
  black box's output, not a Fourier one. `e.color.x = HUE_SENSITIVITY *
  col_params.x` then reads it as an HSV hue modulo 1, so the gain scales
  `|dhue|` linearly and a large one slides the colour through whole revolutions
  on a mutation far too small to change the pattern.

  The cost is measured against the admission bar. At gain **0.5**, `|dhue|` is
  74° at `expedition_sigma` and a hue-only rotation moves the CLIP embedding
  **1.26x** as far as a genuinely different creature does, clearing
  `min_separation` **95%** of the time. At **0.09** it is 14°, 0.22x, and
  **10%**. **A better encoder does not fix this** — CLIP B/16, SigLIP 2 and
  L/14 all score 1.26–1.38 at 74°, and SigLIP 2 is the most colour-sensitive of
  the four. Re-measure with `python -m tools.hue_nuisance`.

  Both defaults are already in the good range (`SimState` and `_Default.json`
  are 0.09), but roughly half the preset library ships **0.5** — so without a
  cap, how much of the archive is hue-duplicates depends on which preset seeded
  the run. `AUTO_HUE_MAX` (0.12, `main.py`) holds the gain down for as long as
  an automatic mode is scoring, re-applied every frame because a preset loaded
  mid-run brings its own. Do not lower it much further: at 0.15 the spread
  across genomes is 107° against 68.6° at 0.5, and it collapses again as the
  gain approaches zero.

  Clamping is safe because colour is DISPLAY-ONLY: the canvas is RG32F velocity
  and the brain senses `ltap.xy`, so `e.color` reaches only `brush.vert` and
  `cam_brush.frag`. It changes what CLIP sees, never how a particle moves. At
  0.09 colour IS heritable, so a text goal naming one is no longer hopeless —
  it was the 0.5 gain that made it so. `hue_sensitivity` is NOT in
  `PHYSICS_PARAMS`, so the search cannot tune it itself. Note `color_by_cohort`
  REPLACES the brain's hue with `hash(cohort)`; tournament mode forces it off,
  so any colour experiment run outside tournament mode measures the wrong
  thing. Guarded by `tests/test_auto_hue_clamp.py`.

### The archive and admission

- **An archive is pinned to ONE encoder AT CREATION, and every control over it
  afterwards is a readout.** `archive_library.create()` writes
  `<archive>/encoder.json`, which sits at the archive ROOT beside `goals.json`,
  not under the layout signature — one archive holds every brain and switching
  brain must not switch embedding space. Creation is the only moment the
  archive holds nothing; pinning any later leaves a live-looking combo over an
  archive whose vectors are already committed, and a control that cannot do
  what it offers is worse than no control. So the CHOICE is in the New Archive
  modal, and the Explore tab's Encoder combo is permanently disabled and
  follows `archive.encoder`. `pin_encoder` refuses to overwrite, the same
  discipline as `save_run_config`. **A missing file means `clip-b32`**, so
  every archive written before the choice existed opens untouched and no
  migration runs. `Archive.load_from_store` refuses a store whose encoder
  differs and sets `encoder_mismatch`, because at equal width a foreign vector
  is silently wrong rather than an error — `clip-b32` and `clip-b16` are both
  512-d. Migrating an archive between encoders is deliberately NOT implemented;
  the thumbnails are 160px against a 224px capture, so re-embedding would mix
  fidelities against entries admitted afterwards. A new archive also reseeds
  `min_separation` from its encoder, because `load_settings` returns `{}` for
  one — "keep what is on screen" would otherwise inherit the outgoing
  archive's bar, which is a distance in a different space.

- **The encoder NOBODY CHOSE is the one that must offer to download itself.**
  Auto's key comes from a combo and `_ensure_auto_service` gates it on
  `is_present`, which is what draws the Download button; Explore's comes off
  `encoder.json` and was handed straight to `VisionScorer`, so a missing model
  surfaced as a raw `ONNXRuntimeError: NO_SUCHFILE` — inside a tab that
  returns before it draws anything else, leaving no control on screen at all.
  `_ensure_archive_service` gates the same way and the `model_missing` branch
  draws the archive ROW as well as the button: the other way out is an archive
  whose space IS on disk, and nothing else in that tab can be drawn.
  `archive_unavailable` is cleared where the driver is REUSED, not only where
  it is built, or the banner outlives the archive that raised it. The request
  is `ArchiveState.download_model_requested` rather than Auto's, handled ABOVE
  `_handle_explore`'s early return — a missing encoder is exactly why the
  driver does not exist. Guarded by `tests/test_missing_archive_encoder.py`.

- **The weights are USER data, and `utilities.paths.get_models_root` is their
  one home.** They used to be read from `"models"` relative to the CWD, which
  made them a property of whichever folder was launched rather than of the
  user: every worktree wanted its own copy — `clip-l14` alone is 859 MB — and
  a packaged build under Program Files cannot write that folder at all. The
  same two reasons `imgui.ini` moved. `migrate_models` runs at every launch
  and follows `migrate_legacy_archive`'s rules exactly: a MOVE, because these
  are gigabytes, and nothing at all once the target exists, so a second
  checkout cannot donate its copy over the one in use. It uses `shutil.move`
  rather than `os.replace` — a rename within one volume, a copy across two,
  which is the difference between an instant migration and downloading
  everything again. It deliberately does NOT create an empty `models/`:
  indistinguishable from a finished migration, it would strand weights still
  sitting beside the app. A failure is not fatal — the weights stay where they
  are and read as not-downloaded, which the button above already answers.
  Guarded by `tests/test_models_migration.py`.

- **Auto's encoder picker is read EVERY FRAME; Explore's is read once.**
  `_ensure_auto_service` returns early once the service exists, so the combo
  beside the prompt reached the scorer exactly once and then changed nothing —
  it sits in a tab that is already open by the time it can be touched.
  `_follow_auto_encoder` runs per frame, stands down whenever Explore owns the
  driver, re-embeds the goal (the old prompt embedding is in the outgoing
  space, and 512 against 768 is not a shape error until the first tile
  arrives) and aborts the generation in flight. Guarded by
  `tests/test_scorer_lifecycle.py`.

- **`_ensure_scorer` names EVERY DRIVER, because the service holds no scorer
  of its own.** `AutoTournamentService.scorer` is a PROPERTY forwarding to
  `self.driver`, and the two drivers are swapped in and out by mode — so
  assigning through the service reached whichever was installed and left the
  other on the encoder that had just been replaced. Which one missed depended
  on the mode that happened to be running when the archive's encoder was
  adopted, and `_follow_auto_encoder` heals it only when Auto's chosen key
  differs from the resident one. At equal width a foreign vector is silently
  wrong rather than an error — `clip-b32` and `clip-b16` are both 512 — so
  this fails quietly. The build happens before any reassignment, so a failure
  leaves every holder on the outgoing encoder rather than half-swapping the
  app. Guarded by `tests/test_scorer_holders.py`.

- **The settings a run was carried out under are a LOG, not a field.**
  `settings.json` is rewritten wholesale, so the `min_separation` that admitted
  entry #4000 is gone the moment the slider moves.
  `<archive>/settings_history.jsonl` is append-only: version 0 carries the
  whole block, every later row is a diff, and each `index.jsonl` row carries
  the `cfg` version in force when it was admitted. A row without one reads as
  version 0, which is every entry admitted before this. Written once per
  generation and once more when the archive is let go — a change made after the
  last generation reaches the log nowhere else, which is why `record_settings`
  rides with `save_settings` in the switch order rather than after it. The
  version in force is read from the LOG rather than from what this session
  wrote, or reopening an archive rewrites version 0. A field vanishing from
  `PERSISTED_FIELDS` is NOT a change: recording it would put a phantom row in
  every archive on the first run after a code change.

- **An entry's physics is TWO layers, and the base is per RUN, not per entry.**
  `<archive>/runs/<run_id>.json` holds the whole `PhysicsConfig` the run was
  carried out under; the entry's `_phys` vector overrides it for the parameters
  the optimizer searched. The vector alone is not enough and never was: with
  `physics_enabled` off it is `zeros(8)`, so a brain-only entry replayed under
  whatever the sliders happened to say — which is the entry's whole physics.
  Nothing in `_phys` ever covered trails, boundary mode, cohorts, hue
  sensitivity or `rule_seed` under any setting. Written from
  `_record_run_physics` off `ast.running` rather than off `start_requested`, so
  a resume and a run already going both reach it, and `save_run_config` refuses
  to overwrite — a run id names ONE set of physics, and a second would
  reinterpret the entries already filed under it. A missing file means "leave
  the sliders alone", so every archive written before this still opens.

- **The archive stores decoded phenotypes, never `z`.** With physics search on,
  `z` is relative to `physics_origin` — the preset loaded at the time — so a `z`
  archived under one preset decodes to a different creature under another. See
  `tests/test_physics_origin_roundtrip.py`.

- **Widening `services/physics_genome.PHYSICS_PARAMS` widens `Archive._phys`,
  and every `vectors.npz` already on disk holds the current width.** A new
  physics parameter is a PRESET parameter by default — `HAZARD_RATE` and
  `V_MAX` are both excluded from the search — and joining the search space
  needs a load-time migration before it needs anything else.

- **Admission does not gate on novelty; capacity prunes.** Everything finite,
  viable and alive is admitted, and `prune_to_capacity()` evicts the least
  novel once over the cap — one bulk pass per generation, from `tell()`, AFTER
  `refresh()`. Do not reintroduce the adaptive threshold removed 2026-08-08.
  Two independent reasons, neither fixable by gain: its `observe()` ran once
  per *candidate*, so it could move 2.18x in one generation at grid 4 (22.7x at
  grid 8) while steering on a rate averaged over ~6 generations; and a
  generation's tiles are not independent draws — they share one parent sample
  or one CMA-ES population, so they clear or miss any bar together.

- **Admission gates on SEPARATION, and that is not the threshold coming back.**
  `min_separation` refuses anything within that cosine distance of a stored
  entry. **The number is PER ENCODER** — 0.02 is `clip-b32`'s, and the
  registry holds the rest; see the encoder section. The unstructured-archive
  rule from
  quality-diversity, with no controller and no gain, and correlated tiles
  landing on top of each other is the case it is *meant* to reject. Without it
  a converging expedition stored its own endpoint 64 times a generation: one
  goal had contributed 25–30% of a real archive. The useful span is 0–0.05; at
  0.05 every archive keeps under 6%. A separation rejection is **not** added to
  the rejects ring, because the ring is memory of regions the search was
  refused and this region is in the archive already.

- **Liveness is a FLOOR on the bulk, never a veto over a chosen entry, and it
  ranks nothing.** `liveness_min` is **0.002**: over all 131 presets liveness
  runs 0.0034–0.0792 with median 0.0241, so a "reasonable-looking" 0.02 rejects
  a third of the curated preset library. Re-run
  `python -m tools.calibrate_imgep --liveness` before changing it. It appears in
  exactly two functional places — the archive's admission gate and the
  `viable`/`shows_something` split in `ImgepDriver.tell` — while `descriptor()`,
  the expedition fitness and parent sampling all ignore it.
  `consider(ignore_liveness=True)` drops the CHANGE half for `summit` and
  `record` only; `cand.viable` still has to hold. The reason is that liveness is
  *higher* during the post-reset transient than once a pattern settles, so a
  converging expedition's endpoint — the very thing the chase is for — scores
  low on it. `keeper` KEEPS the veto, because it fires every generation forever
  and a dead preset would otherwise deposit one frozen tile per generation
  without bound.

- **`keeper` is the most NOVEL tile; `summit` is the best-MATCHING one, and an
  expedition needs both.** A chase converging on a goal makes tiles that
  resemble each other, so its best result is precisely what separation
  discards, and during convergence the most novel viable tile is close to the
  least goal-matching one. `_summit()` is a **ratchet** on the expedition's own
  best fitness, so a converged expedition stops improving and therefore stops
  admitting; worst case is one extra entry per generation. The mark resets in
  `start_expedition_with` as well as `end_expedition`, because contrastive
  fitnesses against different goals are not comparable — and the expedition
  fitness therefore has to be computed BEFORE the admission loop.

- **The record book runs in EVERY regime, and is measured against the ARCHIVE,
  not the run.** A tile matching any enabled text goal better than anything the
  archive holds is admitted with `source="record"` and `goal` set to the phrase
  it beat. Not a variant of `summit`: a run chasing "pepperoni pizza" routinely
  produces the best "a smiley face" the archive has ever had, and nothing else
  keeps it — novelty does not know the goal list exists, and separation asks
  whether the archive holds something SIMILAR, not something BETTER. Scored on
  DESCRIPTORS on both sides, deliberately not the per-snapshot averaging the
  expedition fitness uses, because archive entries have no snapshots.
  `_goal_records` is keyed BY TILE so a tile topping several goals is admitted
  once. It cannot move to `precompute()` — it reads `archive.embeddings`, which
  is main-thread state. Cost is 1 ms/goal at 4808 entries, 3 ms at 13049.

- **`Archive.refresh()` is what makes pruning meaningful, and its budget is a
  FRACTION of the archive, never a fixed count.** Parent choice and eviction
  both rank on *stored* novelty, so what matters is how many generations a full
  sweep takes — `refresh_sweep_gens`, default 10. Staleness is directional and
  self-reinforcing: patterns accumulate near each other so true novelty only
  falls, a stale value is systematically too HIGH (41–59% of entries measured
  inflated), and an inflated entry is both likelier to be chosen as a parent and
  likelier to survive eviction. The old fixed 64/generation took 312 generations
  (14.6 min) to sweep 20000. Cost at sweep=10: 0.9% of a generation at 4808
  entries, 12.7% at 20000.

- **Novelty is a LIVE column, so `index.jsonl` cannot be its home.** The index
  is append-only, so its `novelty` is forever the at-admission value: entry #50
  was scored against 49 neighbours and #4000 against 3999, and the stored column
  correlates **0.075** with a correct rescore. It lives in `vectors.npz`
  (rewritten wholesale, an OPTIONAL key so older archives still open). Trusting
  it as written hands generation 0 — every tile stamped 1.0 by the no-reference
  convention — 100% of the `p ~ novelty^4` parent weight, and one of those
  entries is a black frame. Four things read this column: expansion parents,
  `latent_goal`'s anchor, `prune_to_capacity`, and the browser sort.

- **`rescore_all()` is paid by the CLOSING flush, not by every open, and
  `novelty_n` is what makes that safe.** It is O(n²) — 0.23 s at 3792 entries,
  4.3 s at capacity — and it used to run on every `load_from_store`, so
  switching brain layouts to browse a different archive stalled for a quarter
  second on work only the *search* needs. `maybe_flush(closing=True)` therefore
  rescores when the column is dirty and stamps `novelty_n`, the ARCHIVE-wide
  count it was scored against. **`force` is NOT the trigger**: it also covers
  writes made while the archive stays open — deleting one entry from the
  browser — where a rescore per click is the same stall back again. A periodic
  flush stamps nothing either, since it runs mid-generation and the column
  would be dirty on the next candidate. An open trusts the column when
  **every** layout directory claims the same count and it matches what
  actually loaded — otherwise it rescores exactly as before, which is also
  what a new close path that forgets `closing` costs.
  **A DROPPED ROW IS NOT A REASON TO RESCORE, and requiring that cost 4.4x on
  every open.** An index row with no vectors is the ordinary record of a
  REMOVAL — a browser delete, or an eviction — and `index.jsonl` is
  append-only, so it stays for the rest of the archive's life: one deleted
  entry made the archive re-score itself on every open and on every
  cross-brain click, for good. Measured on a real 7145-entry archive with 69
  deletions, an open went **697 ms → 158 ms** with nothing on disk touched.
  The COUNT is what carries the safety, and it is enough on its own: the
  column is written in the same call as the rescore that produced it, so a
  stamp means "this file's column covers exactly this file's ids", and the
  other direction — a vector row the index lost, where FEWER load than the
  stamp claims — fails the count check. Do not put `not dropped` back. On
  a reopened archive the trusted column is bit-identical to the rescore it
  replaces, both sides reading the same fp16 vectors; entries admitted in
  memory were scored at fp32 and differ by under 2e-3, two orders below
  `min_separation`. Browse-only reopen measured 230 → 60 ms at 3792 entries.
  Only `rescore_all()` may set `_novelty_clean`; `_add` and `_remove` clear it,
  and `refresh()` leaves it alone — a partial sweep does not make a dirty
  column comparable. A file with no `novelty_n` reads as dirty, so every
  archive written before this opens untouched and pays once.

- **Novelty is measured against archive ∪ rejects ring.** The archive is gated,
  so without the ring the search has no memory of the regions it just rejected
  and re-explores them forever.

- **`knn_distances` blocks over query rows.** Only k distances per query
  survive, so the (n, m) matrix is scratch — a whole-archive rescore at capacity
  would ask for 1.6 GB of it at once. Blocking caps it at 64 MB and costs
  nothing measurable.

- **`_remove()` deletes the entry's thumbnail.** Nothing could reach it
  afterwards — `index.jsonl` is append-only and the id is gone from
  `vectors.npz` — and without this a full archive at grid 8 orphans 64 JPEGs
  every ~2.8 s, about 12 MB a minute.

- **`_next_id` comes from `index.jsonl`, NOT from the entries that survived
  reconciliation.** The index is append-only, so it records every id ever
  *issued*; `self.entries` holds only the ids still backed by `vectors.npz`.
  Deriving the counter from the survivors restarts it at the first id whose
  vectors were lost, and the next run re-issues ids that already exist —
  overwriting those entries' thumbnails (the filename derives from the id) and
  leaving duplicate rows that shadow the originals on the following load.
  Measured in `debug09`: one unclean exit stranded 157 entries, and each of the
  two runs after it re-issued 1005–1161. Eviction reaches this too, since
  `prune_to_capacity` can remove the highest-id entry even on a clean quit.
  Guarded by `tests/test_archive_id_reuse.py`.

- **The frame loop is wrapped, because a crash used to cost the run.** An
  exception in `orchestrate_frame` propagated out of `run()` and `cleanup()`
  never ran, losing every admission since the last 200-admission vector flush,
  the goal list, the settings, and any record of the cause — what reached the
  user was a screenful of moderngl `Texture.__del__` errors, which are
  interpreter *teardown* noise (`isinstance(x, None)` once module globals are
  cleared) and never the bug. `run()` now writes the traceback to
  `Documents/Fluoddity/crash.log`, runs cleanup in a `finally`, and re-raises.
  Each step of `cleanup()` is individually guarded by `_step()` and ordered by
  what is lost if it does not run: on a lost device every GL call raises, so
  one unguarded failure would skip the archive flush below it.

- **Entry ids restart at 0 in every archive**, so `ThumbCache` must be released
  on a switch: it is keyed by thumbnail filename, which derives from the entry
  id, so reusing it shows the previous archive's pictures.
  **A BRAIN SWITCH IS NOT THAT SWITCH, and treating it as one reads as a
  reload.** A layout change moves to a SIBLING directory under the same
  archive name, so the ids, the thumbnail keys and the map's positions are all
  unchanged - and clicking an archive entry of another brain performs one.
  Dropping the caches there re-decoded the whole atlas, and dropping the map
  layout left `_render_map` with nothing to project for a frame: the tab
  collapsed to one line of text, which made ImGui clamp the browser window's
  scroll back to the TOP. `_release_archive(keep_thumbs=True)` and a `bind()`
  that keeps its layouts when the PATH and encoder are unchanged are the two
  halves; `_build_archive_set` reuses a surviving cache through
  `set_loader`, since only the stores the loader resolves through are new.
  The ordering is what makes the blank frame reachable at all -
  `_update_map_layout` runs at the top of `orchestrate_frame` and the switch
  runs in `process_commands` below it, so the refit cannot land until the
  NEXT frame. Guarded by `tests/test_archive_switch_thumbs.py`.

- **Explore settings belong to the ARCHIVE, not to the app**, and live in
  `settings.json` beside its `goals.json`. The settings that suit a
  20000-entry archive are not the ones that suit an empty one, so reopening an
  archive restores what it was last worked with. `PERSISTED_FIELDS` in
  `state/archive_state.py` is an explicit **allowlist**: two thirds of
  `ArchiveState` is one-shot commands and view buffers, and persisting
  `start_requested` or `delete_entry_id` would replay a command on load. A new
  field is therefore not persisted until it is named there. `enabled` is
  deliberately absent — opening the app must not resume a search — and a
  missing key keeps its current value, so `{}` means "keep what is on screen"
  and a brand new archive inherits the settings you were just using. Written
  before the store is closed (a switch, and quitting); `grid` is the one
  restored field that also needs `grid_changed`, since the per-frame
  `configure()` push does not rebuild the tournament grid.

- **`Archive.maybe_flush` only rewrites `vectors.npz` every 200 admissions.**
  `index.jsonl` is flushed per entry, so anything that closes an archive —
  quitting, or switching — must call `maybe_flush(force=True)` first or lose
  the trailing entries.

### The search and its goals

- **Scoring runs OFF the frame loop, and the split is exactly `precompute()`.**
  CLIP is 92–97% of a generation's main-thread cost (`tell()` is 1143 ms at 1
  view and 4324 ms at 3 at grid 8), and blocking on it froze the app once a
  generation. `AutoTournamentService` submits `driver.precompute(...)` to a
  one-thread pool and `score_and_tell()` returns **None** until it lands, while
  `update()` keeps returning `Action.SCORE` because neither the phase nor the
  snapshot counter moved — so there is no new state machine. Worst frame gap
  1857 → 68 ms, generation 1.86 → 1.99 s. Three things this deliberately does
  NOT do: it does not move the rest of `tell()` (which mutates the archive the
  UI reads every frame, so it would need a lock around every one of those reads
  to buy ~5% more); it takes no locks in `VisionScorer` (ORT `run` is
  thread-safe, and a lock held for a 4 s vision pass would freeze `set_prompt`);
  and it passes a **copy** of the frame buffer, because `abort_generation()`
  clears the list.

- **The expedition fitness is contrastive, and its logit scale is
  per-modality** — `TEXT_LOGIT_SCALE` is CLIP's own 100, `IMAGE_LOGIT_SCALE` is
  30. Image-image similarity sits above 0.9 where 100 is far too sharp: a +3sd
  latent goal at scale 100 floors 59.6% of tiles to zero, leaving 10.6 of a
  16-tile generation distinguishable to a rank-based optimizer. `contrastive()`
  has no default scale on purpose.

- **A latent or chase goal has ONE reference, which makes its fitness a
  monotone squash of raw cosine — and raw cosine to an arbitrary direction is
  maximised by NOISE.** Text goals were never exposed because
  `DEFAULT_DISTRACTORS` carries "random noise" and "an abstract texture". The
  fix is `capture_health.structure()` as a multiplicative factor on every
  expedition fitness, which took a latent goal's top pick landing in the
  archive's least-coherent decile from 38.0% to 18.8% (10% by chance). It is
  deliberately NOT a CLIP term: the distractors cannot simply join a latent
  goal's reference set, because image-image similarity sits near 0.9 and
  image-text near 0.2, so at one logit scale the text references contribute
  only a constant, which a softmax is invariant to. Applied to text goals too,
  where it barely moves ranking, because one rule beats two.
  **It must look at more than the neighbouring pixel.** Lag-1 autocorrelation
  alone scored a 3px and a 4px lattice at **0.000 — exactly what white noise
  scores** — a false negative on precisely the fine, complex creatures the
  factor must not punish. Best |ρ| over lags (1,2,3,4,6,8,12,16) and each axis
  separately takes noise to 0.017 and both lattices to 1.000. Nothing here is
  temporal: `structure` never compares frames, so a slow pattern scores exactly
  like a fast one. Motion is `liveness`.
  **The SEED was not the problem, and the obvious fix makes it worse.** Seeding
  a latent expedition at its own anchor measured 15.8% → 27.0%, because
  `_seed_index` samples with `banded_alpha` rather than argmaxing, while
  `p ~ NOV^4` concentrates hard and novelty is itself mildly rough-biased. Only
  `novelty_goal` sets `Goal.seed_index`, because it has no embedding to score
  the archive against.

- **A novelty expedition is the third goal kind, and the only thing that
  optimises novelty WITHIN a generation.** Expansion samples parents by novelty
  then mutates blindly; `novelty_goal` makes the generation a CMA-ES hill-climb
  on the same kNN novelty that decides admission and parent choice, with no
  target that might be unreachable. Its objective is deliberately
  non-stationary — admitting entries during the expedition lowers the novelty
  of everything near them — so the covariance adapts on shifting ground.
  `novelty_share` and `latent_share` are clamped rather than normalised
  (rescaling one because the other moved would make neither mean what it says),
  and `_draw_goal` falls THROUGH the kinds rather than failing, because an
  expedition that does not start wastes a whole cadence interval. Novelty alone
  is not a noise cure — top-N by novelty is in the roughest decile 13–19% of
  the time — which is why the coherence factor applies to it as well.

- **Seed selection must use the SAME objective as the fitness.**
  `Archive.nearest()` is `argmax(embeddings @ goal)`, and for a text goal its
  winner is a noise texture: high-frequency noise carries a middling cosine to
  every phrase, so over 15 unrelated prompts it returned only 6 distinct seeds
  and one cyan static tile won 7 of them. `ImgepDriver._seed_index` uses the
  contrastive score instead (14/15 distinct); what rejects noise is
  `DEFAULT_DISTRACTORS`. The seed is then SAMPLED with `p ~ fit^alpha`, so
  repeating a goal explores a different trajectory.

- **The seed pool is a BAND on effective sample size, never a target.** How
  concentrated a goal's matches are is real information: over 20 prompts
  against 4784 entries, ESS at alpha=4 ran 3.1 ("a photograph of a cat" — the
  archive genuinely holds almost nothing cat-like) to 1973 ("circuit board
  traces"). Pinning ESS would tell the cat prompt it has 64 good seeds when it
  has three. `banded_alpha` only clips the ends. Two further traps it handles:
  the floor is capped at N/8, because a floor demands candidates that may not
  exist; and `effective_sample_size` works in log space, because `w**alpha`
  overflows float64 inside the bracket bisection explores.

- **`LATENT_DIMS` must stay small (8).** Whitening equalises the components, so
  a large d puts the push into geometrically tiny directions (d=32 was a no-op,
  seed rank 0.1 against 41.3 at d=8); and the goal lies wholly inside the
  d-dimensional subspace while entries keep most of their energy outside it.
  "Capture more variance" silently turns the push off.

- **`Archive.nearest()` returns `argmax(embeddings @ goal)`**, so an expedition
  always seeds on the archive's best entry under its own goal — fine only if
  the goal leaves room past it. The old `g = b + beta*(b - c)` did not: over 200
  trials the goal's nearest entry WAS its own seed 199 times, and no value of
  beta fixed it. See
  `docs/superpowers/specs/2026-08-08-expedition-objective-design.md`.

- **The descriptor and the expedition fitness are separate computations** over
  the same per-snapshot embeddings. `descriptor()` renormalises the trajectory
  centroid — right for novelty, which needs unit vectors — but `1/||m||` grows
  as snapshots decorrelate, so using it as fitness paid a bonus for *changing*
  rather than for *matching*. Score per snapshot, then average.

- **Expansion draws one parent PER TILE, independently and with replacement,**
  so the grid sets the number of draws and not the number of parents. At the
  default alpha=4 the two nearly coincide (15.5–15.9 distinct of 16 draws,
  58.0–61.9 of 64), but the slider reaches 8, where one archive's ESS is 2.2
  and 64 tiles collapse to ~16 distinct parents. Nothing else scales with the
  grid — `sigma_expand`, `alpha`, `k` and the refresh budget are all
  grid-independent. Expeditions are the exception: CMA-ES takes
  `popsize = tournament.tiles`.

- **`reset()` must ADVANCE `base_seed`, or Reset replays the last run exactly.**
  `base_seed` feeds two things — the optimizer's seed and `gen_seed =
  base_seed + generation`, which is the sim's particle seed — and `generation`
  goes back to 0. Fixed at 1000 it made Reset then Start hand back a
  bit-identical population, in identical tiles, over an identical particle
  field: the optimizer WAS being cleared correctly and the run replayed
  regardless, so a new prompt only reranked creatures already watched. The
  stride is `generation + 1`, read before the counter is cleared, so the two
  runs' `gen_seed` ranges cannot overlap either. The constructor argument still
  fixes the first run — `tools/brain_search_bench.py` passes one — and
  `load_checkpoint` restores it verbatim, because a resume must replay.
  Guarded by `tests/test_auto_tournament_service.py`.

- **A RESET's rule write outranks the phase gate; a GENERATION's does not.**
  Nothing rewrote the grid outside `_begin_generation`, which only runs from
  `start()`, so after Reset the abandoned search's creatures stayed on screen
  and on the GPU until Start — a button that appeared to do nothing. `reset()`
  therefore re-randomises `tournament` and sets `_force_write`, which `update()`
  answers whatever the phase. `_needs_write` stays BELOW the gate: the next
  generation's rules are queued the moment one is scored, so honouring that one
  while paused advances the picture to the next generation instead of freezing
  it. Guarded by `tests/test_auto_tournament_service.py`.

- **Z and G are SINGLE-BRAIN keys, and under a tournament they belong to the
  grid's owner.** Both went through `sim.apply_rule`, which writes SLOT 0 —
  tile 0, overwritten by the next generation — so under a grid they changed one
  square in the bottom-left corner and nothing else. `_grid_owner()` routes them
  by `tournament.enabled`, never by the sub-mode flags, because a closed
  Tournament window clears those; under Auto or Explore the owner is the
  optimizer, since re-randomising the tiles alone would be undone by the next
  generation. G keeps setting `rule_seed` in every mode — that IS the fresh crop
  of mutations — and only the slot-0 write is dropped. Guarded by
  `tests/test_grid_reset_keys.py`.

- **Explore mode reuses Auto mode's `AutoTournamentService` instance**, swapping
  only `.driver`. Both `_handle_auto_tournament` and `_handle_explore` would
  otherwise call `configure()` on the same object every frame, so each bails out
  when the other owns the driver.

### Audio input

- **The window sets FREQUENCY resolution and the HOP sets time resolution, and
  a transient is a hop problem, not a width one.** FFT_SIZE is 2048 and HOP is
  512. Narrowing the window is the obvious fix for a soft transient and it is
  the wrong one: 1024/512 buys a hi-hat peak of 0.812 against 0.749, and pays
  for it with double the bin width, where the bass band (20–250 Hz) has only
  about ten bins to begin with. Halving the hop instead costs one extra FFT per
  block and nothing else.

- **A band SUMS LINEAR ENERGY and takes decibels once; averaging each bin's dB
  level is what stopped it ever reaching zero.** The old measure mapped every
  bin onto `[-90, -20]` dB and averaged the results over the band, so the
  hundreds of bins carrying nothing set the answer — `hi` spans about 600 bins
  at 2048/48k and a cymbal lights a handful. Over the calibration material
  every band read **0.17–0.18 with only room hiss playing** and sat at 0.25–0.44
  between hits on a loud track: a rig that modulates hardest when nothing is
  happening, and a `phase` shaper that never stops travelling. `power`
  (`10·log10(Σ|X|²)`) is the default and reads **0.00** on both room rows.
  `rms`, `peak` and `mean_db` are the alternatives, per band, and `mean_db` is
  kept only so a rig built against it still plays.
  **Each measure carries its OWN dB window, because they are not on one scale**
  — a sum over 600 bins is not a mean over them — and the floor is set ABOVE a
  quiet passage rather than above the noise floor, since a quiet part of a
  track passing a healthy signal is the whole complaint. Nothing here adapts:
  a running floor or a running peak makes the same sound read differently
  depending on what played before it, which is exactly the auto-gain defect one
  caveat down. `python -m tools.measure_audio_response` prints the table the
  windows come from; the room rows must read 0.00.

- **The five loudness signals ask ONE question in five ranges, which is why a
  rig could not follow a song's arrangement.** Two answer different questions.
  `flux` is a MEASURE (`Σ max(0, |X| - |X_prev|)` over the band), so a
  sustained note reads **0.00 however loud it is** and only the attack
  registers — `bass`/flux is a kick and `hi`/flux is a hi-hat, with no new
  signal names and the same window machinery. It is deliberately level-
  dependent like the other measures; a ratio would put a quiet passage's hits
  at full scale, which is the complaint the measures answer. Its floor sits
  **10 dB above the levels'** because broadband noise is new every block, so a
  room's hiss fluxes in `hi` where it barely registers in `power`.
  `centroid` is a SIGNAL — the energy-weighted centre of the spectrum, read on
  a LOG axis over a window in HERTZ, not decibels. It is a ratio, so it says
  nothing about level: a quiet bright break reads high and a loud bass-only
  drop reads low, which no band can tell apart. Three consequences: auto-gain
  must skip it (dividing a ratio by its own running peak means nothing and
  would put every bright moment at 1.0), it starts at 0 so a rig that has
  heard nothing contributes nothing, and it must be HELD below
  `CENTROID_GATE`.
  **The gate is the whole feature, and "any signal at all" is not a gate.**
  Being a ratio, the centroid reports a room's noise floor at full strength
  however quiet the room is — the −60, −50 and −40 dBFS rows read identically
  — and a noise spectrum is a fresh random draw every block, so it WANDERS:
  swing 0.14 on pink and 0.36 on brown, 0.06 per block, with nothing playing.
  The first version gated on `volume > 0`, which a −60 dBFS room already
  clears at 0.08. The gate is 0.25 of `volume`'s own scale, which is measured
  to sit above a loud room and below quiet music; expressing it as a fraction
  rather than in dB means `volume`'s floor slider moves it. The slow envelope needed no new
  machinery at all — `smooth` reaches 30 s now, on a LOGARITHMIC track,
  because every percussive setting is under a second and would otherwise share
  the first pixel.

- **The band smoother is ASYMMETRIC, and the rise is not smoothed at all.** The
  two directions solve different problems: falling slowly is what stops a
  steady note drawing a fuzzy hash, while rising slowly only costs the
  transient. A hi-hat decays in a few milliseconds, so the symmetric 75 ms
  one-pole reported a peak of 0.315 where the signal was 0.749 — it was not
  delaying the highs, it was eating 58% of their height. `ATTACK_SECONDS` is
  therefore 0, `SMOOTHING_SECONDS` 0.075, and the cost is measured in the other
  column: the per-analysis step on a steady note goes 0.00085 → 0.00109, still
  23x calmer than the unsmoothed 0.0256 that made the traces vibrate. Both
  numbers must be read together — `python -m tools.measure_audio_response`. A
  mapping that wants a soft attack asks for one with the `smooth` shaper; the
  analyser cannot give a snap back that it has already thrown away.
  The release is now a SETTING (`AudioInState.release_seconds`, default 0.075)
  and **0 hands every shaper the raw per-block measurement** — the point of the
  shapers is to do the smoothing, so the analyser has to be able to stay out of
  it. It applies to the BAND, last, after the measure and the auto-gain, so it
  means the same thing whichever measure produced the number; the DISPLAY
  spectrum keeps its own fixed smoothing, because the bars are there to be read.

- **The DISPLAY spectrum's dB window is not a band's.** The mel bars are read
  over `[-90, -20]` dB, which rests ordinary material across the middle of the
  scale — right for something to look at, and the reason a band measured off
  those bars could not reach zero. Bands carry their own windows
  (`MEASURE_WINDOWS`), and `volume` measures the whole block at once, which
  sits far above any single bin, so it has its own `[-60, -6]`. Raw linear
  magnitude is what the first version drew and it fails twice over: every band
  pins at 1.0, and the mel rows — unnormalised triangles whose width grows 20x
  from the bottom of the axis to the top — draw any spectrum at all as a ramp
  rising to the right. The rows average instead.

- **A mapping's shaper state is keyed by a `uid` the `Mapping` carries, never
  by `id()`.** CPython hands the address of a freed object straight to the next
  one of its type: over 2000 create/delete cycles of the real class, 1999
  reused an address just released. So a row added after one is deleted
  inherited the deleted row's envelope and LFO phase — starting mid-attack for
  no visible reason — and `shaped` aliased the same way, letting a drawer draw
  another row's trace. `uid` is an ordinary dataclass field, so it takes part
  in `__eq__` and a rig diffed by value sees a delete-and-re-add as the change
  it is; it is copied with the mapping and never persisted, so a loaded rig
  mints fresh ones. Nothing tells the runtime a row was deleted, so
  `AudioRuntime._prune_states` cuts the table back every frame — to the WHOLE
  rig, not the modality on screen, or switching brains and back would restart
  the shapers that were waiting there. Guarded by `tests/test_audio_mapping.py`
  and `tests/test_audio_runtime.py`.

- **NO shaper may generate MOTION from silence, which is NOT the same as
  answering silence with zero.** A latched sample-and-hold and a stopped
  integrator both sit on a perfectly good non-zero value; what none of them may
  do is MOVE while the band is dead. The old `lfo` did: it drove the rate from
  the band but floored that rate at `rate_min` (0.5 Hz), so a dead-zero band
  still swung the full 0..1 forever — a rig that modulated hardest with the
  music off. It was deliberate, and
  `test_lfo_runs_even_when_the_signal_is_silent` asserted it. The FLOOR was the
  defect; driving the rate from the band was right. Guarded by
  `test_no_shaper_moves_on_its_own_when_the_band_is_silent`, derived from
  `SHAPER_KINDS` so a kind added later cannot skip it — and asserting a
  TOLERANCE rather than equality, because `smooth` converges on its target
  instead of arriving and is still creeping by ~1e-18.
  **A HELD signal is the second way in, and the first guard did not see it.**
  That test drives an input of exactly zero, but `centroid` parks at whatever
  the last music was, so a paused track left `phase` integrating 0.73 forever
  — the free-running LFO again, by another road. `SignalSnapshot.held` names
  the signals reporting a remembered value, `AudioRuntime` passes it to
  `modulate`, and `ShaperState.apply` takes `live`. Holding must stop MOTION
  without silencing the signal: the parameter has to stay where the music left
  it, which is the whole reason the centroid holds rather than diving.

- **`phase` INTEGRATES the band: `dphase = band * rate * dt`, so it only ever
  lurches forward.** The band sets how fast the wave TRAVELS, never where it
  sits — which is the whole difference from a waveshaper that maps level
  straight onto phase. A held note therefore keeps it cycling rather than
  parking it, and silence stops it DEAD WHEREVER IT HAD GOT TO rather than
  dragging the parameter back to base. Measured over a track that plays, stops
  and resumes: loud (0.9) travels a full sweep, quiet (0.2) travels 0.345 of
  one in the same time, silence holds at exactly 0.345 with zero travel, and
  the next loud passage carries on from there. `rate` is cycles per second at a
  full-scale band, which is literally the old `rate_max` — so a stored `lfo`
  row carries its rate straight across and loses only the floor. Two things are
  load-bearing: the advance is scaled by `dt` and NOT per frame, or a slow
  render moves the wave less per second of music than a fast one; and `_wave`
  starts at ZERO for all three shapes, so a phase still at 0 — a rig that has
  heard nothing — contributes nothing.

- **`rate_scale` multiplies the RATE, never `dt`.** One knob slides a whole
  rig onto another tempo, which is what matching a beat needs; scaling `dt`
  instead would reach every attack, release and hold as well, and those are
  durations. So `phase` is the only kind that notices it, and the guard
  against generating motion from silence still holds at any setting — the
  band is still a factor in the advance. The traces need nothing of their
  own: the drawer plots `shaped`, which IS the shaper's output.

- **Auto-gain divides each band by a peak the signal REACHES, so any steady
  input normalises to its own top — it ships OFF.** `np.maximum(peaks, raw)`
  puts the peak at the signal, and the 0.9995/block decay only matters on the
  way down, so the output for steady material is 1.0 by construction whatever
  its level. Measured on `mid`: a −50 dB room hiss reads raw 0.229 and gained
  **0.962**; a −70 dB hiss reads raw 0.003 and gained **0.554**, swinging the
  whole range; real material (a −12 dB kick loop) reads 0.735. There is no gap
  between a room and a track, so the feature cannot be tuned into correctness —
  `AudioInState.auto_gain` is therefore `False` and the checkbox is the escape
  hatch. Only digital silence is safe, which is why a loopback endpoint with the
  music off looks clean and a microphone does not, and why this survived:
  `tools/measure_audio_response` passed `auto_gain=False` in every case. It now
  prints the table. The checkbox is pushed to the analyser EVERY frame rather
  than at Start, or the only way out needs a Stop/Start; the peaks are dropped
  on the switch, since they record a level that has gone.

- **The audio brain's base scales come from the LIVE layout, and its `z` is
  never re-encoded on a scale change.** The Brain window's scale sliders and
  the rig write the same slot, and `_apply_brain_layout` handles a scales-only
  change with `sim.set_brain_scales` — which re-decodes from the sim's stored
  `z` and does NOT touch `rule_manager`. So the rule the rig is handed is the
  pre-drag one, and anchoring to the layout captured when audio adopted the
  brain put that brain back every frame: dragging MLP Weight Scale 1.0 → 4.0
  wrote p90|w| 1.5895 and the rig overwrote it with 0.4141. Re-encoding is NOT
  the fix and makes it worse — `decode(encode(P, new), new)` is `P`, the very
  brain that was on screen before the slider moved. Keep the first `z`, take
  the scales off the layout that is live now. `layout.length` is in
  `_base_brain_id` so a WIDTH change still forces a re-encode; the scales
  deliberately are not.

- **The rig file is SHARED by every copy of Fluoddity, so a session writes it
  only if it CHANGED it.** `Documents/Fluoddity/audio_rig.json` is one file for
  every worktree and every instance another session launches, and the write at
  exit used to be unconditional — so a second copy opened and closed without
  going near audio put its empty rig over the one you had just built, which
  reads as a save that only sometimes works. `App._rig_at_start` holds the dict
  `load_rig` produced, and `_save_last_rig` compares `to_dict(audio)` against
  it. Two consequences: a first-ever run that touches nothing never creates the
  file, and two instances that BOTH edit still resolve last-writer-wins, which
  is not fixable — two rigs cannot be merged. Polling was considered and
  rejected: it costs almost nothing but guards only a hard kill, which is the
  one thing that skips `cleanup()`. Named rigs are a separate thing entirely,
  in `audio_rigs/` rather than the user configs folder, because everything
  there appears in File > Load and a rig is not a physics config. Guarded by
  `tests/test_rig_presets.py`.

- **`imgui.ini` is shared between the app and the test suite, and the tests
  must not read or write it.** Dear ImGui persists every window's size in
  `create_context`/`destroy_context`, so the tests saved a layout and consumed
  it on the next run: the archive gallery came back 382px tall, two of its
  eight entries no longer fit, and a test that had always passed began failing
  with no code change. Running the app writes the same file, so the suite's
  result depended on whether anyone had resized a panel. `tests/conftest.py`
  wraps `create_context` to null the filename AND calls `ui.ini_path.suppress()`
  — the app sets the filename during `UI.__init__`, which runs after
  `create_context` and would otherwise put the suite back on the real file.
  That is why the set lives in ONE function and nowhere else. Never read it
  back — `get_ini_filename()` on the null segfaults.

- **The layout file is USER data, and which windows were open is not in it.**
  `imgui.ini` used to sit in `get_app_dir()`, which is the launched folder
  running from source — so five worktrees kept five layouts — and is
  potentially read-only in a packaged build under Program Files. It is now
  `Documents/Fluoddity/imgui.ini`, seeded by `migrate_imgui_ini` from an
  existing app-dir layout before `default_imgui.ini`, and never overwritten
  once it exists. ImGui records each window's position and size but not
  whether it was on screen at all, so the open/closed flags live in
  `PreferencesState` — which round-trips through `asdict`, so a field there
  persists for free. `show_demo_window` is deliberately excluded. The two
  windows whose flag belongs to a feature (`state.audio.show_window`,
  `state.archive.show_browser`) are MIRRORED rather than moved: `get_state()`
  copies state to preference every frame and `_restore_open_windows` applies
  it once at startup. Restoring the browser sets `open_browser_requested`,
  which reloads the archive — `rescore_all()` is load-bearing there, so it
  must be asked for exactly ONCE and never per frame.

### Recording

- **`generate_view_texture()` returns two different things, and the recorder
  gets whichever arrived.** The camera views (Camera, Tiled, Particles+Trails —
  `cam_brush_mode`) hand it `cam_brush_target`, a FRAMEBUFFER-sized buffer with
  the camera already baked in; every other view hands it `sim.view_tex` at
  canvas resolution with no camera applied. `FrameAssembler` sizes itself from
  its input, so the assembled texture inherits that shape. Black bars exist
  only in the first case, and so does zoom's cost: the world is rasterised into
  however many pixels the zoom leaves it, so **no crop can restore resolution a
  zoom-out never drew**. Recovering it needs the recording to own its
  framebuffer and camera — a second particle raster per frame — which was
  considered and rejected. Framing near fit-to-window is the user's half of the
  bargain.

- **The crop RECT is re-derived every frame; the crop TARGET is frozen at
  record start.** The encoder rejects a mid-stream dimension change, so a zoom
  rescales into the fixed target rather than resizing the file. The rect comes
  from `camera.assembled_view_rect` and is never recomputed — that field is
  assigned with the texture it describes, and recomputing it later crops one
  frame's pixels with a later frame's camera. It is intersected with the
  texture, so zoomed in the crop is an identity and framing is untouched.
  Because the target no longer tracks the window, a mid-take resize stops
  splitting the file. `record_sizes` derives the video size in OUTPUT space:
  cropping first and dividing by the supersample kernel after can land on an
  odd number, which the recorder pads — putting a black edge back on the side
  the crop just removed. Guarded by `tests/test_record_view_gl.py`, whose rects
  are all off-centre, because a centred camera cancels the `v`-flip and
  validates any orientation bug you like.

- **The SOUNDTRACK is the recording's clock, and wall time is only a
  cross-check.** The tap starts and stops with the recorder, so both streams
  cover the same wall interval and `frame_count / audio_seconds` is exactly the
  rate that makes them the same length — sync is arithmetic rather than a
  measurement, and it rides the sound card's clock rather than the frame
  loop's. This is why the sidecar is muxed at close rather than piped live:
  `-framerate` is fixed when the encoder starts, and the real rate is not
  knowable until the take ends, so `-itsscale` retimes it during a stream copy.
  `-itsscale` is an INPUT option and only affects the input it PRECEDES. Wall
  time catches a device that died mid-take, where the audio is far too short
  and the derived rate would silently speed the video up to match it.

- **Because the clock is honest, hitting real time is a taste decision, not a
  correctness one.** With audio the `speedmult` override in `main.py` is
  skipped and the rate follows the user's slider: blur samples ARE physics
  sub-steps, so forcing them up renders far below real time. Raising the slider
  trades fps for blur and sync is unaffected. Silent recording keeps the fixed
  50 fps and its exact previous behaviour.

- **The soundtrack delay is the `adelay` FILTER, and `-itsoffset` is the trap.**
  In front of a headerless raw input `-itsoffset` is accepted and shifts
  nothing: the file still plays, so the only symptom is that the control does
  nothing. What is left in the output is the AAC encoder's own priming delay,
  0.021 s, identical at every setting — which reads like a working control with
  a bad scale factor. An argv test cannot see this, so
  `tests/test_record_delay_e2e.py` decodes the muxed audio and finds a click.
  `adelay` takes MILLISECONDS and needs `all=1`, or only the first channel
  moves. The picture lags the sound it reacts to — analysis block, the band
  smoother's release, a frame, and the readback's frame — so delay is the
  direction that matters and the slider does not go negative.

- **The tap is guarded SEPARATELY from the analysis, not by the same
  `try`.** Sharing it means a full disk stops the signals driving the sim.
  Cleared before teardown in `stop()`, as `_analyzer` is, so a callback in
  flight cannot write into a file the main thread is closing. Every path
  through `RecordingAudio.finish()` leaves the output file in place — a
  soundtrack is worth strictly less than the recording it belongs to.

- **Nothing on the frame loop may touch the pixels, and `stdin.write` may not
  happen on it at all.** Recording used to cost **44.1 ms per frame** at
  1920x1080 before ffmpeg saw anything — a `read()` of an RGBA **float32**
  target (16.6 ms, 33.2 MB, and a synchronous readback stalls the whole
  pipeline) then `*255 -> clip -> astype -> flipud -> tobytes` on the host
  (27.5 ms, four passes over eight million floats). It then blocked on a pipe
  to an encoder at preset `slow`, whose **worst single write was 1118 ms**.
  Three fixes, all needed: an RGBA8 target read through a PAIR of buffers
  mapped a frame late (a synchronous `texture.read_into(host)` is 24.5 ms; via
  a PBO it is 2.5); the flip done in the vertex shader so the readback IS the
  `rgba` ffmpeg is given, with no host pass; and a writer thread, so a full
  pipe never reaches the caller. Now **8–15 ms mean**.
  `python -m tools.measure_recording`. The trap when re-measuring is the one
  the canvas-format caveat names: this is a laptop, so two runs measure its
  thermal state as much as the code. Read the columns against each other in
  ONE run, never across runs.

- **The encoder must DRAIN faster than the sim produces.** At 1080p, preset
  `slow` drains 32–48 fps with a worst single write over a SECOND; `veryfast`
  drains 64–84 with a worst write of 29 ms. `PRESET` is therefore
  `veryfast`/CRF 20. The queue is bounded by MEMORY (256 MB, 4..32 frames) and
  BLOCKING: a full queue slows the sim rather than dropping a frame, so the
  file is exactly what the sim produced — which is also what lets
  `recording_fps` divide by it.

- **The readback is a frame LATE, so `finish()` collects the one in flight —
  but never past `max_frames`.** The frame count is what the soundtrack's
  length is divided by, so losing the last frame would stretch the take and
  keeping one too many would run a limited take long. `VidSaver._max_frames`
  is remembered for exactly that check. Guarded by
  `tests/test_vid_saver_audio.py::test_hitting_the_frame_limit_still_muxes`,
  which caught the extra frame the drain first introduced.

- **RGB8 is smaller and NOT a required color-renderable format.** It works on
  this machine (2.55 ms against RGBA8's 3.09, and 6.2 MB over the pipe against
  8.3) and it is the same trap as RGB32F elsewhere in this file. RGBA8 is
  required, ffmpeg takes `rgba` natively, and the difference is 79.6 fps
  against 76.1 — so the format is RGBA8 and the alpha is ffmpeg's problem.

- **`VidSaver._plan` is the ONE authority on the output size, and it rounds
  DOWN to even.** H.264 refuses odd dimensions, and the old path rebuilt the
  whole image into a padded host array every frame to fix it. `record_sizes`
  already produced even sizes for a cropped take; the uncropped branch now
  does too, and `FFmpegVideoRecorder` raises on odd rather than accepting what
  it cannot encode. The reader reaches the same number by the same rule — a
  frame whose size disagreed with the encoder's header would be refused.

- **The flip lives in the vertex shader, so both capture paths share it.**
  `save_frame_gpu` (screenshots, synchronous, PNG, exact dimensions) and
  `AsyncFrameReader` (recording) run the same program; a leftover `np.flipud`
  on either side would put that one upside down silently. Guarded by
  `tests/test_video_recording.py::test_the_screenshot_path_agrees_with_the_recorder_on_which_way_is_up`.

### UI and platform

- **There is ONE save dialog, and every Save button in the app opens it.**
  A save is a *subject* — `(kind, arg, tiles)`, see `services/save_targets.py` —
  and `UI.open_save_popup()` is the only way to start one; nothing writes a
  config without asking for a name first. `CommandHandler._handle_file_save`
  dispatches on `kind`, where `""` is the ordinary File > Save and must keep
  behaving exactly as it did. Everything lands in the same user configs folder
  and appears under File > Load > Custom, because a second folder is a second
  place to look. Three rules this encodes, each from a real failure: several
  selected tiles get **suffixed** names (`reef_tile1`, `reef_tile4`) since one
  shared name reproduces the clobbering inside a single click; the overwrite
  check covers **every** target, not just the first; and the typed name goes
  through `safe_stem()`, because the old path interpolated it straight into a
  filename, so a separator wrote outside the folder and then did not appear in
  the load menu. An Auto **checkpoint** is deliberately outside all of this —
  it resumes the optimizer, is not a config, and stays in the run folder.

- **A save that only prints to the console reads as a no-op.** The file lands
  somewhere not on screen, so every save path sets a `notice` (or `warning` on
  failure) rendered by `ui/notices.py`. Pass a distinct `scope` — two tabs both
  showing a "Dismiss" button would otherwise collide on the ImGui id, per the
  caveat below.

- **Browsing an archive costs no CLIP, and `_open_archive` is where that line
  is drawn.** Extras > Archive Browser loads the entries, thumbnails and
  projection and nothing else; `_ensure_archive_service` calls the same
  (idempotent) helper and only then builds the scorer and the driver. Two
  consequences worth keeping: opening the gallery does not pay the ONNX
  session cost, and the browser still works where the optional packages are
  absent — which is the same rule manual tournament mode already follows. A
  reload is what must not happen twice: `load_from_store` rescores everything.

- **A closed Tournament window must turn its sub-modes OFF, and only the tab
  that is drawn can do that.** `render_tournament_window` returns early when
  the window is shut, so neither tab runs and neither clears its own
  `enabled` — leaving the app applying Auto/Explore's overrides (forced grid,
  square tiles, no motion blur) to what the user sees as an ordinary single
  simulation. The early-return branch clears both. This is what makes the
  archive browser usable as a plain browser with the tournament closed.

- **The browser's live preview pushes onto the SAME rule stack as
  File > Load's preview, and `pop_rule()` returns `(None, None)` when the
  popped rule was the only one.** Handing that `None` to `sim.apply_rule` is
  not a restore. `preview_entry_id` is deliberately a CONTINUOUS field, not a
  one-shot: the UI writes whatever the pointer is over each frame (`-1` for
  nothing, set before every early return in `render_archive_window`) and
  `CommandHandler` diffs it against what it is already showing. A one-shot
  would need the UI to track transitions itself, and a browser that stops
  being drawn would strand the preview. Physics are snapshotted before the
  FIRST push and restored whole — restoring per-entry would put back the
  previous *entry's* sliders rather than the user's.

- **The gallery's grid has a 48px FLOOR, and `ThumbCache.reserve` is the other
  half of it.** The size slider runs 16..160; at and below `GALLERY_LIST_MAX`
  the gallery draws a LIST rather than a grid, one thumbnail per row, which the
  clipper caps near 30 however the window is sized. That floor is what bounds
  the grid: a frame touching more than the cache's `capacity` evicts every
  texture and re-decodes the whole visible set on the next one, forever, and at
  32px in a wide window the grid would ask for 400+. Neither half works alone -
  even at the floor a 2560-wide window asks for around 350, past the historical
  256 - so the gallery calls `reserve()` with the count it is about to draw,
  BEFORE drawing any of it, and the cache raises capacity to `MAX_CAPACITY`.
  The sort modes come from `services/gallery_sort.GALLERY_SORTS`, read by the
  Sort combo AND by the list's column headers, which both write the one
  `sort_by`/`sort_desc` pair: a second list of modes is what would let a header
  and the combo mean different things, and switching view would reorder the
  gallery under the pointer. Column visibility, width and order are ImGui's own
  table settings and live in `imgui.ini` - which is exactly why the THUMBNAIL
  column is `no_resize`. A resizable column's width is restored from that file,
  so the width handed to `table_setup_column` only ever applies to a table that
  has none: the pictures grew with the slider and the column they sat in did
  not. `imgui.internal.table_set_column_width` is NOT the way out - it asserts
  on both `IsLayoutLocked == false` and `MinColumnWidth > 0`, and the same call
  sets both, so it cannot be reached from inside the table it would resize.
  Guarded by
  `tests/test_thumb_cache.py`, `tests/test_gallery_sort.py` and
  `tests/test_archive_window_render.py`.

- **A canvas that reads the wheel must CLAIM it, and no child flag can do
  that.** ImGui routes the wheel during `NewFrame`, before any of our code
  runs, so a canvas reading `io.mouse_wheel` to zoom also scrolls the panel it
  sits in. `no_scroll_with_mouse` is exactly the wrong instrument — it is
  ImGui's *"give the parent a chance to scroll"* flag, so the pair
  `no_scrollbar | no_scroll_with_mouse` GUARANTEED the forwarding it was
  credited with preventing, and the map zoomed and scrolled together for as
  long as that was believed. `_draw_map` calls
  `imgui.set_item_key_owner(imgui.Key.mouse_wheel_y)` on the `map_hit` item
  instead, which claims the wheel while that item is hovered and leaves every
  other panel scrolling normally. The child and `no_scrollbar` stay, for
  clipping and for not drawing a scrollbar — neither has anything to do with
  the wheel. **A flags assertion cannot see any of this**: the old test read
  the flags and passed throughout. `tests/test_map_wheel.py` DRIVES the wheel,
  and it needs two things a naive version gets wrong — the panel must be
  parked MID-scroll, since at scroll 0 a wheel-up has nowhere to go and a
  broken build passes, and the mouse position must be resolved AFTER the
  scroll, since scrolling moves the canvas out from under a point computed
  before it.

- **A control the user has to FIND cannot live in a folded section, and a
  render test cannot see one either.** ImGui clips a window's contents to the
  WINDOW, not to the display, so `frame()`'s host in
  `tests/test_archive_window_render.py` must be sized taller than the tab — at
  the default size the whole settings column falls outside it and draws no
  vertices, which turned "opening the sections drew more" into a coin flip on
  two vertices of header arrow. The encoder combo shipped inside the
  default-closed `Admission` header and Auto's inside the weights-missing
  branch: both rendered, neither was reachable, and every source-level reading
  of the tab said they were. Assert on the labels a real frame DRAWS
  (`_combo_labels`), not on where the call sits.

- **A widget's label is drawn to its RIGHT and is CLIPPED, not scrolled.**
  ImGui's default item width is 65% of the window, so at any narrow width the
  label runs past the edge and simply vanishes — there is no horizontal
  scrollbar to find it with. Every settings panel therefore pushes
  `layout.push_settings_width()`, which reserves room for `WIDEST_LABEL`, and
  button rows use `layout.wrap_row()` rather than bare `same_line()`. A label
  wider than `WIDEST_LABEL` fails `tests/test_label_widths.py`, measured in
  real pixels — the font is proportional, so character counts do not predict
  width.

- **A POPUP BODY only runs while the popup is open, so a render smoke test
  never enters one.** Every widget call inside `begin_popup*` is unexecuted by
  an ordinary test pass, and imgui_bundle's bindings raise `TypeError` on a bad
  signature rather than failing to compile — `imgui.selectable(label)` is
  missing its `p_selected` and takes the whole app down the first time a user
  right-clicks. A test opens the body by wrapping `begin_popup_context_item`
  and calling `imgui.open_popup(str_id)` first: both hash `str_id` against the
  same window and ID stack, so this reaches a popup nested inside a `push_id`.
  Pair it with a test that the wrapper opened something, or the coverage is
  imaginary. Guarded by
  `tests/test_audio_reactive_window_render.py::test_the_forced_popup_helper_really_opens_something`.

- **An ImGui widget's identity IS its label, and a duplicate silently kills the
  loser.** Two visible items hashing to one ID puts Dear ImGui's "conflicting
  ID" dialog over the app and stops one of them responding to the mouse at all
  — it is not a warning. A `collapsing_header("Archive")` shipped in the same
  window as `combo("Archive")`, and the control picking which archive the search
  writes into could not be clicked. `##suffix` keeps the visible text and
  changes the ID. Guarded by
  `tests/test_archive_window_render.py::id_clashes`.

- **The map's crowding is the problem, not its projection — and its dot
  density is NOT the archive's density.** Measured over the real archives, the
  2-D PCA holds 46% of the variance at *every* size, but kNN(10) preservation
  falls from 27.0% at 500 entries to 3.1% at 13049 purely because more points
  share the same pixels. Meanwhile the true distance to the ten entries the map
  puts nearest is flat at ~0.038 — nearly double `min_separation` (0.02) — so at
  full size two touching dots are on average further apart than the distance at
  which the archive calls two entries different. Never read local density off
  dot overlap; `novelty` is the real measure and it is computed in full 512-d.
  `services/map_view.py` therefore offers filtering (fewer points), a novelty /
  liveness colour ramp, and a log-scaled density heatmap — all opt-in, with the
  historical scatter as every default. Filtering renormalises to the filtered
  subset, so narrowing also expands what is left. **Crowding is not the only
  complaint, and this caveat used to claim it was**: PCA also fails to
  SEPARATE, which no amount of filtering answers — see the UMAP caveat below,
  which is how that half is fixed. PCA remains the default because its
  `transform()` is a matmul, so a new entry places instantly against stable
  axes.

- **UMAP lays out a SNAPSHOT; kNN places everything admitted after it.** PCA
  holds 46% of the variance at every archive size, which through two linear
  axes of a CLIP space does not SEPARATE - a different complaint from crowding,
  and the one filtering cannot answer. Over six archives UMAP roughly DOUBLES
  kNN(10) preservation at every size (`default`: 13.6 -> 29.2% at 500,
  4.8 -> 9.0% at 2000, 2.2 -> 5.2% at 4800), for a fit of 1-7.5 s.
  `python -m tools.measure_map_layout`. Note the absolute number falls with n
  for BOTH engines - two dimensions cannot hold a 512-d neighbourhood - so read
  the engines against each other at one size, never across sizes.
  `umap.transform` is never called and no reducer is pickled: entries admitted
  after a fit are placed at the mean 2-D position of their nearest neighbours
  in the FULL space, so `<archive>/map_layout.npz` holds only numbers and
  cannot break across a library upgrade. **A refit is seeded and aligned or it
  is useless** - `init=` the previous layout, then Procrustes onto it, because
  UMAP rotates AND mirrors between fits and a rearranged map is a new map;
  that, not fit time, is what made UMAP unusable the first time it was tried.
  It refits on exactly three triggers - no cache, 25% growth, or the Relayout
  button - never on admission. **Relayout reaches PCA too**, or the button does
  nothing under the default engine. The fit runs on a worker and is handed a
  COPY, and the map keeps drawing what it has until one lands, so `_drawing`
  falls back to PCA rather than to a blank canvas. Positions are keyed by
  `(layout signature, id)`: an id is unique only inside one brain's directory
  and the map pools every layout. A cache from another encoder is discarded,
  since at equal width a foreign vector is silently wrong.

- **The map atlas REPLACES the scatter, and it is binned in UNIT space.** Two
  things were wrong in the first version and both were reported as "clunky".
  It drew dots as well - under the pictures, where they are invisible and still
  cost a draw call - and a cell whose thumbnail had not arrived kept its dot,
  so pictures and points visibly fought each other on every pan. Nothing is
  drawn for a cell that has no picture yet. And cells were binned in SCREEN
  space, where a ONE-PIXEL PAN moves every boundary, changes which entry wins
  each cell, and reshuffles the whole atlas under the pointer.
  `quantised_cell` picks a POWER-OF-TWO cell in unit space whose on-screen size
  lands in `[px, 2*px)`, so the assignment is invariant under pan and under
  zoom within a level, and `_atlas_plan` caches it until a level boundary is
  crossed. Cost is then FLAT in archive size and below the plain scatter's:
  0.52 ms against 5.92 ms at 13049 entries, because it draws a few hundred
  images rather than 13049 circles. Only `ATLAS_NEW_PER_FRAME` thumbnails are
  DECODED per frame - a JPEG decode is ~1 ms and crossing a level renews every
  cell at once - and `ThumbCache.peek` is what keeps that budget off the
  resident ones. `reserve` is still called with the visible-cell count, for the
  reason the gallery calls it.
  **It is a CHECKBOX that disables Colour and Draw, not a Draw mode.** Pictures
  carry their own colour and their own marks, so both combos are dead while it
  is on; leaving them live is a control that cannot do what it offers. The
  earlier attempt made it a fourth `RENDER_MODES` entry on the grounds that
  Colour and Draw must stay orthogonal - that rule is about the DENSITY
  heatmap silently overriding the Colour combo, and it does not apply to a
  layer that replaces the marks outright.
  **The working set MUST fit the cache, and the visible cell count is set by
  the SLIDER and the canvas - not by the zoom or the archive.** It is
  `(canvas_w/px) * (canvas_h/px)`, so at the 16px end of the slider a map asks
  for ~1700 cells against `MAX_CAPACITY` 1024: the atlas then evicts the cells
  it just drew and re-decodes them for as long as it is open, which reads as a
  sweep of thumbnails flashing forever. `atlas_winners` therefore takes a
  `budget` (`ATLAS_BUDGET`, 640) and keeps the most novel cells, so the map
  goes SPARSE rather than churning, and `ATLAS_BUDGET + ATLAS_HEADROOM` must
  stay under `MAX_CAPACITY` - raising one without the other puts the flashing
  straight back. The headroom exists because `_map_hover_card` and
  `_render_map_selection` call `get()` on entries that are not cell winners;
  with the reserve set to exactly the drawn count, every hover evicts a cell
  that is still on screen and the next frame decodes it again. The reserve is
  the whole PLAN, not the cells on screen - reserving the viewport shrinks the
  cache as you zoom in and evicts the level you came from.
  **Both axes quantise off ONE level.** Quantising them apart crosses x and y
  at different zooms, so an octave reshuffles the whole atlas twice instead of
  once, and each reshuffle is a visible sweep; it also makes the cell a
  RECTANGLE, which stretches the thumbnail inside it. `atlas_cell` takes the
  power of two on x and scales y by the canvas aspect - ROUNDED, because the
  caller scales both spans by the zoom and an unrounded ratio is a fresh cache
  key every frame. A finer level KEEPS every winner the coarser one had (the
  coarse winner is still the winner of one of its sub-cells), so a zoom-in
  only decodes the new sub-cells; the budget is what breaks that subset
  relation, which is why returning to a level is cheap rather than free.
  A thumbnail decode is 0.399 ms measured over 300 real 160px JPEGs, so
  `ATLAS_NEW_PER_FRAME` is 8. PIL `draft()` is NOT worth it for time (0.31 ms
  at 1/4) but cuts decoded pixels 16x, which is the lever if the budget ever
  needs to be much larger than the cache.
  **The map decodes its OWN thumbnails, small, and that is what lets every
  cell have a picture.** `gl_loader(..., max_px=ATLAS_TEX_PX)` uses PIL's
  `draft()`, which is JPEG DCT scaling rather than a resize: 80px from the
  stored 160, an eighth of the memory. Draft is NOT a speed win - 0.31 ms
  against 0.399 measured over 300 real thumbnails - it is a MEMORY win, and
  memory is what bounds how many cells can hold a picture. The map therefore
  keeps a SECOND `ThumbCache`; the gallery's 160px one is unchanged, and both
  are released on an archive switch, since ids restart per archive.
  `max_capacity` is per instance for the same reason - the ceiling is a memory
  budget and these textures are not the gallery's size.
  **The PLAN is uncapped and the VISIBLE set is capped.** Capping the plan by
  novelty leaves gaps exactly where the map is zoomed IN, because the cells in
  view need not be among the whole map's most novel - which reads as a zoom
  that will not show you anything new. The plan is a few int arrays and costs
  nothing to hold; the cap belongs in `_atlas_rects`, on the cells actually
  drawn, which is the working set the cache has to hold. The reserve is a
  CONSTANT, never the visible count: sizing the cache to the viewport shrinks
  it as you zoom in and throws away the level you came from.
  **The atlas is DOUBLE-BUFFERED, and that is what removes the sweep.** A plan
  reaches the screen only once every picture in its VISIBLE cells is resident;
  until then the previous snapshot keeps drawing, scaling with the zoom the way
  map tiles do. Without it a wholesale change - a zoom level, or switching PCA
  to UMAP - wiped new thumbnails across the map a row at a time. The snapshot
  is `(pts, plan, cell)` and EVERYTHING is read from it, positions and `idx`
  alike: switching projection replaces `pts`, so the current one's `idx` names
  different entries. Only the visible cells are waited for - a level holds
  thousands and none of the ones off screen are what the eye is on - and an
  unreadable thumbnail counts as arrived, or one missing file holds the plan
  back for good. First open is the exception: with nothing to fall back on it
  adopts as it fills, since a blank canvas is worse than a fill.
  **The snapshot names rows of the ARCHIVE it was built from**, so it is
  retired when the archive changes and not merely when `pts` does. Switching
  to a shorter archive otherwise indexes past the end of its entry list
  mid-frame, and takes the app down inside `thumb_key`.
  **The hover reads the plan being DRAWN.** The scatter is not drawn in this
  mode, so resolving a hover against it named an entry the user could not see.
  **The SIZE slider is continuous because only the ZOOM is quantised.**
  Quantising the cell snapped 49 slider positions onto 3 values. A pan or a
  zoom may not disturb the binning; moving the slider is a deliberate act and
  may rebin.
  **`UmapLayout.available()` uses `find_spec`, never `import umap`.** That
  import pulls in numba and llvmlite and costs **7.7 seconds**; the Layout
  combo asks every frame whether to OFFER the engine, so the first Map frame
  froze the whole app - on startup, since the browser reopens itself. The real
  import happens inside `fit()`, which runs on a worker. 2.15 ms now.
  **The cost the user actually felt was the atlas's own work, never the
  scatter.** The scatter was present in both states, so it cannot explain a
  toggle; do not repeat that diagnosis. Re-measure by timing points-only,
  atlas-plus-scatter and atlas-only in ONE process.

- **Colour and Draw are ORTHOGONAL: the heatmap carries the same colour the
  dots would, and count moves to the ALPHA channel.** A density map coloured by
  its own cell counts silently overrides the Colour combo, so switching to
  novelty appeared to do nothing in that draw mode. A ramp mode averages the
  cell (`cell_means`); `source` cannot be averaged — two packed colours added
  together are not a colour — so it takes the cell's **majority**
  (`cell_majority`). The alpha floor is well above transparent, because a
  one-entry cell fading to nothing hides exactly the frontier the log scale
  exists to show.

- **Opening the Explore tab is where every lazy cost lands at once.** The
  biggest was a directory walk: `list_archives` spent 3418 ms, essentially all
  of it `_size_mb` walking every thumbnail with `Path.rglob` to print one number
  — `os.scandir` is 69.6x faster and byte-identical. What remains is real and
  must not be "fixed" by caching: `VisionScorer()` 1207 ms is the ONNX sessions
  the mode runs on, and archive open (0–2439 ms) is `load_from_store`, which
  pays `rescore_all()` on the first open after the closing flush learned to
  stamp `novelty_n` and once per unclean exit thereafter.

- **Undo detects a change by DIFFING declared state, and a field nobody
  classified is the defect that guards against.** Every field of `SimState` and
  `PreferencesState` is in `UNDOABLE_FIELDS` or in `NOT_UNDOABLE` with a
  reason, and `tests/test_undo_fields.py` derives its cases from
  `__dataclass_fields__` so a new field fails until someone says which. An
  explicit `push_undo()` at each mutation site was rejected for exactly this:
  it fails nothing when forgotten. A call site may `tag()` a step to name it,
  which is advisory — a forgotten tag costs a name, never coverage. The diff is
  only viable because nothing writes the authoritative state per frame: sweeps
  and jitter are computed in the shader, and audio modulation hands
  `sim.apply_state` a `replace()`d copy rather than touching `ui_state.sim`.

- **Applying an undo step must REBASE the journal, and a HOVER PREVIEW must be
  refused a recording outright.** Both write the state the diff watches.
  Without `rebase()` the next frame reads an undo as a fresh change and commits
  it — the history growing in the direction it was asked to shrink. The
  preview is worse and ORDERING DOES NOT FIX IT: capturing before the preview
  only defers its write to the NEXT frame's capture, which commits, which
  shifts the rows under the pointer, so a different row previews and commits in
  turn — the history fills in seconds. `_record_undo_step` therefore returns
  early while `_undo_preview_base` is set, which also covers the frame the
  preview is handed back on. Applying a step CLEARS that base, or the pointer
  leaving afterwards restores the pre-hover state and silently undoes the
  click.

  **THE UNDO PANEL IS NOT THE ONLY THING THAT HOVERS**, and covering only its
  own preview is the shape this bug came back in. File > Load, the archive
  browser and the clipboard each put a borrowed rule and its run physics on
  screen the same way, and each recorded twice per row — once hovering, once
  restoring. `CommandHandler.preview_active` is the one predicate over all
  three flags, so a fourth preview is covered by naming its flag there rather
  than by remembering a check at a fourth site. It reads the flags rather than
  the borrow, because a same-brain preview borrows nothing. Suppression ends
  on the CLICK, which clears the flag in the frame it commits, so a preset the
  user actually loads still records.

  Guarded by `tests/test_undo_preview_loop.py`, which drives the real `App`
  methods — the frame-loop tests that modelled the sequence with a local helper
  all passed while this was live.
  The capture is separately deferred while `any_widget_active`, which is read
  inside `ui.render()`: `orchestrate_frame` runs BETWEEN frames. That deferral
  is the whole of gesture coalescing.

- **Restoring a brain must move the WINDOW, not just the sim.**
  `_handle_brain_layout` applies `ui_state.brain` every frame — that is how a
  count slider reaches the decode with no one-shot flag — so a restore that
  writes only `sim.brain_layout` is undone by the very next frame, which reads
  as an undo that works and then keeps the new brain on top.
  `_restore_snapshot_brain` therefore calls `_put_brain_window` as well, and
  still calls `apply_brain_layout` itself rather than leaving it to that frame,
  because the rule it pushes immediately afterwards is measured against the
  LIVE layout and silently refused on a mismatch. Both directions go through
  it, so redo has the same requirement. `_apply_brain_layout` early-returns on
  an unchanged layout, which is what stops the following frame paying for a
  second archive rebuild.

- **An undo step must be something the USER did, and three parts of the app
  drive undoable preferences on their own.** An automatic mode, a recording
  and a screenshot each commandeer `speedmult`, `motion_blur` and
  `blur_quality` and put them back when they finish. All three are undoable,
  so every phase change of those state machines looked to the per-frame diff
  exactly like a slider moving: a generation deposits several, `MAX_STEPS` is
  200, and leaving Explore running discarded every real step the user had
  made — after which half the surviving steps restore `speedmult = 0`, which
  stops the sim and leaves a black canvas under a working UI.
  `App._preferences_are_borrowed` is the one predicate over all three, for the
  same reason `preview_active` is one over the previews: a fourth borrower is
  covered by naming it there. Guarded by `tests/test_undo_ownership.py`.

- **A DECODE SCALE is not in the signature, and the restore has to compare the
  settings as well.** Keeping scales out of `signature()` is deliberate —
  dragging one must not tear down the archive — but `Snapshot.brain_settings`
  carries them, so a drag DOES commit a step. Gated on the signature alone
  that step restored nothing, and the `rebase()` immediately after it
  overwrote the old scale with the new one, putting the value out of reach for
  good: the panel showed a row, Ctrl+Z did nothing, and the number was gone.
  `_brain_matches` compares both; a snapshot carrying no settings at all means
  "the signature is all we know", not "the defaults". MLP is the modality this
  matters most for, since `w_scale`/`b_scale` are what a rig modulates.

- **A hover BORROWS the state an undo would write, so an undo is refused while
  one is live.** `_end_archive_preview` restores the pre-hover physics
  wholesale when the pointer leaves, and `pop_rule()` takes back the rule — so
  an undo applied underneath a preview is reverted a moment later with nothing
  on screen to say so, and the rebase has already consumed the step it undid.
  Hovering does not set `want_capture_keyboard`, so Ctrl+Z genuinely reaches
  the orchestrator with a borrow outstanding. `_handle_undo` therefore returns
  early on `preview_active` with a notice, which is the same predicate
  `_record_undo_step` already uses.

- **A step's NAME travels as a one-shot on `ui_state`, never through the
  journal.** `UndoHistory.tag()` existed from the start and nothing called it,
  so every row was `describe()`'s guess — and a preset load moves a dozen
  fields at once, so every preset in the library produced a row saying
  "Settings". `CommandHandler` holds no journal (it sets `ui_state.undo_tag`
  and the orchestrator forwards it, the pattern every other command follows),
  and the tag is cleared by the frame that USES it: a load deferred behind a
  live widget keeps its name, while a load that changed nothing drops it
  rather than naming whatever moves next. Still advisory — the DIFF is what
  commits a step, so a call site that forgets to tag costs a name and never
  coverage. Guarded by `tests/test_undo_labels.py`.

- **Undo covers the recipe, never the picture.** The canvas and entity buffers
  are out, so Clear Canvas, Reset and Fill have nothing to restore, and
  deleting a preset or an archive entry stays outside. Under a tournament the
  brain half is skipped and the settings still apply, the same split
  `_grid_owner()` already makes for Z and G. Costs and the `Mapping.uid`
  prerequisite for audio are in
  `docs/superpowers/specs/2026-08-14-undo-redo-design.md`.

- **A render test must not read `imgui.ini`.** ImGui restores each window's
  saved size, position and scroll from it, and the file is gitignored — so a
  test that draws a window passes on a fresh clone and fails on a machine that
  has run the app. Sizing the HOST window is not enough; the window under test
  picks up its own saved geometry. `tests/conftest.py` handles the whole suite
  by wrapping `create_context`, and a fixture must NOT set the filename again
  itself — see the `imgui.ini` caveat under UI and platform for why that set
  has exactly one home.

- **`sim.py` is user-owned** — do not restructure without asking. It has its own
  hardcoded param lists in `entity_update()` and `_write_multi_load_ssbo()`.

- **CLIP/evolution dependencies are optional and lazy** — `onnxruntime-directml`,
  `tokenizers` and `cmaes` must never be imported at startup. Auto tournament
  mode degrades to a message when they are absent; manual mode must keep working.

- **`calculate_setting()` ignores `slider_value` whenever any sweep or jitter is
  non-zero.** Zeroing a slider is not enough to disable a parameter; zero the
  sweeps too.

- **Windows platform** — use forward slashes or `os.path`; use `rm` not `del` in
  bash commands.

- **Shaders must be in `shaders/`** relative to the executable for builds to work.

### Brains and modalities

- **The sim does not reproduce itself run to run, so a trajectory diff cannot
  validate a shader change.** Particles splat additively into a shared texture,
  which races. Measured at 30k particles / 200 steps, the *same tree run twice*
  moves 93.6% of particles by more than 1e-3 — as much as two genuinely
  different shaders do. Aggregate velocity statistics are equally blind. Compare
  a shader change by evaluating the affected function as a PURE function over a
  fixed input grid (`purefn` pattern: one invocation per point, no shared
  writes), which IS bit-reproducible. That is the only instrument that resolved
  a 56%-of-signal mutation defect which parameter checksums had scored as
  "within 1–3%".

- **The brain's input scale is a property of the PRESET, and it varies ~900x.**
  Measured over all 23 of `physics_configs/Core` by
  `tools/brain_input_scale.py`, the median `|input|` per preset runs **0.0016**
  (Searching) to **1.43** (Bubbles), median 0.057, median p90 0.36, median p99
  0.68. No constant can suit all of them, which is why Gabor's `Input Scale` is
  a setting and not a number in the module. Read a preset's figure off the tool,
  or watch the Inspector — at the right setting the tiles are blobs, at the
  wrong one they are broad ramps.

  Gabor is the modality that cares, because it compares `x` DIRECTLY to a
  centre in 4-D: at the old spread of 2.0, `||c|| ~ 4` against `||x|| ~ 0.4`, so
  the envelope was constant over everything a particle reads and a Gabor filter
  degenerated into a plain oscillation. Measured as the correlation between a
  unit and the same unit with its envelope removed (1.0 = it IS a Fourier), the
  median preset scored 0.85 at spread 2.0 against 0.74 at 0.35, and presets
  with room to work moved much further (Salt 0.84 -> 0.61, Bubbles 0.71 -> 0.35).
  Lenia does NOT need it: its bump compares `w.x`, and the projection amplifies
  by `W_SCALE` and sums four terms, which already lands near its `mu` range.

  For roughly a third of the presets the input barely moves at all (Searching
  p50 0.0016, p90 0.0032). Over a range that small every smooth brain is
  approximately linear, so the modality cannot matter much and the lever is
  `SENSOR_GAIN`, not the brain.

- **A setting that is declared but never read is the recurring defect here.**
  It has happened twice: every non-count slider before `BrainLayout.scales`
  existed, and `fourier.low_freq_bias` after. Both rendered, both moved, both
  did nothing. `tests/test_brain_scales.py` derives its cases from
  `settings_schema()` rather than a hand-written list, because the hand-written
  list is what let the second one through.

- **"No rule loaded" means one generated brain PER COHORT, for every modality,
  and there is no blank-brain state.** It used to mean two different things:
  Fourier answered an all-zero buffer with a GPU-side generator, and the other
  three got one CPU brain shared by every cohort. Since `MUTATION_SCALE`
  defaults to **0.0** and `num_cohorts` to **64**, that was 64 independent rules
  against 64 copies of one — a monoculture — and the GPU path could never have
  been shared, because it builds `FourierCenter`s. `generated_brains()` now
  draws them on the host via the registry, into the cohort slots
  (`COHORT_BRAIN_SLOT0`, mirrored in `_header.glsl`), selected by
  `BRAIN_PER_COHORT`. Fourier lost nothing: `generate_random_centers()` and
  `FourierModality.random()` are the same formula.

- **An all-zero rule is the "no brain" marker AT THE RIGHT WIDTH TOO.** The Z
  key, the undo history and `_Default.json` all send a zeroed `(10, 8)`, which
  *is* 80 floats. While the GPU fallback existed this was harmless; uploaded
  verbatim it is a brain that outputs zero for every input. Measured on
  `_Default` when `apply_rule` accepted it: the brain's own p90 output fell from
  0.431 to **0.034**. Anything that fills a slot — `apply_rule`,
  `_write_multi_load_ssbo`, `write_tournament_rules` — must substitute a
  generated brain rather than write zeros, or that slot is silent.

- **A brain's mutation belongs to its modality, not to a generic per-float
  helper.** The Fourier mutation is structured: ONE scalar scales all four
  components of a centre's frequency (so the frequency *vector* keeps its
  direction and only changes length), amplitudes take a vec4 from a single
  `hash4()` per centre, and the seed is hashed from the rule's own CONTENT plus
  the cohort. Jittering the four frequency components independently rotates the
  vector instead — a different function, 56% mean deviation, and it still looks
  plausible on screen. A modality that mutates structurally implements it in its
  own `.glsl` and exposes `<modality>_param_at()` so the click-to-adopt
  writeback returns the rule the particle was actually running.

- **A hover BORROWS another brain; only a click switches to one.** A switch
  releases and rebuilds the archive — `load_from_store` rescores every entry —
  and resets the optimizer, so it can never run off the pointer position. A
  borrow moves the GPU side alone: `realloc_brain_buffers` is 0.03–0.05 ms at
  every layout, because the driver does not touch the pages, and the modality
  uniform already follows `sim.brain_layout`. Four things are load-bearing.
  `_handle_brain_layout` is suppressed for the borrow's duration, or the Brain
  window's own layout undoes the hover — with a full teardown — once a frame.
  The layout goes back BEFORE the rule popped off the stack under it, since
  `apply_rule` measures a rule against whatever layout is live and silently
  refuses a mismatch. A commit returns it before `_adopt_foreign_entry`, which
  then switches in the SAME frame, so no frame renders the previewed brain
  under the user's. And the Inspector draws slot 0 with `sim.brain_layout`
  rather than the Brain window's, which are the same except during a borrow.
  Re-borrowing keeps the FIRST base, because sliding down the Load menu across
  two brains borrows twice with no return in between. Guarded by
  `tests/test_foreign_preview.py`.

  **EVERY hover path has to use it, including File > Load.** That menu previews
  on hover and its click then takes a `preview_rule_active` branch that only
  FINALISES - so a preset naming another brain, whose preview had applied
  nothing, did not load at all while the console printed that it had. The
  direct paths (Ctrl+V, a click with no hover) go through
  `_apply_config_with_locks` and were never affected, which is why fixing those
  looked like fixing the feature. `_config_borrow_layout` takes the decode
  scales from the file's own `brain_settings`, which an archive entry does not
  have.

  **A borrow therefore has an OWNER, and only its owner may return it.** Two
  previews borrow - `"menu"` and `"gallery"` - and both run every frame, the
  gallery second. Its teardown fires whenever no entry is hovered, so an
  unowned return undid the menu's borrow in the same frame it was taken, and
  the click then found nothing to make permanent: the exact symptom the borrow
  was added to fix, one layer down. Guarded by
  `tests/test_menu_cross_brain_load.py`, which runs `_handle_archive_preview`
  between the hover and the click because testing the menu alone missed it.

- **`Archive.brains` is padded to the WIDEST layout the archive holds; anything
  that decodes must use `brain_at(i)`.** One wider foreign entry re-widths the
  pooled column, so the padded row reaches `apply_rule` at the wrong size and
  is refused — which breaks the ordinary same-brain preview, not just the
  cross-brain one.

- **EVERY sampler that picks a PARENT OR A SEED must filter through
  `native_rows()`, and `novelty_goal` is where that was missed.** An archive
  pools every layout, so most of what it holds may be another brain's — and a
  seed becomes the optimizer's mean, re-encoded under the running layout, so a
  foreign row is not a worse start but an unreadable one. `_ask_expansion` and
  `_seed_index` both filtered; `novelty_goal` sampled `archive.entries` whole
  and is the ONE goal kind that carries its own `seed_index`, which
  `start_expedition_with` then bounds-checked but did not test for nativeness.
  It killed a 2.5-hour overnight run inside Fourier's `encode` — "cannot
  reshape array of size 71 into shape (10,8)", 71 being an `mlp-n3.4.4`
  genome — and it is inherently **probabilistic**: the foreign entries were 5%
  of the archive, so it took 2.5 hours to draw one. A mixed archive is the
  normal case, not the exotic one, because changing brain mid-run leaves the
  previous layout's entries in place. `_parent_z` now RAISES on a foreign row
  rather than letting it reach a modality's reshape, since the old message
  named neither the sampler nor the layouts. Guarded by
  `tests/test_novelty_goals.py`.

- **Every REGIME decision counts NATIVE entries; everything that POOLS counts
  the archive.** `regime`, the bootstrap progress readout, the
  expedition-cadence gate and the popsize-change fallback all ask `_native_n`,
  because switching brain inside a full archive leaves the new brain with
  nothing to expand FROM. Counting the whole thing put the driver in
  "expansion", where `_ask_expansion` privately fell back to bootstrap while
  every expedition declined for want of a native seed — so it neither
  bootstrapped nor expedition'd, and the progress bar named a regime it was
  not in. `len(archive)` still sizes novelty, the refresh sweep, the map and
  `seed_index`'s bounds check, all of which are about pictures rather than
  genomes. The browser states the split whenever an archive holds more than
  one layout, because a full-looking archive that is still bootstrapping has
  no other explanation on screen.

- **A layout change while a rollout is IN FLIGHT must drop the population, and
  `GenomeSpec` raises on a z from another space.** `abort_generation()` restarts
  the rollout WITHOUT re-asking — that is what makes it right for a resize or a
  shader reload, where the genomes are still readable. A layout switch is not
  that case: `set_layout` therefore clears `_z` and `tile_physics`, and
  `update()` asks again before the phase machine can reach `score_and_tell`.
  Left standing, the old population reached `tell()` under the new spec. It did
  not fail at the boundary either, because `decode`/`split` SLICE and a slice
  past the end is short rather than an error — so an 88-wide Fourier+physics z
  read as a 324-float MLP brain and died inside `mlp.decode` as
  `operands could not be broadcast together with shapes (324,) (88,)`, naming
  neither space. `_check_width` is the same discipline `_parent_z` already
  applies to a foreign row. Guarded by `tests/test_search_layout_wiring.py`.

- **A layout change is PRINTED, because nothing else records that it
  happened.** The archive is keyed by signature, so a switch silently
  redirects where results are filed and strands the previous brain's entries
  as un-breedable. A run config names the layout it ran under, but only once a
  run starts, and `settings_history.jsonl` never sees it — reconstructing a
  2.5-hour Fourier run inside a deep-MLP archive afterwards needed two
  `runs/*.json` files and the per-directory entry counts.
  `_apply_brain_layout` logs both signatures, and says so louder when a search
  is running.

### The MLP layer stack

- **`MAX_MLP_WIDTH` is COMPILED PER LAYOUT, and that is the only reason a deep
  layer may be wide.** Any depth above 1 must materialise a layer's
  activations, so `mlp.glsl` ping-pongs two `float[MAX_MLP_WIDTH]` locals — and
  the driver allocates those per invocation whatever the `BRAIN_DEPTH <= 1`
  branch does, in a program that also holds Fourier, Gabor and Lenia. So the
  cap is not a cost the wide stack pays, it is one **every brain in the build**
  pays. At 300k particles, the SAME layout under different caps: Fourier
  0.92 ms at cap 8 and **1.79x** at 48; depth-1 `mlp-n16-a0` 1.18 ms at 8 and
  **2.95x** at 48. That is what a raised constant would have cost, and it is
  why the old shared cap was 8. Caps 4 and 8 measure the same (±4%), so the
  floor is free — `SCRATCH_BUCKETS` starts at 4 because `mlp_hidden` seeds
  `cur[0..3]` before it looks at any width, not for speed.

  `shader_defines(layout)` is asked of **every** modality, not the running one,
  precisely because one program carries all four shaders; a modality answering
  for someone else's layout returns its floor. `sim.py` keeps one compiled
  entity-update program per distinct set of defines — Fourier, Gabor, Lenia and
  every depth-1 MLP share one — so a hover borrow is a dict lookup and never a
  recompile. Widths are BUCKETED so a slider drag lands on a handful of
  variants. The shader's own `#ifndef` default is `MAX_WIDTH`, because the
  builds that do not prepend (the Inspector, `tools/shader_compile_check.py`,
  the GPU tests) must carry any stack; never lower it to save time there.

  What a bucket costs, against depth-1 `[16]`: `[8,8]` 1.06x, `[8]x8` 2.31x,
  `[48]` 1.46x, `[16,16]` **2.62x**, `[24,24]` **5.42x**, `[32,16]` 6.15x,
  `[48,8]` **8.85x**. Cost tracks the BUCKET, not the float count — `[48,8]` is
  668 floats against `[24,24]`'s 820 and costs 1.6x more — which is why the
  Brain window names the bucket rather than reporting a slope. The Inspector's
  redraw stays under 0.74 ms throughout. `python -m tools.measure_brain_depth`;
  the trap when re-measuring is that the shader must be swapped inside ONE
  process against ONE entity snapshot, or what is timed is the laptop's
  thermal state.

- **The float budget is the only limit that bites, and `_shape_from_layers`
  CLAMPS to it.** Nothing else stops `[48]x8`, which is sixteen times over
  `MAX_BRAIN_FLOATS`; and clamping rather than raising is load-bearing, because
  the input may be a config written by a build with different limits. It is
  applied as ONE cap across the stack, largest that fits, so a layer already
  below it is left alone; it always terminates, since every layer at
  `MIN_WIDTH` is 27 floats at the deepest. `+ Add layer` is stopped by
  `MAX_DEPTH` or by the budget: a width-1 layer makes a brain SMALLER, because
  it narrows the output layer's fan-in from `w_k` to 1. The UI derives every
  limit by asking whether the layout BUILDS and comes back **unchanged in every
  row** — not just the row being dragged, since the budget is shared and a
  looser test lets one slider quietly narrow the row above it.

- **Depth 1 is bit-identical, and that is what makes the rest safe.** `shape` is
  `(w1, a1, w2, a2, …)`, so a one-layer stack is `(16, 0)` — the same length
  `9h+4`, the same field offsets, the same flat decode, the same initial draw,
  and a signature of `mlp-n16-a0` character for character. Deeper stacks grow
  parts: `mlp-n16.8-a0.1`. Nothing on disk was renamed or rewritten, and
  `hidden`/`activation` remain accepted as input forever.

- **`decode` is FLAT in every layer; fan-in normalisation is INITIALISATION
  only.** A gain folded into `decode` would not be a normalisation but a cap —
  `W_SCALE · sqrt(4/16) = 1.0` on hidden→hidden weights — putting a whole region
  out of reach of the search and of live editing. `random()` instead scales only
  the hidden→hidden slices of `z`, which leaves a depth-1 draw untouched. The
  cost is conditioning, which CMA-ES adapts to.

- **A modality may own its signature AND the parser that reads it back.**
  `signature_of`, `settings_from_signature`, `settings_of`, `unit_count` and
  `layout_uniforms` are optional hooks; `layout_from_signature` keeps its
  parse → rebuild → **compare** verification either way, which is what makes a
  modality-owned parser safe. The generic parser used to take `kind == "int"`
  only while `settings_of` emitted `int` and `choice`, so `mlp-n16-a1` rebuilt
  as activation 0 and was correctly refused — every sin and gelu archive entry
  was unadoptable and unpreviewable, and a same-WIDTH mismatch is invisible to
  `apply_rule`'s width check. Both directions now use `STRUCTURAL_KINDS`.

- **A config's SIGNATURE wins over its `brain_settings`.** The signature is what
  the rule was decoded under and what `apply_rule` measures width against; the
  settings only carry the decode scales it leaves out. `_restore_brain_settings`
  used to load the settings, so a non-default layout whose settings were missing
  or stale came back as the modality's defaults — the wrong brain, silently, and
  then the genome refused on width.

- **A layer operation needs THREE snapshots, and they answer three different
  questions.** `_layer_open` is the genome as the menu opened and is what Reset
  puts back; `_layer_base` is what Scale multiplies, REBASED after a reroll so
  the next drag scales what is on screen; `_layer_live` is what the GPU holds,
  because a scale in progress has not been pushed onto the rule stack and so
  cannot be read back from it. Collapsing open into base makes Reset a no-op
  after two rerolls — it undoes the edits it was rebased by. Scale multiplies a
  snapshot rather than the live value, or a drag compounds and the layer
  explodes; it pushes ONE history entry on close, and none at all if the slider
  never moved. Guarded by `tests/test_brain_layer_ops.py`.

- **Scale is deliberately unclamped, and what it costs is only SEEDING.** The
  archive stores decoded params, so a rescaled layer saves and plays back
  exactly as it looks. `imgep_driver.py` and `command_handler.py` call
  `encode(archive.brain_at(i))` to start CMA-ES from an entry, and `encode` is
  `arctanh(params / W_SCALE)`, which clips — so an expedition seeded from a
  past-the-rail creature starts from a clipped version of it. If creatures out
  there turn out to be interesting the conclusion is that `W_SCALE` is wrong,
  not that the region should be fenced; promoting it to a `Setting(kind=
  "float")` makes it a decode scale, which neither splits the archive nor
  resets a search. That is what `w_scale` and `b_scale` now are.

- **`w_scale` 0.25–8.0 and `b_scale` 0.0–4.0 are measured, and the DEFAULTS
  decode bit-identically to the constants they replaced.** MLP was the one
  modality whose settings were all structural, so `brain_targets` offered
  nothing and the audio panel's Brain section was empty under it — which reads
  as the panel failing rather than as a property of that brain. Below `w_scale`
  0.25 the hidden layer is near-linear and output magnitude stops tracking the
  slider at all — p50 output is 0.41, 0.38, 0.35 at 0.1, 0.25, 0.5, i.e. flat
  and slightly BACKWARDS. Upward the limit is saturation: on a lively input
  units railed at |h| > 0.99 run 3.3% at 4.0, 21% at 8.0 and 38% at 12. Bias
  stops at 4.0 on the same measure, 12% railed against 36% at 6.0. Two traps:
  `encode` must divide by the SAME scales `decode` multiplies by, or the
  modulator's encode-once/decode-many moves the brain the instant audio touches
  it; and `b_scale` reaches 0 legitimately, so encode floors it rather than
  dividing by zero. `python -m tools.measure_mlp_scale`. Guarded by
  `tests/test_brain_scales.py`, which derives its cases from
  `settings_schema()` — that is what proves a new scale is actually READ.

- **A layer edit has an OWNER, and two states have none.** Under a tournament
  slot 0 is tile 0 of a running grid, which is rewritten every generation, so
  nothing an edit could survive; during a hover borrow slot 0 holds someone
  else's brain and the commit path would keep the edit. Both are refused in
  `_handle_brain_source`, and the Source combo says which. With no rule loaded
  there is no single brain at all — every cohort has its own — which is why the
  Source selector exists rather than the window guessing.

- **The Inspector draws the LAST hidden layer's units.** They are the only ones
  that decompose additively into the output; a unit in an earlier layer reaches
  it through further nonlinearities and has no contribution to show. `shape[0]`
  would size the atlas off layer 1 and index past the units that exist, so
  `unit_count()` is a modality hook.

## Key Documentation

- `docs/imgep.md` — **how Explore mode works**: the regime loop, every fitness
  and admission equation in pseudocode, and what each setting does. Read this
  before the search and archive caveats above, which assume it.
- `README.md` — the USER-facing walkthrough of tournament mode and the brain
  modalities: which menu, which button, what each control does. Keep the
  how-to-drive-it there and the why-it-is-built-this-way here.
- `ARCHITECTURE.md` — comprehensive architecture, file map, data flows, subsystem docs
- `ui/README.md` — mixin architecture, how to add windows/sliders
- `docs/adding_ui_shader_params.md` — step-by-step guide for new parameters
- `docs/testing_checklist.md` — manual verification passes for the sim
- `BUILD.md` — PyInstaller build instructions
