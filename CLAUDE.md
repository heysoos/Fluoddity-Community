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

- **Emboss's height field is `100 * length(canvas.xy)`, and the 100 is
  load-bearing.** It used to be the canvas `.z`, a `0.01 * coverage` channel
  the RG canvas no longer has. Flow magnitude is `|velocity| * coverage`, so
  the gain makes the two agree at `|velocity| = 0.01`, the middle of the
  library's range. Without it the field is beyond the slider: the shading
  compares the gradient to `0.5/I^5`, so a SMALLER field needs a LARGER
  intensity, and the two shipped emboss presets would want 1.09 and 2.27
  against a slider that stops at 1. `Core/Adrift` (0.584 → 0.423) and
  `Advanced/DrawOnMe` (0.656 → 0.866) were re-solved by bisecting on the
  RENDERED result until each matched its old strength — 15.9% and 33.4% of the
  image's own brightness — because the gradient-ratio arithmetic got the
  direction of the correction backwards.

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

### CLIP and the capture

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
  `min_separation` (0.02, measured) refuses anything within that cosine
  distance of a stored entry — the unstructured-archive rule from
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
  (rewritten wholesale, an OPTIONAL key so older archives still open), and
  `load_from_store` calls `rescore_all()` unconditionally — 0.42 s at 4808
  entries, 4.3 s at capacity. Skipping it hands generation 0 — every tile
  stamped 1.0 by the no-reference convention — 100% of the `p ~ novelty^4`
  parent weight, and one of those entries is a black frame. Four things read
  this column: expansion parents, `latent_goal`'s anchor, `prune_to_capacity`,
  and the browser sort.

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
  to buy ~5% more); it takes no locks in `CLIPScorer` (ORT `run` is
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

- **A wheel event reaches the zoom AND the scrollbar, unless a child eats it.**
  ImGui scrolls the hovered window during `NewFrame`, so a canvas that reads
  `io.mouse_wheel` to zoom also scrolls the panel it sits in — which reads as
  the wheel doing something different every time. The map canvas is therefore
  a child window with **both** `no_scrollbar` and `no_scroll_with_mouse`: with
  only the latter, ImGui walks up to the parent and scrolls that instead.
  Guarded by `tests/test_archive_window_render.py::test_the_map_canvas_is_a_child`.

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
  subset, so narrowing also expands what is left. UMAP was considered and not
  used: PCA's `transform()` is a matmul, so a new entry places instantly against
  stable axes, while UMAP is non-parametric and its refit relayouts everything.

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
  must not be "fixed" by caching: `CLIPScorer()` 1207 ms is the ONNX sessions
  the mode runs on, and archive open (0–2439 ms) is `load_from_store`'s
  load-bearing `rescore_all()`.

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

### The MLP layer stack

- **A DEEP stack's width cap is paid by every MLP, including the one-layer
  ones.** Any depth above 1 must materialise a layer's activations, so
  `mlp.glsl` ping-pongs two `float[MAX_MLP_WIDTH]` locals — and the driver
  allocates those per invocation whatever the `BRAIN_DEPTH <= 1` branch does.
  Against the pre-stack shader, at 300k particles, a depth-1 `mlp-n16-a0` step
  costs **1.02x at cap 8, 1.22x at 12, 1.41x at 16 and 4.09x at 48**. So a LONE
  layer keeps its historical 48 (`MAX_WIDTH`) and every layer of a deeper stack
  is capped at 8 (`MAX_DEEP_WIDTH`) — which is why adding a second layer
  NARROWS the first, and the Brain window says so before you click. Depth costs
  roughly what its float count costs: against depth-1 `[16]`, `[8,8]` is 1.07x,
  `[8]x8` 2.16x, and the historical `[48]` 1.47x. The Inspector's redraw stays
  under 0.26 ms throughout. `python -m tools.measure_brain_depth`; the trap when
  re-measuring is that the shader must be swapped inside ONE process against ONE
  entity snapshot, or what is timed is the laptop's thermal state.

- **The float budget is slack now, and the WIDTH cap is the only limit that
  bites.** The deepest reachable stack is `[8]x8` = 580 floats against
  `MAX_BRAIN_FLOATS` 1024 — still past 512, so the raise is load-bearing, but
  nothing the UI can build goes over. `+ Add layer` is stopped by `MAX_DEPTH`
  alone: a width-1 layer makes a brain SMALLER, because it narrows the output
  layer's fan-in from `w_k` to 1. The UI derives both limits by asking whether
  the layout BUILDS and comes back unclamped, never by repeating the packing
  formula — which already has two homes, `layer_spans()` and `mlp.glsl`, guarded
  by the GPU-versus-NumPy parity test.

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
