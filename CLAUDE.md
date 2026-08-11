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
  in one frame. The default IS the range max, because a preset saved before
  this parameter existed must load unbraked. Two traps when re-measuring:
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

- **Colour is genetic but, at the default hue gain, NOT HERITABLE — so no text
  goal naming a colour can work.** `e.color.x = HUE_SENSITIVITY * col_params.x`
  is an HSV hue read modulo 1, while `col_params.x` is an unbounded Fourier
  output, so at `hue_sensitivity = 0.5` a mutation far too small to change the
  pattern slides the colour through whole revolutions: `|dhue|` is 74.3° at
  `expedition_sigma`, against a 90° random baseline, while the pattern survives
  (CLIP cosine 0.936 vs a 0.895 random-pair baseline). Lowering the gain fixes
  it and *raises* variety — at 0.15, `|dhue|` is 23.4° and the spread across
  genomes rises from 68.6° to 107°. `hue_sensitivity` is NOT in
  `PHYSICS_PARAMS`, so the search cannot fix this itself. Note `color_by_cohort`
  REPLACES the brain's hue with `hash(cohort)`; tournament mode forces it off,
  so any colour experiment run outside tournament mode measures the wrong thing.

### The archive and admission

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

## Key Documentation

- `docs/imgep.md` — **how Explore mode works**: the regime loop, every fitness
  and admission equation in pseudocode, and what each setting does. Read this
  before the search and archive caveats above, which assume it.
- `ARCHITECTURE.md` — comprehensive architecture, file map, data flows, subsystem docs
- `ui/README.md` — mixin architecture, how to add windows/sliders
- `docs/adding_ui_shader_params.md` — step-by-step guide for new parameters
- `docs/testing_checklist.md` — manual verification passes for the sim
- `BUILD.md` — PyInstaller build instructions
