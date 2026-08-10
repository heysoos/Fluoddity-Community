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

- **Entry ids restart at 0 in every archive**, so `ThumbCache` must be released
  on a switch: it is keyed by thumbnail filename, which derives from the entry
  id, so reusing it shows the previous archive's pictures.

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

- **An ImGui widget's identity IS its label, and a duplicate silently kills the
  loser.** Two visible items hashing to one ID puts Dear ImGui's "conflicting
  ID" dialog over the app and stops one of them responding to the mouse at all
  — it is not a warning. A `collapsing_header("Archive")` shipped in the same
  window as `combo("Archive")`, and the control picking which archive the search
  writes into could not be clicked. `##suffix` keeps the visible text and
  changes the ID. Guarded by
  `tests/test_archive_window_render.py::id_clashes`.

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
