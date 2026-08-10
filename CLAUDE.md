# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fluoddity is a GPU-accelerated 2D particle simulation for generative art. Thousands of particles follow neural-net-like "Rules" governing their response to trail density, producing emergent patterns. Physics runs entirely on the GPU via GLSL compute shaders. Built with Python 3.12, ModernGL (OpenGL 4.3), GLFW, imgui_bundle, NumPy, and FFmpeg.

## Commands

```bash
# Run (development)
pip install -r requirements.txt
python main.py

# Build distributable (Windows, PowerShell)
.\build.ps1

# Manual build
python -m PyInstaller --clean --noconfirm Fluoddity.spec
Move-Item -Path "dist\Fluoddity\_internal\shaders" -Destination "dist\Fluoddity\shaders"
```

There are no automated tests. Verification is manual (see `docs/testing_checklist.md`).

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

Uses **mixin-based decomposition** (multiple inheritance). The `UI` class in `core.py` inherits 8 mixins so all render methods share `self.*` — chosen because ImGui's immediate-mode paradigm requires shared widget state. Import via `from ui import UI`.

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
3. **`sim.py`** — add `tryset()` call in `entity_update()`
4. **`ui/physics_window.py`** — add ImGui widget (use `slider_float_with_range_menu` for full context menu support)

No additional wiring needed — the orchestrator pattern handles the rest.

## Important Caveats

- **CLIP/evolution dependencies are optional and lazy** � `onnxruntime-directml`,
  `tokenizers` and `cmaes` must never be imported at startup. Auto tournament mode
  degrades to a message when they are absent; manual mode must keep working.
- **Tile and cohort indices both derive from the particle index**, so the cohort count
  must be a multiple of the tile count � otherwise each tile holds exactly one cohort
  and stays a monoculture. See `services/cohort_tiling.py`.
- **`calculate_setting()` ignores `slider_value` whenever any sweep or jitter is
  non-zero.** Zeroing a slider is not enough to disable a parameter; zero the sweeps too.

- **`world_size` cannot change particle density.** It scales entity count by
  `ws` and canvas area by `ws` (side by `sqrt(ws)`), so particles-per-texel is
  a constant 0.572 for every world size - it buys *more world*, not a more or
  less crowded one. `particle_density` is the multiplier that moves density.
  Lowering it is *cheaper*: at ws=4.0, density 0.1 runs 4.7x faster than 1.0.

- **`ACTIVE_COUNT` in `entity_update.glsl` is the denominator for every
  index-derived slice** - cohorts (`get_cohort`) and tournament tiles
  (`tournament_home_tile`). It must equal the real allocated count, so it is
  `float(ENTITY_COUNT)` and must never be re-derived from `WORLD_SIZE`. A
  denominator larger than the buffer silently compresses every slice into the
  bottom of the range: at density 0.5 the top half of the tournament grid would
  render empty.

- **The archive stores decoded phenotypes, never `z`.** With physics search on,
  `z` is relative to `physics_origin` — the preset loaded at the time. A `z`
  archived under one preset decodes to a different creature under another. See
  `tests/test_physics_origin_roundtrip.py`.

- **Admission does not gate on novelty; capacity prunes.** Everything finite,
  viable and alive is admitted, and `prune_to_capacity()` evicts the least
  novel once over the cap — one bulk pass per generation, from `tell()`, AFTER
  `refresh()`. The adaptive threshold that used to gate was removed 2026-08-08
  for two independent measured reasons. Its `observe()` ran once per
  *candidate* — 16 tiles a generation at grid 4, 64 at grid 8 — each
  multiplying the threshold by 1.05 or 0.95, so it could move **2.18x** in one
  generation (22.7x at grid 8) while steering on a rate averaged over ~6
  generations; simulated on a *stationary* novelty distribution it admitted
  nothing in 61% of generations at a rate std of 0.315 against a Bernoulli
  floor of 0.089. And separately, a generation's tiles are not independent
  draws — they share one parent sample or one CMA-ES population, so they clear
  or miss any bar together, which alone raises "every tile admitted" 12x. No
  gain fixes either. Do not reintroduce a novelty threshold without addressing
  both.

- **`Archive.refresh()` is what makes pruning meaningful, and its budget is a
  FRACTION of the archive.** Parent choice and eviction both rank on *stored*
  novelty, so what matters is how many generations a full sweep takes —
  `refresh_sweep_gens`, default 10, from which `ImgepDriver._refresh_count()`
  derives `ceil(len / gens)`. A fixed count does not hold as the archive grows:
  the old 64/generation swept 4808 entries in 75 generations and 20000 in 312
  (14.6 min) — and at grid 8 that is also 64 *admissions* a generation, so a
  sweep took exactly one full turnover. Staleness is directional and therefore
  self-reinforcing: patterns accumulate near each other so true novelty only
  falls, a stale value is systematically too HIGH (41–59% of entries measured
  inflated), and an inflated value makes an entry both likelier to be chosen as
  a parent and likelier to survive eviction. Cost at sweep=10: 0.9% of a 2.8 s
  generation at 4808 entries, 12.7% at 20000.

- **`_remove()` deletes the entry's thumbnail.** Nothing could reach it
  afterwards — `index.jsonl` is append-only and the id is gone from
  `vectors.npz`, so the row is dropped on the next open. Without this a full
  archive at grid 8 orphans 64 JPEGs every ~2.8 s, about 12 MB a minute.

- **Novelty is a LIVE column, so `index.jsonl` cannot be its home.** The index
  is append-only; its `novelty` is forever the at-admission value, measured
  against however much archive existed at the time. Entry #50 was scored
  against 49 neighbours and entry #4000 against 3999, and the search compares
  them as if they were on one scale. Measured 2026-08-08, the stored column
  correlates **0.075** with a correct rescore. It therefore lives in
  `vectors.npz` (rewritten wholesale, and an OPTIONAL key so pre-existing
  archives are not quarantined), and `load_from_store` calls `rescore_all()`
  unconditionally — 0.42 s at 4808 entries, 4.3 s at the 20000 capacity.
  Skipping that rescore hands generation 0 — every tile stamped 1.0 by the
  no-reference convention — **100.0%** of the `p ~ novelty^4` parent weight
  (ESS 58 of 4808), and one of those entries is a black frame. Four things
  read this column: expansion parents, `latent_goal`'s anchor,
  `prune_to_capacity`, and the browser sort.

- **`knn_distances` blocks over query rows.** Only k distances per query
  survive, so the (n, m) matrix is scratch — and a whole-archive rescore at
  capacity would ask for 1.6 GB of it at once. Blocking caps it at 64 MB and
  costs nothing measurable; the matmul is the same either way.

- **Novelty is measured against archive ∪ rejects ring.** The archive is gated,
  so without the ring the search has no memory of the regions it just rejected
  and re-explores them forever.

- **Liveness is small and its scale is measured, not intuited.** Over all 131
  presets at 2000 steps / 6 snapshots it runs 0.0034–0.0792 with median 0.0241,
  so `liveness_min` is **0.002**. A "reasonable-looking" 0.02 rejects a third of
  the curated preset library. Re-run `python -m tools.calibrate_imgep --liveness`
  before changing it, and note that liveness is *higher* during the transient
  after a reset than once a pattern settles into its attractor.

- **`canvas_view_rect` returns a GL texture coordinate (v=0 at the BOTTOM);
  `tex_to_screen` returns top-down window coordinates.** The flip between them
  cancels exactly when the canvas is centred in the window, so a centred camera
  validates any orientation bug you like. Test the capture crop with the camera
  *panned* — a mirrored rect displaces every tile by twice the pan.

- **Bloom in the capture runs per tile, not over the grid.** It is a 5-level
  mip chain reaching 30–60px, so blooming the whole grid put a quarter of a
  tile's worth of neighbouring glow into every crop and the optimizer scored it
  as the creature's. `CaptureView.draw_grid` splits first and blooms each tile
  alone, which makes cross-tile bleed impossible rather than merely small.

- **The tournament capture does not crop the window view — it re-renders the
  canvas** at `grid*224` with an identity camera (`services/capture_view.py`).
  Pan, zoom and window size cannot affect what the optimizer scores, and there
  is no crop rect to get wrong. Anything that needs the *displayed* image
  (screenshots, video) still uses `camera.assembled_texture`.

- **A tournament tile is a SMALL WORLD, and gets the world's own boundary
  condition.** It is not a box with walls. Three places enforce the tile edge —
  the particle boundary block and the sensor confinement in
  `entity_update.glsl`, and `getBlur` in `canvas.frag` — and all three must
  agree with `BOUNDARY_CONDITIONS_MODE`. Under wrap a tile is a TORUS: it has
  no edge, so nothing can pile up against one. Before 2026-08-09 the tile was
  always a walled box, which produced exactly the artifacts it should:
  particles stacked on a line (the old bounce SET the position to the wall
  instead of reflecting by the overshoot), and a band of width `sample_dist`
  around every tile where both sensors clamped to the same texel and the
  steering differential was identically zero — along both axes at once in a
  corner, which is why corners looked worst. Verified on the GPU in
  `tests/test_tile_isolation_gl.py`; the non-tournament path is bit-identical.

- **The diffusion's tile guard must be the tile's own BOX, never a tile index.**
  `tournament_tile_uv` derived the index with `clamp(uv, 0, 0.999999)`, so a
  neighbour probe that walked off the canvas clamped back into the *same* tile,
  the zero-flux substitution was skipped, and the tap fell through to the
  sampler — which has `repeat_x/repeat_y` set and duly returned the opposite
  edge of the canvas. Measured 2026-08-09: one lit tile at (3,0) put **64.4%**
  of its brightness into tile (0,0) *and* into (3,3), while its genuinely
  adjacent neighbours stayed at exactly 0, and it retained only **74.6%** of
  its own trail. Only tiles touching the canvas border leaked — 12 of 16 at
  grid 4, 28 of 64 at grid 8 — and corner tiles on two edges each.

- **A tile owns a whole number of TEXELS, in INTEGER arithmetic, and `%` is
  banned near a seam.** The canvas is `int(1024*sqrt(world_size))` — **647** at
  the default world_size of 0.40 — and the grid slider is 2..8, so no canvas
  size divides for every setting. Three independent defects came out of this at
  grid 8, each of which alone reproduced "the middle rows and columns are not
  toroidal":
  1. *An equal share of the WORLD is not a whole number of texels.* 647/8 =
     80.875, so seams ran through the middle of a texel. Column 4 retained
     **25.7%** of its own trail and **39 of 64** tiles lit a tile they could not
     legally reach.
  2. *Float arithmetic decides a seam by its last bit.* GLSL does not require
     division to be correctly rounded, and 647*4/8 is exactly 323.5 — this GPU
     evaluates `floor((323.5/647)*8)` as **3** where the true answer is 4, so
     texel 323 fell outside its own tile's box and bridged tiles 3 and 4.
     Snapping alone made it **worse** (4.6%). `tile_lo_texel()` is integer
     ceil-division.
  3. *`%` is undefined in GLSL when either operand is negative*, and the south
     and west probes are always at `lo - 1`. That cost **84%** of a tile's
     trail at 647 and *nothing at all* at 1024, where the tile is 128 wide and
     the compiler's bitmask happens to be right for negatives. **This is why
     testing at a power-of-two resolution hides all three.**
  One definition, repeated in `entity_update.glsl`, `canvas.frag` and
  `brush.vert`, plus `services/tile_geometry.py` for the CPU (the capture crop
  uses it too — an even split put ~1.4px of the neighbour into each 224px
  tile). `tests/test_tile_isolation_gl.py` runs the whole grid at 647/8, 647/3
  and 641/7 on purpose.

- **CLIP is strongly POSITION-dependent, and CLIP time is not free.** Measured
  2026-08-09 on real archive thumbnails, rolling one 16px on the torus moves
  its embedding 0.078–0.088 — **2.5–2.7x** the distance to its nearest genuine
  neighbour, and past the 0.02 separation bar for 100% of tiles. The dip at
  exactly 32px (0.037) is ViT-B/32's patch stride, which is the mechanism.
  Averaging random sub-crop views buys the invariance back:

  | views | roll 16px | repeat noise | same/unrelated |
  |---|---|---|---|
  | 1 | 0.0784 | 0.0000 | 0.459 |
  | **3** | **0.0319** | **0.0069** | 0.425 |
  | 5 | 0.0236 | 0.0050 | 0.445 |

  `repeat` is what averaging COSTS: views 1..n are random draws, so the same
  image no longer embeds identically. At 3 the nuisance falls 0.046 and the new
  noise is 0.007 — about 7:1 — and 0.0069 is a third of the separation bar, so
  a duplicate still cannot pass on noise alone. Use `embed_mean()`, never
  `embed()` directly: `embed()` returns `B*v` rows unreduced, and feeding those
  to the archive makes three sub-crops of one tile into three descriptors.

  **The old spec's "96 images at n_views=1 ≈ 7 ms" is wrong by ~40x**, and
  most of the real cost was never on the GPU. Measured 2026-08-09 per 64
  images: `preprocess` 112.6 ms, DirectML 58.4 ms, `augment` 28 ms — CPU
  normalisation was **54–58% of the whole CLIP path**. Two fixes, both in
  `clip_scorer.py`:
  - `preprocess` transposes while the data is still **uint8** (one byte per
    element, not four) and folds the normalisation into one in-place
    multiply-add: 112.6 → 48.7 ms, and exact in fp16.
  - `_embed_images` **pipelines**: the next chunk is normalised on a worker
    thread while the GPU runs the current one. The stages are now comparable
    (48.7 ms vs 58.4 ms) and numpy drops the GIL, so they overlap — 1.39x at
    1152 images. Lookahead is exactly one chunk; submitting all of them would
    hold 345 MB preprocessed at grid 8. Only `preprocess` leaves the main
    thread: `augment` draws from `self._rng`, and only one thread may call
    `session.run`.

  Grid 8 with 6 snapshots, end to end: 916 → **561 ms** at 1 view, 3136 →
  **1802 ms** at 3. Views stay a slider because 3 views is still ~2x of 1.
  Note DirectML device 0 is already the discrete GPU here (0.90 ms/image);
  device 1 is the Intel iGPU at 63 ms/image, so never pin `device_id`.

  **Centring on the centre of mass was measured first and rejected.** It cancels
  a shift exactly, but 26–34% of tiles have no well-posed centre (resultant
  R median 0.065–0.073) — these are space-filling textures, not localised
  objects — so it pushed real same-goal near-duplicates APART by 15–19% in two
  of three archives.

- **Admission gates on SEPARATION, and that is not the novelty threshold coming
  back.** `min_separation` (0.02, measured) refuses anything within that cosine
  distance of an entry already stored — the unstructured-archive rule from
  quality-diversity. Without it a converging expedition stores its own endpoint
  64 times a generation: measured on the real archives, ONE goal had already
  contributed **25%** of `debug07` (3200 of 12672) and **30%** of `debug05`.
  Replaying each archive in insertion order at l=0.02 keeps 33.3% of `default`
  and cuts that goal's share about twice as hard as everything else, taking a
  generation from 16–64 admissions to 5–14. At 0.05 every archive keeps under
  6%, so the useful span really is 0–0.05.
  This differs from the threshold removed 2026-08-08 in both ways that killed
  it: there is no feedback controller and no gain, and correlated tiles landing
  on top of each other is the case it is *meant* to reject rather than a
  pathology. Three things follow: the driver forces the generation's most novel
  viable tile through (`force=True`) so a run always leaves a trail; `force`
  never bypasses liveness; and a separation rejection is **not** added to the
  rejects ring, because the ring is memory of regions the search was refused and
  this region is in the archive already.

- **Colour is genetic but, at the default hue gain, NOT HERITABLE — so no text
  goal naming a colour can work.** `e.color.x = HUE_SENSITIVITY * col_params.x`
  is an HSV hue, read modulo 1, while `col_params.x` is an unbounded Fourier
  output. At `hue_sensitivity = 0.5` the product spans several whole turns, so
  a mutation far too small to change the pattern still slides the colour
  through multiple revolutions. Measured 2026-08-09 by rendering real archive
  genomes headlessly (10 bases x 5 mutations, 2000 steps):

  | condition | mean \|dhue\| | vs the 90 deg random baseline |
  |---|---|---|
  | same genome, new reset seed | 4.8 deg | 0.05 |
  | sigma 0.10 (`expedition_sigma`) | 74.3 deg | **0.83** |
  | sigma 0.30 | 88.9 deg | 0.99 |

  The pattern survives that mutation (CLIP cosine 0.936 against a 0.895
  random-pair baseline); only the colour does not. Lowering the gain fixes it,
  and *raises* variety — at 0.15, `|dhue|` is 23.4 deg and the hue spread across
  genomes is 107 deg, against 80.8 deg and 68.6 deg at 0.5. Selection response
  (Spearman rho between the real "purple" contrastive fitness and measured
  purpleness over 24 real mutants) goes 0.263 -> 0.507.

  Not a bug in CLIP: on hue-rotated real tiles it picks the right rotation at
  2.6x chance, and purple is already 5.9-9.5% of every archive. `hue_sensitivity`
  is NOT in `PHYSICS_PARAMS`, so the search cannot fix this itself — it is a
  preset value. Note also that `color_by_cohort` REPLACES the brain's hue with
  `hash(cohort)`; tournament mode forces it off (`_tournament_plain_colour`), so
  any colour experiment run outside tournament mode measures the wrong thing.

- **Entry ids restart at 0 in every archive**, so `ThumbCache` must be released
  on a switch — it is keyed by thumbnail filename, which is derived from the
  entry id. Reusing it shows the previous archive's pictures under the new
  archive's entries.

- **`Archive.maybe_flush` only rewrites `vectors.npz` every 200 admissions.**
  `index.jsonl` is flushed per entry, so anything that closes an archive —
  quitting, or switching to another one — must call `maybe_flush(force=True)`
  first or lose the trailing entries.

- **Explore mode reuses Auto mode's `AutoTournamentService` instance**, swapping
  only `.driver`. Both `_handle_auto_tournament` and `_handle_explore` would
  otherwise call `configure()` on the same object every frame, so each bails out
  when the other owns the driver.

- **The expedition fitness is contrastive, and its logit scale is
  per-modality.** `TEXT_LOGIT_SCALE` is CLIP's own 100; `IMAGE_LOGIT_SCALE` is
  30. Image-image similarity sits above 0.9 where 100 is far too sharp - a +3sd
  latent goal at scale 100 floors **59.6%** of tiles to zero, leaving 10.6 of a
  16-tile generation distinguishable to a rank-based optimizer. `contrastive()`
  has no default scale on purpose.

- **Seed selection must use the SAME objective as the fitness.**
  `Archive.nearest()` is `argmax(embeddings @ goal)`, and for a text goal its
  winner is a noise texture: high-frequency noise carries a middling cosine to
  every phrase, so measured over 15 unrelated prompts it returned only 6
  distinct seeds and one cyan static tile won 7 of them. Every text expedition
  started from the same bad image. `ImgepDriver._seed_index` uses the
  contrastive score instead (14/15 distinct); what rejects noise is that
  `DEFAULT_DISTRACTORS` contains "random noise" and "an abstract texture".
  The seed is then SAMPLED with `p ~ fit^alpha` - E&E's parent rule, same
  alpha - so repeating a goal explores a different trajectory.

- **The seed pool is a BAND on effective sample size, never a target.** How
  concentrated a goal's matches are is real information: measured over 20
  prompts against 4784 entries, ESS at alpha=4 ran 3.1 ("a photograph of a
  cat" - the archive genuinely holds almost nothing cat-like) to 1973 ("circuit
  board traces" - it holds a great deal). Pinning ESS to a value would tell the
  cat prompt it has 64 good seeds when it has three. `banded_alpha` only clips
  the ends, leaving 6 of those 12 prompts untouched. Two further traps it
  handles: the floor is capped at N/8, because a floor demands candidates that
  may not exist; and `effective_sample_size` works in log space, because
  `w**alpha` overflows float64 inside the bracket bisection explores.

- **`LATENT_DIMS` must stay small (8).** Two independent reasons, both measured:
  whitening equalises the components, so a large d puts the push into
  geometrically tiny directions (d=32 was a no-op, seed rank 0.1 against 41.3 at
  d=8); and the goal lies wholly inside the d-dimensional subspace while entries
  keep most of their energy outside it, an effect that scales with
  `dim / LATENT_DIMS`. "Capture more variance" silently turns the push off.

- **The descriptor and the expedition fitness are separate computations** over
  the same per-snapshot embeddings. `descriptor()` renormalises the trajectory
  centroid - right for novelty, which needs unit vectors - but `1/||m||` grows
  as snapshots decorrelate, so using it as fitness paid a bonus for *changing*
  rather than for *matching* (0.22 SD of the spread). Score per snapshot, then
  average, as `PromptDriver` does.

- **`Archive.nearest()` returns `argmax(embeddings @ goal)`**, so an expedition
  always seeds on the archive's best entry under its own goal. That is fine only
  if the goal leaves room past it. The old `g = b + beta*(b - c)` did not - over
  200 trials the goal's nearest entry WAS its own seed 199 times - and no value
  of beta fixed it. See `docs/superpowers/specs/2026-08-08-expedition-objective-design.md`.

- **Scoring runs OFF the frame loop, and the split is exactly `precompute()`.**
  CLIP is 92–97% of a generation's main-thread cost — measured 2026-08-10 at
  grid 8 / 6 snapshots: `tell()` is 1143 ms at 1 view and 4324 ms at 3, of
  which `_embed` is 1057 / 4194; `consider` × 64 is 2 ms, `refresh` 30–77 ms,
  `prune` 0. Blocking on it froze the app once a generation, which is what
  "the program gets stuck during archive growth" was. `AutoTournamentService`
  submits `driver.precompute(list(self._buffer))` to a one-thread pool and
  `score_and_tell()` returns **None** until it lands; `update()` keeps
  returning `Action.SCORE` because neither the phase nor the snapshot counter
  moved, so there is no new state machine. Measured with a 60 Hz stand-in
  frame loop: worst frame gap **1857 ms → 68 ms**, generation wall clock
  1.86 → 1.99 s.
  Three things this deliberately does NOT do. It does not move the rest of
  `tell()`: everything after `precompute` mutates the archive, which the UI
  reads every frame for the gallery, the map and the status, so that half
  would need a lock around every one of those reads to buy ~5% more. It takes
  no locks in `CLIPScorer` — ORT `run` is thread-safe and the vision and text
  sessions are separate objects; a lock held for a 4 s vision pass would
  freeze `set_prompt` and put the stall back. And it passes a **copy** of the
  frame buffer, because `abort_generation()` clears the list.
  A test that drives `update()` in a tight Python loop can starve the worker
  outright — 500 iterations fit inside one 5 ms GIL switch interval — so
  `tests/test_auto_tournament_service.scored()` sleeps 1 ms on a `None`. The
  real frame loop renders, so this does not arise in the app.

- **`keeper` is the most NOVEL tile; `summit` is the best-MATCHING one, and an
  expedition needs both.** Separation asks "do we already have one of these",
  and a chase converging on a goal necessarily makes tiles that look like the
  ones it just made — so the tile that best matches the goal, the actual
  result of the expedition, is precisely what separation throws away.
  `keeper` never covered it: during a converging chase the most novel viable
  tile is close to the least goal-matching one. `_summit()` is a **ratchet**
  on the expedition's own best fitness: a climb over rough ground leaves a
  checkpoint at every genuine gain, a converged expedition stops improving and
  therefore stops admitting, and the worst case is one extra entry per
  generation (`expedition_gens`, 50) against the 64 a generation that flooding
  produced. It forces past separation only — never past liveness or viability
  — and the mark resets in `start_expedition_with` as well as
  `end_expedition`, because contrastive fitnesses against different goals are
  not comparable. These entries carry `source="summit"` so they are findable
  afterwards; the expedition fitness therefore has to be computed BEFORE the
  admission loop rather than after it.

- **A latent or chase goal has ONE reference, which makes its fitness a
  monotone squash of raw cosine — and raw cosine to an arbitrary direction is
  maximised by NOISE.** `contrastive` with a single reference is
  `sigmoid(30·(⟨e,g⟩ − ⟨e,c⟩))`, so the latent path had no degenerate-image
  defence at all; text goals were never exposed because
  `DEFAULT_DISTRACTORS` carries "random noise" and "an abstract texture".
  Measured 2026-08-10 against lag-1 spatial autocorrelation of the stored
  thumbnails (validated: white noise 0.00, smooth ramp 0.99), the entry a
  latent goal ranks FIRST is in its archive's roughest decile **30.2%
  (default) / 19.8% (debug05)** of the time, against 10% by chance.
  The answer is `capture_health.structure()` as a multiplicative factor on
  every expedition fitness: **14.0% / 5.0%** after. It is deliberately NOT a
  CLIP term — the distractors cannot simply join a latent goal's reference
  set, because image-image similarity sits near 0.9 and image-text near 0.2,
  so at one logit scale the text references contribute only a constant, which
  a softmax is invariant to. It is applied to text goals too, where it barely
  moves ranking (real entries score 0.93–0.98), because one rule beats two.
  **The SEED was not the problem, and the obvious fix makes it worse.**
  Seeding a latent expedition at the anchor it was extrapolated from measured
  15.8% → **27.0%** on `default` (13.5% on debug05): `_seed_index` does not
  argmax, it samples with `banded_alpha`, which already spreads the draw,
  while `p ~ NOV^4` concentrates hard and novelty is itself mildly
  rough-biased. Only `novelty_goal` sets `Goal.seed_index`, because it has no
  embedding to score the archive against.

- **A novelty expedition is the third goal kind, and the only thing that
  optimises novelty WITHIN a generation.** Expansion samples parents by
  novelty then mutates blindly; `novelty_goal` makes the generation a CMA-ES
  hill-climb on the same kNN novelty that decides admission and parent choice,
  with no target that might be unreachable. Its objective is deliberately
  non-stationary — admitting entries during the expedition lowers the novelty
  of everything near them, so the optimizer chases a receding target. That is
  what an explorer should do, but it does mean the covariance is adapting on
  shifting ground. `novelty_share` and `latent_share` are independent sliders
  clamped rather than normalised (rescaling one because the other moved would
  make neither mean what it says); text takes the remainder, and `_draw_goal`
  falls THROUGH the kinds rather than failing, because an expedition that does
  not start wastes a whole cadence interval. Novelty alone is not a noise cure
  either — top-N by novelty is in the roughest decile 19.0% / 13.2% of the
  time — which is why the coherence factor applies to it as well.

- **The record book runs in EVERY regime, and it is measured against the
  ARCHIVE, not against the run.** A tile that matches any enabled text goal
  better than anything the archive holds is admitted — `source="record"`,
  `goal` set to the phrase it beat rather than the one being chased. This is
  not a variant of `summit`: a run chasing "pepperoni pizza" routinely
  produces the best "a smiley face" the archive has ever had, and nothing else
  keeps it — novelty does not know the goal list exists, and separation asks
  whether the archive already holds something SIMILAR, which is a different
  question from whether it holds something BETTER.
  Scored on DESCRIPTORS on both sides via `contrastive()` with the distractor
  set — the same objective `_seed_index` ranks seeds by. Deliberately not the
  per-snapshot averaging the expedition fitness uses: archive entries have one
  descriptor and no snapshots, so averaging one side only would compare two
  different quantities. `_goal_records` is keyed BY TILE so a tile topping
  several goals is admitted once, credited to its largest margin; two goals
  won by two different tiles is two entries, which is correct.
  Cost is 2 `contrastive()` calls per enabled goal, and the archive side is
  what scales: measured 2026-08-10 at **1 ms/goal at 4808 entries and
  3 ms/goal at 13049**, so 10 goals is 10–29 ms on a ~2 s generation. It
  cannot move to `precompute()` — it reads `archive.embeddings`, which is
  exactly the shared state kept on the main thread.

- **Liveness is a FLOOR on the bulk, never a veto over a chosen entry, and it
  ranks nothing.** It appears in exactly two functional places — the archive's
  admission gate and the `viable`/`shows_something` split in
  `ImgepDriver.tell` — and `descriptor()`, the expedition fitness and parent
  sampling all ignore it. A hard floor is infinite priority, though, so
  `consider(ignore_liveness=True)` drops the CHANGE half of the alive gate for
  the summit and nothing else: `cand.viable` (black or blown-out frame) still
  has to hold. The reason is in the liveness caveat above — liveness is
  *higher* during the post-reset transient than once a pattern settles, so a
  converging expedition's endpoint, the very thing the chase is for, scores
  low on it. `keeper` KEEPS the veto, because it fires every generation
  forever and a dead preset would otherwise deposit one frozen tile per
  generation without bound.
  Measured 2026-08-10 over the three real archives, the floor is close to
  inert on real content anyway: the 1st percentile of admitted liveness is
  0.0062–0.0078, three to four times the 0.002 floor, and only 0.1–0.5% of
  entries sit within 2x of it. Its rank correlation with novelty is small and
  inconsistent in sign (+0.251, −0.077, +0.139), so it is not skewing the
  archive's contents either.

- **Expansion draws one parent PER TILE, independently and with replacement,
  so the grid sets the number of draws and not the number of parents.**
  `_ask_expansion` samples `p ~ novelty^alpha`, re-encodes each parent's
  stored phenotype under the current origin, and adds isotropic
  `sigma_expand` noise — no crossover, no covariance, no shared distribution.
  At the default alpha=4 the two numbers nearly coincide (measured
  2026-08-10: 15.5–15.9 distinct parents of 16 draws, 58.0–61.9 of 64), but
  the slider reaches 8, where `default`'s ESS is **2.2** and one entry takes
  **66.6%** of the weight — 64 tiles collapse to 16 distinct parents. Nothing
  else scales with the grid: `sigma_expand`, `alpha`, `k` and the refresh
  budget are all grid-independent. Expeditions are the exception — CMA-ES
  takes `popsize = tournament.tiles`, so there a bigger grid IS a bigger
  population.

- **An ImGui widget's identity IS its label, and a duplicate silently kills
  the loser.** Two visible items hashing to one ID puts Dear ImGui's
  "conflicting ID" dialog over the app and stops one of them responding to the
  mouse at all — it is not a warning. A `collapsing_header("Archive")` shipped
  in the same window as `combo("Archive")`, and the control that picks which
  archive the search writes into could not be clicked. `##suffix` keeps the
  visible text and changes the ID; different windows, child windows and tree
  nodes are already separate scopes.
  `tests/test_archive_window_render.py::id_clashes` monkeypatches every
  ID-bearing widget and compares `imgui.get_id(label)` at each call site, so
  the next one fails a test instead of a click.

- **Opening the Explore tab is where every lazy cost lands at once, and the
  biggest one was a directory walk.** Measured 2026-08-10 on the real archives
  root (10 folders, ~33k thumbnails): `list_archives` **3418 ms**,
  `CLIPScorer()` **1207 ms** (570 vision + 498 text ONNX sessions), archive
  open 0 ms at 0 entries / 429 ms at 4808 / 2439 ms at 13049. All of
  `list_archives` was `_size_mb` walking every thumbnail with `Path.rglob` to
  print one number under the combo; `os.scandir` is **69.6x** faster
  (2596 → 37 ms) and byte-identical, taking the whole call to 58 ms. What is
  left is real: the ONNX sessions are what the mode runs on, and
  `load_from_store`'s `rescore_all()` is load-bearing (see the novelty caveat
  above). Do not "fix" the remainder by caching either of them.

- **`sim.py` is user-owned** — do not restructure without asking. It has its own hardcoded param lists in `entity_update()` and `_write_multi_load_ssbo()`.
- **Windows platform** — use forward slashes or `os.path`; use `rm` not `del` in bash commands.
- **No test suite** — changes must be verified manually.
- **Shaders must be in `shaders/`** relative to the executable for builds to work.

## Key Documentation

- `ARCHITECTURE.md` — comprehensive architecture, file map, data flows, subsystem docs
- `ui/README.md` — mixin architecture, how to add windows/sliders
- `docs/adding_ui_shader_params.md` — step-by-step guide for new parameters
- `BUILD.md` — PyInstaller build instructions
