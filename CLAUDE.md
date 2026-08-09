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

- **`Archive.refresh()` is what makes eviction cheap.** Eviction drops the
  lowest *stored* novelty, which is only meaningful because 64 entries per
  generation are re-scored against the full archive. Turning `refresh_per_gen`
  down to 0 silently degrades eviction into "drop whatever was least novel when
  it was admitted".

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
  read this column: expansion parents, `latent_goal`'s anchor, `_evict_one`,
  and the browser sort.

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

- **`sim.py` is user-owned** — do not restructure without asking. It has its own hardcoded param lists in `entity_update()` and `_write_multi_load_ssbo()`.
- **Windows platform** — use forward slashes or `os.path`; use `rm` not `del` in bash commands.
- **No test suite** — changes must be verified manually.
- **Shaders must be in `shaders/`** relative to the executable for builds to work.

## Key Documentation

- `ARCHITECTURE.md` — comprehensive architecture, file map, data flows, subsystem docs
- `ui/README.md` — mixin architecture, how to add windows/sliders
- `docs/adding_ui_shader_params.md` — step-by-step guide for new parameters
- `BUILD.md` — PyInstaller build instructions
