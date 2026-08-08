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

- **`sim.py` is user-owned** — do not restructure without asking. It has its own hardcoded param lists in `entity_update()` and `_write_multi_load_ssbo()`.
- **Windows platform** — use forward slashes or `os.path`; use `rm` not `del` in bash commands.
- **No test suite** — changes must be verified manually.
- **Shaders must be in `shaders/`** relative to the executable for builds to work.

## Key Documentation

- `ARCHITECTURE.md` — comprehensive architecture, file map, data flows, subsystem docs
- `ui/README.md` — mixin architecture, how to add windows/sliders
- `docs/adding_ui_shader_params.md` — step-by-step guide for new parameters
- `BUILD.md` — PyInstaller build instructions
