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

- **`sim.py` is user-owned** — do not restructure without asking. It has its own hardcoded param lists in `entity_update()` and `_write_multi_load_ssbo()`.
- **Windows platform** — use forward slashes or `os.path`; use `rm` not `del` in bash commands.
- **No test suite** — changes must be verified manually.
- **Shaders must be in `shaders/`** relative to the executable for builds to work.

## Key Documentation

- `ARCHITECTURE.md` — comprehensive architecture, file map, data flows, subsystem docs
- `ui/README.md` — mixin architecture, how to add windows/sliders
- `docs/adding_ui_shader_params.md` — step-by-step guide for new parameters
- `BUILD.md` — PyInstaller build instructions
