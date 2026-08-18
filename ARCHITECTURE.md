# Fluoddity Architecture

Fluoddity is a GPU-accelerated 2D particle simulation for generative art. Thousands of particles follow neural-net-like "Rules" that govern how they respond to trail density, producing emergent patterns ranging from flowing rivers to branching lightning. The physics engine is a generalization of [Sage Jenson's physarum transport model](https://cargocollective.com/sagejenson/physarum). Built with Python, ModernGL (OpenGL compute shaders), GLFW, and Dear ImGui.

## Core Architecture: Orchestrator Pattern

```
                        App (main.py)
                   ┌───────┼───────────┐
                   │       │           │
            CommandHandler  │   SimulationRunner
                   │       │           │
         ┌─────────┴───────┴───────────┴─────────┐
         │                                       │
    ┌────┴────┐   ┌─────┐   ┌────────┐   ┌──────┴──┐
    │   UI    │   │ Sim │   │ Camera │   │Services │
    │ (imgui) │   │(GPU)│   │(render)│   │         │
    └─────────┘   └─────┘   └────────┘   └─────────┘
```

**App** is a thin orchestrator. Each frame it:
1. Reads UI state via `ui.get_state()`
2. Delegates one-shot commands to `CommandHandler`
3. Delegates camera input to `process_camera_input()`
4. Manages recording/screenshot state machines
5. Applies state to Sim and Camera
6. Delegates physics + frame assembly to `SimulationRunner`
7. Renders camera view and UI

Components never talk to each other directly. All coordination flows through the orchestrator.

## Key Design Patterns

### Passive UI
The UI renders ImGui widgets and exposes state via `get_state()`. It does **not** run simulation logic, manage rules, or coordinate services. The orchestrator reads UI state and acts on it.

### One-Shot Flags
UI sets boolean flags (e.g., `request_reset`, `request_save_file`, `toggle_recording`) that the orchestrator reads and resets each frame. This keeps the UI stateless with respect to application behavior.

### State Containers
All mutable state lives in dataclasses in `state/`. The UI modifies these directly via widget bindings. The orchestrator reads them and applies to components.

### GPU Compute Pipeline
Physics runs entirely on the GPU via GLSL compute shaders. `Sim` manages shader programs, buffers, and textures. The CPU-side code just dispatches compute calls and reads back results when needed.

## File Map

### Orchestration Layer
| File | Lines | Description |
|------|-------|-------------|
| `main.py` | 936 | App orchestrator: init, frame loop, recording/screenshot state machines |
| `command_handler.py` | 1123 | Processes one-shot UI commands (resets, config save/load, mouse clicks, history) |
| `simulation_runner.py` | 394 | Physics stepping + frame assembly (motion blur / non-motion blur) |
| `camera_input.py` | 89 | WASD/QE camera movement + scroll-to-zoom (standalone function) |

### UI Package (`ui/`)
Mixin-based architecture. The `UI` class in `core.py` inherits all mixins via multiple inheritance, so every method shares the same `self` for access to shared state. See [`ui/README.md`](ui/README.md) for details.

| File | Lines | Description |
|------|-------|-------------|
| `core.py` | 717 | UI class definition, `__init__`, GLFW callbacks, `get_state()`, render dispatch |
| `physics_window.py` | 634 | Physics settings panel: all slider groups, multi-load mode |
| `slider_widgets.py` | 450 | Slider with range menu, context menus, jitter, sweep/range buttons |
| `help_windows.py` | 370 | Controls, tutorial, parameter sweeps, performance, video recording windows |
| `menu_bar.py` | 433 | File/Reset/Help/Extras menus, load submenu with preview, auto-close |
| `history_window.py` | 219 | Rule history display, tooltip shader rendering |
| `preferences_window.py` | 367 | World size, physics frequency, mouse mode, view, appearance settings |
| `config_browser.py` | 232 | Config file scanning, caching, hierarchical load submenu rendering |
| `popup_modals.py` | 85 | Save/Overwrite/Delete confirmation dialogs |
| `archive_window.py` | 905 | Explore (IMGEP) tab, archive picker, gallery, and the 2-D semantic map |
| `audio_reactive_window.py` | 731 | Audio device and bands, the mapping rig, per-row shapers, rig presets |
| `brain_window.py` | 572 | Brain modality picker, per-modality settings, the MLP layer stack, Inspector |
| `undo_window.py` | 54 | Undo history: every step, hover to preview, click to jump |

### Simulation & Rendering
| File | Lines | Description |
|------|-------|-------------|
| `sim.py` | 1052 | GPU particle simulation: shaders, buffers, physics dispatch, parameter sweeps |
| `camera.py` | 428 | Camera state, coordinate transforms, screen rendering |

### Services (`services/`)
Stateless or near-stateless helpers owned by the orchestrator.

| File | Lines | Description |
|------|-------|-------------|
| `config_saver.py` | 558 | Save/load physics configs (JSON + legacy binary formats) |
| `multi_load_service.py` | 150 | Mix up to 64 configs simultaneously, cohort assignment |
| `arrow_debug_service.py` | 95 | Debug overlay rendering trail flow vectors as arrows |
| `entity_picker.py` | 63 | Find nearest particle to mouse click (CPU readback) |
| `rule_manager.py` | 60 | Rule history stack with push/pop/undo |
| `video_recorder.py` | 45 | Thin facade over VidSaver for video recording |
| `undo_history.py` | 177 | The undo/redo journal: snapshot the declared fields, name what changed |
| `record_audio.py` / `record_crop.py` / `record_view.py` | 102 / 56 / 41 | The soundtrack tap, the world-space crop rect, and which texture a take reads |

### Audio input (`services/`)
A block of samples becomes named signals; a rig binds those to physics and
brain parameters. Nothing here writes the user's sliders — the runtime hands
the sim a modulated COPY, which is also what keeps the undo journal quiet.

| File | Lines | Description |
|------|-------|-------------|
| `audio_analysis.py` | 469 | FFT, bands and their measures, the centroid and its gate, the mel display |
| `audio_capture.py` | 220 | Device enumeration and the capture thread; a loopback endpoint or a microphone |
| `audio_runtime.py` | 216 | Per-frame seam: shaper state, the modulated copy, the slider overlays |
| `audio_mapping.py` | 163 | What a rig row is, which targets exist, and the modulation maths |
| `audio_shapers.py` | 115 | Per-row shaping; none of them may generate motion from silence |
| `audio_track.py` | 71 | Writes the raw soundtrack a recording is muxed with |
| `audio_brain.py` | 69 | Encode once, decode many: audio moving a brain's decode scales |
| `audio_rig_io.py` | 68 | The last-used rig and the named rig presets |

### Search and exploration (`services/`)
Auto (Prompt) mode and Explore (IMGEP) mode share one rollout machine
(`AutoTournamentService`) and differ only in their `SearchDriver`. Both score
through a `VisionScorer`, built from a `services/vision_models.py` registry
entry; Auto picks its encoder freely, Explore takes the one its archive is
pinned to.

**`docs/imgep.md` is the reference for how Explore works** — the regime loop,
every fitness and admission equation, and what each setting does.

| File | Lines | Description |
|------|-------|-------------|
| `imgep_driver.py` | 831 | Bootstrap / expansion / expedition regimes; the archive admission call site |
| `auto_tournament_service.py` | 410 | The rollout machine both modes share; scoring runs off the frame loop |
| `vision_scorer.py` | 330 | Any registered encoder via ONNX Runtime + DirectML; `embed_mean` multi-view averaging |
| `capture_health.py` | 128 | Viability, spatial coherence, confounded-sweep detection |
| `capture_view.py` | 243 | Re-renders the canvas at `grid*224` with an identity camera, blooming per tile |
| `tile_geometry.py` | 43 | Integer tile-to-texel arithmetic, shared with the shaders |
| `optimizers.py` | 331 | CMA-ES wrapper with a uniform ask/tell interface |
| `genome_spec.py` / `physics_genome.py` | 96 / 141 | What `z` encodes, and the physics-parameter half of it |
| `archive.py` | 565 | Admission gates, separation, capacity/eviction, novelty refresh |
| `archive_io.py` | 222 | `index.jsonl` + atomic `vectors.npz` + thumbnails; degrades to a no-op on disk trouble |
| `archive_library.py` | 253 | Named archive directories: safe names, listing, create / clear / delete |
| `goal_source.py` | 271 | `Goal`, the user's text `GoalList`, and `latent_goal()` whitened frontier extrapolation |
| `expedition_fitness.py` | 92 | Contrastive expedition objective; per-modality logit scales |
| `prompt_driver.py` | 153 | Auto mode's driver: CMA-ES climbing a text prompt |
| `vision_models.py` | 96 | The encoder registry: paths, preprocessing, logit scales, separation defaults |
| `settings_history.py` | 47 | Diff and replay for an archive's settings change log |
| `novelty.py` | 273 | k-NN novelty, `NOV^alpha` parent sampling, the rejects ring |
| `thumb_cache.py` | 79 | LRU of GL textures for gallery thumbnails, with explicit release |
| `descriptor.py` | 63 | Trajectory-centroid behaviour descriptor and ASAL liveness |
| `archive_projection.py` | 74 | PCA: 2-D for the map view, 8-D (with variances) for latent goals |
| `search_driver.py` | 46 | The `SearchDriver` protocol both drivers implement |

Encoder weights live under `Documents/Fluoddity/models/<subdir>/`, one
directory per registry entry, downloaded on demand by `tools/fetch_models.py`.
User data rather than app data, so every checkout and every installed copy
shares one set; `utilities.paths.migrate_models` moves an older `models/`
folder in from beside the app at launch.

Archives live under `Documents/Fluoddity/archives/<name>/`, one directory per
archive, each with the `index.jsonl` / `vectors.npz` / `goals.json` / `thumbs/`
shape `ArchiveStore` writes. There is no registry file: the filesystem is the
list, so a folder copied in from elsewhere just appears. `App._switch_archive`
rebuilds the archive, goal list, projection and thumbnail cache and repoints
every holder at the new directory; which archive is active is
`preferences.archive_name`.

`services/phase_metrics.py` (381 lines) is the offline half: per-cell measures
for the phase-diagram sweeps driven by `tools/phase_diagram.py`,
`tools/phase_figure.py` and `tools/phase_view.py`. It runs headless and is not
on the frame loop.

### State (`state/`)
Plain dataclasses. No logic, just fields with defaults.

| File | Lines | Description |
|------|-------|-------------|
| `sim_state.py` | 117 | Physics parameters (ALL_CAPS), rule seed, view options, sweep config |
| `preferences_state.py` | 122 | User preferences: motion blur, recording, mouse mode, keybindings, active archive |
| `ui_state.py` | 94 | Combined state snapshot returned by `UI.get_state()` |
| `archive_state.py` | 129 | Explore (IMGEP) settings, browser view state, and one-shot flags |
| `audio_in_state.py` | 221 | The audio rig: device, bands, mappings, and its persistence allowlist |
| `brain_state.py` | 85 | Which modality is live and its per-modality settings |
| `auto_tournament_state.py` | 53 | Auto (Prompt) settings: prompt, encoder key, optimizer |
| `multi_load_state.py` | 24 | Multi-load toggle and config list |
| `tournament_state.py` | 24 | Manual tournament: grid size and tile selection |
| `camera_state.py` | 11 | Camera position + zoom |
| `recording_state.py` | 7 | Recording active flag |

### Utilities (`utilities/`)
| File | Lines | Description |
|------|-------|-------------|
| `ffmpeg_recorder.py` | 272 | FFmpeg pipe-based video encoder |
| `save_frame_gpu.py` | 248 | GPU-side screenshot with supersampling |
| `frame_assembler.py` | 221 | Temporal accumulation (motion blur) + final composite |
| `keybinding_management.py` | 146 | Rebindable keyboard shortcuts |
| `paths.py` | 165 | Platform-aware path resolution (app dir vs user Documents) |
| `gl_helpers.py` | 115 | OpenGL utilities (tryset, readback_rule, buffer helpers) |
| `vid_saver.py` | 76 | Frame-buffered video saver (wraps ffmpeg_recorder) |

### Shaders (`shaders/`)
| File | Description |
|------|-------------|
| `entity_update.glsl` | **Core physics** compute shader: particle movement, sensing, rule application |
| `fourier4_4.glsl` | Compute shader: trail diffusion via Fourier convolution |
| `canvas.vert/.frag` | Trail rendering to canvas texture |
| `camera.vert/.frag` | View texture generation from canvas (camera transform) |
| `frame_assembly.vert/.frag` | Final composite: gamma, sweep reticle, motion blur accumulation |
| `brush.vert/.frag` | Mouse drawing brush rendering |
| `cam_brush.vert/.frag` | Camera-space brush overlay (draw trail cursor) |
| `arrow_debug.vert/.frag` | Debug arrow overlay for trail flow vectors |
| `tooltip_graphic.frag` | Animated shader for physics tooltip visualization |

## Data Flow: Physics Parameter

```
SimState.SENSOR_GAIN                  # state/sim_state.py - dataclass field
    ↓ (UI widget binding)
imgui.slider_float(v=state.sim.SENSOR_GAIN)  # ui/physics_window.py
    ↓ (get_state returns combined state)
ui_state = ui.get_state()                     # main.py orchestrate_frame
    ↓ (apply state)
sim.apply_state(ui_state.sim)                 # main.py → sim.py
    ↓ (set uniform)
tryset(program, 'SENSOR_GAIN', state.SENSOR_GAIN)  # sim.py entity_update()
    ↓ (GPU)
uniform float SENSOR_GAIN;                    # shaders/entity_update.glsl
```

For a step-by-step guide to adding new parameters, see [`docs/adding_ui_shader_params.md`](docs/adding_ui_shader_params.md).

## Data Flow: One-Shot Command

```
User clicks "Reset" in menu
    ↓
UI sets self._request_reset = True            # ui/menu_bar.py
    ↓
get_state() packages it into UIState          # ui/core.py
    ↓
App reads ui_state.request_reset              # main.py
    ↓
CommandHandler.process_commands() handles it  # command_handler.py
    ↓
sim.reset() called                            # sim.py
```

## Data Flow: Config Save/Load

```
Save: UI → request_save_file flag → CommandHandler → ConfigSaver.create_config(SimState, rule) → JSON file
Load: UI → request_load_file flag → CommandHandler → ConfigSaver.load_from_file() → apply_config(SimState) → sim.apply_rule()
Preview: Hover in load menu → push rule to stack → unhover → pop rule from stack
```

## Key Subsystems

### Rule System
A "Rule" is a 10x8 float32 matrix that parameterizes each particle's neural-net-like behavior function. Rules are managed as a stack by `RuleManager`:
- **Left click** (Select Particle mode): Read back the clicked particle's mutated rule, push to stack
- **Right click**: Pop rule (undo)
- **Full reset (Z)**: Push a zero rule
- Config save/load pushes rules to the stack

### Parameter Sweeps
Physics parameters can vary spatially (X/Y sweep) or by particle cohort. When enabled, a reticle shows the current parameter values at its position. Left click samples parameters at that position; right click enters preview mode (temporarily disables sweeps).

### Motion Blur
Temporal accumulation: run N physics steps per display frame, render intermediate frames, and blend them. Controlled by `speedmult` (physics steps per frame) and `blur_quality` (render every Nth step). The `SimulationRunner` handles both motion-blur and non-motion-blur paths through shared helpers.

### Video Recording
State machine: idle → (optional pending wait for scheduled frame) → recording → finished. While recording, `speedmult` and `blur_quality` are locked to recording settings. The orchestrator saves and restores user settings around recording.

### Screenshot
2-frame state machine: pending → in_progress → save + restore. On the override frame, settings are temporarily maxed for quality (motion blur enabled, blur_quality=1, speedmult=motion_blur_samples).

## Technologies
- **Python 3.12** - Application logic
- **ModernGL** - OpenGL 4.3 compute shader dispatch + rendering
- **GLFW** - Window management, input
- **imgui_bundle** (Dear ImGui) - Immediate-mode GUI
- **NumPy** - CPU-side array operations
- **FFmpeg** - Video encoding (via subprocess pipe)
