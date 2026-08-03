# Interactive Tournament Mode — Design

**Date:** 2026-08-03
**Status:** Approved (pending spec review)
**Scope:** Add an interactive evolutionary "tournament" mode to Fluoddity (2D). A 4×4 grid
of live, independent sub-simulations, each running a different Fourier-brain genome. The
user clicks the panels they find interesting; the app breeds the next generation of 16
genomes by mutating (and keeping) the selected ones.

## 1. Goal & non-goals

**Goal:** Human-in-the-loop interactive evolution of the per-particle steering brain
(the 80-float `rule`). No fitness function — the human is the selector.

**Non-goals (YAGNI):**
- No automated/quantitative fitness.
- No evolution of the 12 physics sliders — brain-only. Physics sliders stay at their
  current single global values, shared by all tiles.
- No 3D (Fluoddity3D) support.
- Crossover between genomes is optional and OFF by default.

## 2. User-facing behavior

- A **Tournament mode** toggle (menu bar). When on, the normal single-sim path is
  replaced by the tiled 4×4 tournament path; when off, the app behaves exactly as before.
- The single 1024² canvas is partitioned into a **4×4 grid (16 tiles)**. Each tile is an
  independent sub-simulation running one genome. Tiles are **hard-isolated**: particles
  bounce off tile edges and trails do not diffuse across tile borders.
- The user **clicks any number of tiles** to select them (K = number selected), shown with
  a highlighted border overlay.
- **Next Generation** builds 16 new genomes: selected genomes are kept unchanged
  (elitism); remaining slots are filled by mutating random selected parents; an optional
  number of fresh-random genomes are injected for diversity. On advance, the canvas is
  **cleared and particles are reseeded** so every genome is judged from the same clean start.
- **Undo generation** restores the previous population from an undo stack.
- **Save selected** writes each selected genome to a physics-config JSON (reusing the
  existing `config_saver` / `physics_configs/` format) so keepers can be loaded normally.

## 3. Architecture — Approach B (single-canvas tiling)

Chosen over "16 independent Sim instances" because it reuses the existing single-sim path:
one particle buffer, one compute dispatch, one canvas pass. Total particles (600k) and
canvas size (1024²) are unchanged, so **per-tile particle density ≈ the normal single-sim
density** and the 16 panels cost roughly one sim's worth of GPU.

### 3.1 Genome & population representation
- Genome = one `rule`: numpy array shape (10, 8) float32 = 80 floats. Identical to the
  existing rule format used everywhere (`apply_rule`, config JSON `"rule"`, multi-load).
- `population`: a list of exactly 16 genomes, tile index `= ty*4 + tx` (row-major,
  `tx, ty ∈ [0,4)`).
- **Random genome generator** (Python, mirrors `generate_random_centers` in
  `shaders/fourier4_4.glsl`): for each of 10 centers,
  `freq_scale = 1 + 2*rand()**2`; `frequency[xyzw] = (rand()*2-1) * freq_scale`;
  `amplitude[xyzw] = rand()*2-1`. Use a seeded `numpy.random.Generator` for reproducibility.
- **Mutation operator** (Python, mirrors `mutate_rule` in `entity_update.glsl`):
  `amplitude += strength * (-1 + 2*rand(shape))`;
  `frequency *= 1 + 0.5*strength * (rand(shape) - 0.5)`.
  `strength` comes from a UI slider (default 0.15).
- **Crossover (optional, default off):** for each of the 10 centers, take that center from
  parent A or parent B at 50/50 (uniform center-wise crossover).

### 3.2 GLSL changes
New uniforms in `shaders/entity_update.glsl`:
- `uniform int TOURNAMENT_MODE;   // 0 = off, 1 = on`
- `uniform int TOURNAMENT_GRID;   // grid side, = 4`

Tournament rules live in a **16-slot rule SSBO**. Reuse `multi_load_rule_buffer`
(binding 4, already sized `64 * SIZE_OF_RULE_STRUCT`), writing the first 16 slots; the
shader indexes `configs`/rule array by tile index. (If reuse proves awkward, add a
dedicated `tournament_rule_buffer` at a new binding — decided during implementation, but
reuse is preferred.)

Shader logic when `TOURNAMENT_MODE == 1` (gated so `== 0` is byte-for-byte the old path):
1. **Tile of a particle:** `vec2 uv = pos*0.5 + 0.5;` `ivec2 tile = ivec2(floor(uv*TOURNAMENT_GRID));`
   `tile = clamp(tile, 0, TOURNAMENT_GRID-1);` `int config_index = tile.y*TOURNAMENT_GRID + tile.x;`
2. **Rule selection:** use `config_index` to fetch this tile's genome (overrides the normal
   rule path). Mutation/`RULE_SEED` per-cohort logic still applies *within* a tile if
   `MUTATION_SCALE > 0`, but default tournament `MUTATION_SCALE = 0` (variety comes from the
   16 genomes, not intra-tile cohorts).
3. **Reset / seeding:** on reset, map particle index → tile
   (`config_index = int(floor(normalized_index * 16))`) and scatter the particle uniformly
   within that tile's box (tile origin + rand()*tile_size), so tiles start with equal
   particle counts confined to their cells.
4. **Boundary = per-tile bounce:** compute the tile box `[tile_min, tile_max]` in world
   space; if the integrated position leaves the box, reflect the offending velocity
   component and clamp position back into the box (per-tile version of the existing
   `edgeflect` bounce). Applies regardless of the global `BOUNDARY_CONDITIONS_MODE`.
5. **Sensor isolation:** clamp both sensor sample coordinates to the tile box before
   `get_can()`, so a particle near a border never senses a neighbor tile's trail.

New uniforms in `shaders/canvas.frag` (trail decay/diffusion): `TOURNAMENT_MODE`,
`TOURNAMENT_GRID`. When on, the 5-tap diffusion blur **skips (zeroes) any neighbor tap that
falls in a different tile** (compute each tap's tile, compare to center tile), giving hard
trail isolation at borders. `sim.py::calculate_setting` sync note updated accordingly.

### 3.3 Python changes
- **`services/tournament_service.py`** (new): owns `population` (16 genomes), `selected`
  set, `undo_stack`, `mutation_strength`, `inject_randoms`, `crossover_enabled`, an RNG,
  and a `dirty` flag. Methods: `init_population()`, `toggle_select(tile)`,
  `next_generation()`, `undo()`, `reset()`, `save_selected(dir)`, `write_to_buffer(sim)`
  (packs 16 genomes → the rule SSBO), `is_dirty()/clear_dirty()`.
- **`state/tournament_state.py`** (new dataclass): UI-facing state — `enabled`,
  `mutation_strength`, `inject_randoms`, `crossover_enabled`, plus one-shot command flags
  (`next_gen_requested`, `undo_requested`, `reset_requested`, `save_requested`,
  `clicked_tile`).
- **`sim.py`:** add `apply_tournament(service)`; in `entity_update`/`can_update`, set
  `TOURNAMENT_MODE`/`TOURNAMENT_GRID` uniforms; when tournament dirty, write the 16-rule
  buffer (like `_write_multi_load_ssbo` but 16 slots, rules only). On generation advance,
  call the existing `clear_canvas()` + force a particle reset (reuse `frame_count==0` reset
  path or a `force_reset` flag).
- **`main.py` / `command_handler.py`:** route tournament one-shot flags (next gen, undo,
  reset, save, tile click) to `TournamentService`, mirroring how existing one-shot commands
  are handled. When `tournament_state.enabled`, take the tournament sim path.
- **UI (`ui/`):** a new tournament window/overlay mixin — a 4×4 clickable grid aligned to
  the canvas tiles (click → `clicked_tile`), selection highlights, and controls:
  **Next Generation**, **Mutation strength** slider (0–0.5, default 0.15),
  **Inject randoms** slider (0–4, default 1), **Crossover** checkbox (default off),
  **Reset population**, **Undo generation**, **Save selected**.

## 4. Data flow (per frame, tournament on)

```
UI → TournamentState (enabled, sliders, one-shot flags: click/next/undo/reset/save)
command_handler applies one-shot flags → TournamentService
    click      → toggle_select(tile)
    next_gen   → next_generation(); clear canvas + reseed; mark dirty; push undo
    undo/reset → restore/replace population; clear canvas + reseed; mark dirty
    save       → save_selected() → physics_configs/*.json
sim.apply_tournament(service)
sim.update():
    if service.is_dirty(): write 16 genomes → rule SSBO; clear_dirty()
    entity_update: TOURNAMENT_MODE=1 → per-tile rule/bounce/sensor-clamp
    can_update:    TOURNAMENT_MODE=1 → per-tile isolated diffusion
camera renders full canvas → appears as a 4×4 grid
UI overlay reads service.selected for highlights
```

## 5. Testing / verification

Because this is GPU/visual, verification is a mix of unit tests (pure-Python logic) and
manual visual checks:
- **Unit (pytest):** `random_genome()` shape/dtype/range; `mutate()` changes values and
  respects `strength=0` (no-op) and scales with `strength`; `next_generation()` keeps
  selected genomes unchanged (elitism), produces exactly 16, injects the requested number
  of randoms; `undo()` restores prior population; genome↔config-JSON round-trips via
  `config_saver`.
- **Buffer packing:** `write_to_buffer` produces `16 * SIZE_OF_RULE_STRUCT` bytes matching
  the existing multi-load rule packing (compare against `_write_multi_load_ssbo` output for
  the same rules).
- **Manual visual:** toggle tournament mode → 16 visibly distinct panels; confirm hard
  isolation (no trail bleed at borders; particles stay in-tile); select a few → Next Gen →
  survivors visually persist and neighbors resemble them; Undo restores; Save produces a
  loadable config.
- **Regression:** with `TOURNAMENT_MODE=0`, output is unchanged from current `dist` behavior.

## 6. Open implementation decisions (resolved during coding, low-risk)

- Reuse `multi_load_rule_buffer` (binding 4) vs. a dedicated tournament rule buffer —
  prefer reuse; switch to dedicated only if multi-load and tournament states conflict.
- Exact reseed mechanism (reuse `frame_count==0` reset vs. a new `force_reset` uniform).
- Whether the tournament overlay is a separate ImGui window or drawn over the viewport —
  prefer drawn-over-viewport for precise tile alignment.
