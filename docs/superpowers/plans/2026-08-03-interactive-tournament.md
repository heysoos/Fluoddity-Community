# Interactive Tournament Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 4×4 grid of live, hard-isolated sub-simulations to Fluoddity where each tile runs a different Fourier-brain genome; the user clicks interesting tiles and the app breeds the next generation by keeping the selected genomes and mutating around them.

**Architecture:** Approach B — one `Sim`, one 1024² canvas partitioned into 16 spatial tiles. A particle's "home tile" is derived from its buffer index (stable), which selects that tile's genome from the existing 16-slot rule SSBO (binding 4), confines the particle via per-tile bounce boundaries, and clamps its trail sensing to the tile. `canvas.frag` blocks trail diffusion across tile borders. All evolutionary logic (population, mutation, selection, undo) lives in a pure-Python `TournamentService`; a `TournamentState` dataclass + ImGui window drive it via the existing one-shot-flag pattern. When `TOURNAMENT_MODE == 0`, every code path is byte-for-byte the current `dist` behavior.

**Tech Stack:** Python 3.12, ModernGL (OpenGL 4.5 compute + fragment GLSL), imgui-bundle, NumPy. Tests: pytest (added as dev dependency).

## Global Constraints

- Git branch: `dist` (the repo's default). Create a feature branch `tournament-mode` for this work.
- Genome representation everywhere: `numpy.ndarray` shape `(10, 8)` dtype `float32`. Columns 0–3 = FourierCenter `frequency`, columns 4–7 = `amplitude`. Flattened row-major, this exactly matches the GLSL `Rule { FourierCenter centers[10]; }` / `FourierCenter { vec4 frequency; vec4 amplitude; }` layout = 80 floats = `SIZE_OF_RULE_STRUCT` (320 bytes).
- Grid is fixed at 4×4 = 16 tiles. `TOURNAMENT_GRID = 4`, `TOURNAMENT_TILES = 16`.
- Tournament reuses the existing rule SSBO at binding 4 (`multi_load_rule_buffer`, sized `64 * SIZE_OF_RULE_STRUCT`); it writes the first 16 slots. Tournament and multi-load are mutually exclusive (never both enabled).
- Evolve the brain only. The 12 physics sliders stay at their single global values. While tournament is active, the effective per-particle `MUTATION_SCALE` is forced to 0 so all particles in a tile share the exact uploaded genome.
- `TOURNAMENT_MODE == 0` must leave existing behavior unchanged (regression-verified in Task 11).
- Determinism in tests: always construct `numpy.random.default_rng(seed)` with an explicit seed; never call global `np.random`.

---

### Task 1: Dev tooling — pytest harness

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `pytest.ini`

**Interfaces:**
- Produces: a runnable `pytest` setup so all later pure-Python tasks can be TDD'd. `tests/conftest.py` puts the repo root on `sys.path` so `from services... import ...` works.

- [ ] **Step 1: Add pytest to requirements**

Append to `requirements.txt`:
```
pytest>=8.0.0
```

- [ ] **Step 2: Create the tests package + path shim**

Create `tests/__init__.py` (empty file).

Create `tests/conftest.py`:
```python
import sys
from pathlib import Path

# Put the repo root on sys.path so `import services`, `import state`, etc. resolve
# when pytest is run from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

- [ ] **Step 3: Create pytest.ini**

Create `pytest.ini`:
```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -q
```

- [ ] **Step 4: Install and verify pytest collects nothing yet**

Run:
```bash
pip install pytest && python -m pytest
```
Expected: exit code 5 (`no tests ran`) — confirms pytest is installed and configured.

- [ ] **Step 5: Commit**

```bash
git checkout -b tournament-mode
git add requirements.txt tests/__init__.py tests/conftest.py pytest.ini
git commit -m "test: add pytest harness for tournament mode"
```

---

### Task 2: Genome operations (pure functions)

**Files:**
- Create: `services/genome.py`
- Test: `tests/test_genome.py`

**Interfaces:**
- Produces:
  - `random_genome(rng: np.random.Generator) -> np.ndarray` — shape `(10,8)` float32.
  - `mutate(genome: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray` — returns a new array; `strength == 0` is a no-op copy.
  - `crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray` — per-center 50/50 pick.
  - Module constants `N_CENTERS = 10`, `GENOME_SHAPE = (10, 8)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_genome.py`:
```python
import numpy as np
from services.genome import random_genome, mutate, crossover, GENOME_SHAPE


def test_random_genome_shape_and_dtype():
    rng = np.random.default_rng(0)
    g = random_genome(rng)
    assert g.shape == GENOME_SHAPE
    assert g.dtype == np.float32


def test_random_genome_ranges():
    rng = np.random.default_rng(1)
    g = random_genome(rng)
    freq, amp = g[:, :4], g[:, 4:]
    # amplitude in [-1, 1]; frequency in [-3, 3] (freq_scale in [1,3])
    assert amp.min() >= -1.0 and amp.max() <= 1.0
    assert freq.min() >= -3.0 and freq.max() <= 3.0


def test_random_genome_is_deterministic_per_seed():
    a = random_genome(np.random.default_rng(42))
    b = random_genome(np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_mutate_zero_strength_is_noop():
    rng = np.random.default_rng(2)
    g = random_genome(rng)
    out = mutate(g, 0.0, np.random.default_rng(2))
    assert np.array_equal(out, g)
    assert out is not g  # must be a copy


def test_mutate_changes_values_and_scales_with_strength():
    base = random_genome(np.random.default_rng(3))
    small = mutate(base, 0.05, np.random.default_rng(7))
    big = mutate(base, 0.5, np.random.default_rng(7))
    assert not np.array_equal(small, base)
    # Larger strength => larger amplitude deviation on average
    assert np.abs(big[:, 4:] - base[:, 4:]).mean() > np.abs(small[:, 4:] - base[:, 4:]).mean()


def test_crossover_takes_each_center_from_one_parent():
    a = np.zeros((10, 8), dtype=np.float32)
    b = np.ones((10, 8), dtype=np.float32)
    child = crossover(a, b, np.random.default_rng(5))
    # Every row must be all-zeros (from a) or all-ones (from b)
    for row in child:
        assert np.all(row == 0.0) or np.all(row == 1.0)
    assert child.dtype == np.float32
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_genome.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.genome'`.

- [ ] **Step 3: Implement the module**

Create `services/genome.py`:
```python
"""Pure genome operations for the interactive tournament.

A genome is a (10, 8) float32 array: columns 0-3 = FourierCenter frequency,
columns 4-7 = amplitude. This matches the GLSL Rule struct byte layout exactly.
The distributions mirror generate_random_centers() and mutate_rule() in
shaders/fourier4_4.glsl / shaders/entity_update.glsl (distribution-level, not
bit-exact — genomes are generated on the CPU and uploaded verbatim).
"""
import numpy as np

N_CENTERS = 10
GENOME_SHAPE = (N_CENTERS, 8)


def random_genome(rng: np.random.Generator) -> np.ndarray:
    """A fresh random brain. Frequencies biased toward low magnitude."""
    freq_scale = 1.0 + 2.0 * rng.random((N_CENTERS, 4)) ** 2      # [1, 3]
    freq = (rng.random((N_CENTERS, 4)) * 2.0 - 1.0) * freq_scale  # [-3, 3], low-biased
    amp = rng.random((N_CENTERS, 4)) * 2.0 - 1.0                  # [-1, 1]
    return np.concatenate([freq, amp], axis=1).astype(np.float32)


def mutate(genome: np.ndarray, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Additive jitter on amplitude, multiplicative jitter on frequency."""
    g = genome.astype(np.float32).copy()
    if strength == 0.0:
        return g
    freq = g[:, :4]
    amp = g[:, 4:]
    amp += strength * (-1.0 + 2.0 * rng.random((N_CENTERS, 4)))
    freq *= 1.0 + 0.5 * strength * (rng.random((N_CENTERS, 4)) - 0.5)
    return g.astype(np.float32)


def crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform per-center crossover: each of the 10 centers comes from a or b."""
    mask = rng.random(N_CENTERS) < 0.5           # True => take from a
    out = np.where(mask[:, None], a, b)
    return out.astype(np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_genome.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add services/genome.py tests/test_genome.py
git commit -m "feat: genome random/mutate/crossover operations"
```

---

### Task 3: TournamentService (population, selection, breeding, undo)

**Files:**
- Create: `services/tournament_service.py`
- Modify: `services/__init__.py`
- Test: `tests/test_tournament_service.py`

**Interfaces:**
- Consumes: `services.genome.random_genome/mutate/crossover`.
- Produces `TournamentService` with:
  - attrs: `population: list[np.ndarray]` (len 16), `selected: set[int]`, `mutation_strength: float` (default 0.15), `inject_randoms: int` (default 1), `crossover_enabled: bool` (default False), `initialized: bool`.
  - `TILES = 16` class constant.
  - `init_population()` — fill 16 random genomes, clear selection, set `initialized=True`, mark dirty.
  - `toggle_select(tile: int)` — add/remove tile from `selected`.
  - `next_generation()` — survivors (selected) stay pinned to their tile indices; empty tiles filled with `inject_randoms` fresh randoms then mutated offspring of the parent pool; pushes an undo snapshot; marks dirty.
  - `undo()` — restore previous population+selection from the undo stack (no-op if empty); marks dirty.
  - `reset()` — new all-random population; marks dirty.
  - `is_dirty() -> bool`, `mark_dirty()`, `clear_dirty()`.
  - `pack_rule_bytes() -> bytes` — the 16 genomes packed for the SSBO (used by Task 6; tested here).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tournament_service.py`:
```python
import numpy as np
from services.tournament_service import TournamentService
from services.genome import GENOME_SHAPE


def make_service(seed=0):
    svc = TournamentService(rng=np.random.default_rng(seed))
    svc.init_population()
    return svc


def test_init_population_makes_16_distinct_genomes():
    svc = make_service()
    assert len(svc.population) == svc.TILES == 16
    for g in svc.population:
        assert g.shape == GENOME_SHAPE and g.dtype == np.float32
    # Not all identical
    assert not np.array_equal(svc.population[0], svc.population[1])
    assert svc.initialized is True


def test_toggle_select():
    svc = make_service()
    svc.toggle_select(3)
    assert svc.selected == {3}
    svc.toggle_select(3)
    assert svc.selected == set()


def test_next_generation_keeps_selected_genomes_pinned():
    svc = make_service()
    svc.inject_randoms = 0
    svc.toggle_select(5)
    svc.toggle_select(10)
    keep5 = svc.population[5].copy()
    keep10 = svc.population[10].copy()
    svc.next_generation()
    assert len(svc.population) == 16
    assert np.array_equal(svc.population[5], keep5)   # survivor stays in its tile
    assert np.array_equal(svc.population[10], keep10)


def test_next_generation_injects_requested_random_count():
    svc = make_service()
    svc.inject_randoms = 3
    svc.toggle_select(0)                     # 1 survivor => 15 empty tiles
    before = [g.copy() for g in svc.population]
    svc.next_generation()
    # Count tiles whose genome is neither the pinned survivor nor a mutation of it
    # (indirect check: exactly `inject_randoms` tiles differ from any `before` genome
    #  by being freshly random is hard to assert; instead assert population size and
    #  that the survivor is pinned, plus that at least inject_randoms tiles changed).
    changed = sum(0 if np.array_equal(svc.population[i], before[i]) else 1 for i in range(16))
    assert changed >= svc.inject_randoms
    assert np.array_equal(svc.population[0], before[0])


def test_next_generation_with_no_selection_mutates_all():
    svc = make_service()
    svc.inject_randoms = 0
    before = [g.copy() for g in svc.population]
    svc.next_generation()
    # With nothing selected, every tile should change (whole population mutated)
    assert all(not np.array_equal(svc.population[i], before[i]) for i in range(16))


def test_undo_restores_previous_population():
    svc = make_service()
    snapshot = [g.copy() for g in svc.population]
    svc.next_generation()
    svc.undo()
    for i in range(16):
        assert np.array_equal(svc.population[i], snapshot[i])


def test_pack_rule_bytes_layout():
    svc = make_service()
    data = svc.pack_rule_bytes()
    assert len(data) == 16 * 320   # 16 genomes * 80 floats * 4 bytes
    arr = np.frombuffer(data, dtype=np.float32).reshape(16, 10, 8)
    assert np.array_equal(arr[7], svc.population[7])


def test_dirty_flag_lifecycle():
    svc = make_service()
    assert svc.is_dirty() is True     # init marks dirty
    svc.clear_dirty()
    assert svc.is_dirty() is False
    svc.toggle_select(1)
    svc.next_generation()
    assert svc.is_dirty() is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tournament_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.tournament_service'`.

- [ ] **Step 3: Implement the service**

Create `services/tournament_service.py`:
```python
"""TournamentService — interactive evolutionary selection over 16 brain genomes.

Human-in-the-loop: the user selects interesting tiles; next_generation() keeps
selected genomes pinned to their tiles and breeds the rest by mutating the
selected pool (plus a few fresh randoms for diversity). No fitness function.
"""
import copy
import numpy as np
from services.genome import random_genome, mutate, crossover, GENOME_SHAPE


class TournamentService:
    TILES = 16

    def __init__(self, rng: np.random.Generator | None = None):
        self._rng = rng if rng is not None else np.random.default_rng()
        self.population: list[np.ndarray] = [
            np.zeros(GENOME_SHAPE, dtype=np.float32) for _ in range(self.TILES)
        ]
        self.selected: set[int] = set()
        self.mutation_strength: float = 0.15
        self.inject_randoms: int = 1
        self.crossover_enabled: bool = False
        self.initialized: bool = False
        self._undo_stack: list[tuple[list[np.ndarray], set[int]]] = []
        self._dirty: bool = False

    # --- lifecycle ---
    def init_population(self) -> None:
        self.population = [random_genome(self._rng) for _ in range(self.TILES)]
        self.selected.clear()
        self._undo_stack.clear()
        self.initialized = True
        self.mark_dirty()

    def reset(self) -> None:
        self._push_undo()
        self.population = [random_genome(self._rng) for _ in range(self.TILES)]
        self.selected.clear()
        self.mark_dirty()

    # --- selection ---
    def toggle_select(self, tile: int) -> None:
        if not (0 <= tile < self.TILES):
            return
        if tile in self.selected:
            self.selected.remove(tile)
        else:
            self.selected.add(tile)

    # --- breeding ---
    def next_generation(self) -> None:
        self._push_undo()
        parents = ([self.population[i] for i in sorted(self.selected)]
                   if self.selected else list(self.population))

        new: list[np.ndarray | None] = [None] * self.TILES
        # Survivors stay pinned to their own tiles.
        for i in self.selected:
            new[i] = self.population[i].copy()

        empty = [i for i in range(self.TILES) if new[i] is None]
        self._rng.shuffle(empty)
        n_random = max(0, min(self.inject_randoms, len(empty)))

        for j, tile in enumerate(empty):
            if j < n_random:
                new[tile] = random_genome(self._rng)
            else:
                p = parents[self._rng.integers(len(parents))]
                if self.crossover_enabled and len(parents) >= 2:
                    q = parents[self._rng.integers(len(parents))]
                    child = crossover(p, q, self._rng)
                else:
                    child = p
                new[tile] = mutate(child, self.mutation_strength, self._rng)

        self.population = [g for g in new]  # all slots filled
        self.mark_dirty()

    def undo(self) -> None:
        if not self._undo_stack:
            return
        pop, sel = self._undo_stack.pop()
        self.population = [g.copy() for g in pop]
        self.selected = set(sel)
        self.mark_dirty()

    def _push_undo(self) -> None:
        self._undo_stack.append(([g.copy() for g in self.population], set(self.selected)))
        if len(self._undo_stack) > 50:
            self._undo_stack.pop(0)

    # --- GPU packing ---
    def pack_rule_bytes(self) -> bytes:
        data = bytearray()
        for g in self.population:
            data.extend(np.ascontiguousarray(g, dtype=np.float32).tobytes())
        return bytes(data)

    # --- dirty flag ---
    def is_dirty(self) -> bool:
        return self._dirty

    def mark_dirty(self) -> None:
        self._dirty = True

    def clear_dirty(self) -> None:
        self._dirty = False
```

- [ ] **Step 4: Export from the services package**

In `services/__init__.py`, add `TournamentService` to the imports/`__all__` alongside the existing services (match the file's existing style — add a line `from .tournament_service import TournamentService` and include `"TournamentService"` in `__all__` if present).

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_tournament_service.py -v`
Expected: PASS (8 passed).

- [ ] **Step 6: Commit**

```bash
git add services/tournament_service.py services/__init__.py tests/test_tournament_service.py
git commit -m "feat: TournamentService population/selection/breeding/undo"
```

---

### Task 4: TournamentState + UIState wiring

**Files:**
- Create: `state/tournament_state.py`
- Modify: `state/__init__.py`
- Modify: `state/ui_state.py`
- Test: `tests/test_tournament_state.py`

**Interfaces:**
- Produces `TournamentState` dataclass fields: `enabled: bool = False`, `mutation_strength: float = 0.15`, `inject_randoms: int = 1`, `crossover_enabled: bool = False`, and one-shot flags `clicked_tile: int = -1`, `next_gen_requested: bool = False`, `undo_requested: bool = False`, `reset_requested: bool = False`, `save_requested: bool = False`.
- `UIState` gains `tournament: TournamentState` field.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tournament_state.py`:
```python
from state.tournament_state import TournamentState
from state.ui_state import UIState


def test_tournament_state_defaults():
    ts = TournamentState()
    assert ts.enabled is False
    assert ts.mutation_strength == 0.15
    assert ts.inject_randoms == 1
    assert ts.crossover_enabled is False
    assert ts.clicked_tile == -1
    assert ts.next_gen_requested is False
    assert ts.undo_requested is False
    assert ts.reset_requested is False
    assert ts.save_requested is False


def test_uistate_has_independent_tournament_states():
    a = UIState()
    b = UIState()
    a.tournament.enabled = True
    assert b.tournament.enabled is False   # default_factory, not shared
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tournament_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'state.tournament_state'`.

- [ ] **Step 3: Create the dataclass**

Create `state/tournament_state.py`:
```python
"""State for the interactive tournament feature."""
from dataclasses import dataclass


@dataclass
class TournamentState:
    """UI-facing tournament state + one-shot command flags."""
    # Persistent controls
    enabled: bool = False
    mutation_strength: float = 0.15   # 0.0 - 0.5
    inject_randoms: int = 1           # 0 - 4 fresh randoms per generation
    crossover_enabled: bool = False

    # One-shot flags (reset after UI.get_state each frame)
    clicked_tile: int = -1            # -1 = no click this frame; else tile 0..15
    next_gen_requested: bool = False
    undo_requested: bool = False
    reset_requested: bool = False
    save_requested: bool = False
```

- [ ] **Step 4: Export it**

In `state/__init__.py`, add `from .tournament_state import TournamentState` and include it in `__all__` if the file defines one (match existing style).

- [ ] **Step 5: Add the field to UIState**

In `state/ui_state.py`:
- Add import near the other state imports (line ~6): `from .tournament_state import TournamentState`
- Add the field inside `UIState` next to `multi_load` (line ~16):
```python
    tournament: TournamentState = field(default_factory=TournamentState)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_tournament_state.py -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add state/tournament_state.py state/__init__.py state/ui_state.py tests/test_tournament_state.py
git commit -m "feat: TournamentState and UIState wiring"
```

---

### Task 5: Physics shader — tournament tiling in entity_update.glsl

**Files:**
- Modify: `shaders/entity_update.glsl`
- Test: `tests/test_shader_source.py`

**Interfaces:**
- Consumes: uniforms `TOURNAMENT_MODE` (int), `TOURNAMENT_GRID` (int), set by Task 6.
- Produces: when `TOURNAMENT_MODE == 1`, each particle uses its index-derived home tile for genome selection, reset scatter, bounce boundary, and sensor clamping. When `TOURNAMENT_MODE == 0`, the shader is unchanged.

This task can't be unit-tested on the GPU here; verification = a source-substring test (guards the edits stay present) plus the runtime compile + visual check in Task 10/11.

- [ ] **Step 1: Write the failing source test**

Create `tests/test_shader_source.py`:
```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(p):
    return (ROOT / p).read_text()


def test_entity_update_has_tournament_hooks():
    src = read("shaders/entity_update.glsl")
    assert "uniform int TOURNAMENT_MODE;" in src
    assert "uniform int TOURNAMENT_GRID;" in src
    assert "tournament_home_tile" in src
    assert "tournament_tile_box" in src


def test_canvas_frag_has_tournament_isolation():
    src = read("shaders/canvas.frag")
    assert "uniform int TOURNAMENT_MODE;" in src
    assert "uniform int TOURNAMENT_GRID;" in src
```

- [ ] **Step 2: Run test to verify the entity_update assertion fails**

Run: `python -m pytest tests/test_shader_source.py::test_entity_update_has_tournament_hooks -v`
Expected: FAIL — substrings not found.

- [ ] **Step 3: Add the uniforms**

In `shaders/entity_update.glsl`, immediately after the line `uniform bool WRITE_RULES; // Set true for one frame when rule buffer readback is needed` (line ~61) add:
```glsl
// Tournament mode: partition the canvas into a TOURNAMENT_GRID x TOURNAMENT_GRID grid
uniform int TOURNAMENT_MODE;   // 0 = off, 1 = on
uniform int TOURNAMENT_GRID;   // grid side length (4 => 16 tiles)
```

- [ ] **Step 4: Add the tournament helper functions**

In `shaders/entity_update.glsl`, immediately after the `get_particle_config_index()` function closes (after its `}` at line ~133) add:
```glsl
// Tournament: a particle's stable "home tile" is derived from its buffer index,
// so a particle never migrates between tiles even if it drifts spatially.
int tournament_home_tile(uint index){
    int n = TOURNAMENT_GRID * TOURNAMENT_GRID;
    int tile = int(floor(float(index) / float(ACTIVE_COUNT) * float(n)));
    return clamp(tile, 0, n - 1);
}
// Entity-space bounding box [lo, hi] of a tile index.
void tournament_tile_box(int tile, out vec2 lo, out vec2 hi){
    float ca = canvas_resolution.x / canvas_resolution.y;
    vec2 half_extent = vec2(sqrt(ca), 1.0 / sqrt(ca));
    int tx = tile % TOURNAMENT_GRID;
    int ty = tile / TOURNAMENT_GRID;
    vec2 cell = (2.0 * half_extent) / float(TOURNAMENT_GRID);
    lo = -half_extent + vec2(float(tx), float(ty)) * cell;
    hi = lo + cell;
}
```

- [ ] **Step 5: Override genome selection for tournament**

In `main()`, the block currently reads (line ~473):
```glsl
    Rule current_rule=get_particle_target_rule();
    //if a few arbitrary coefficients are exactly 0, then assume target_rule is all 0s (no target) and generate a random rule instead.
    if(current_rule.centers[0].frequency==vec4(0) && current_rule.centers[5].amplitude==vec4(0)){
        current_rule = Rule(generate_random_centers(get_particle_rule_seed()+floor(cohort)));
    }
```
Replace it with:
```glsl
    Rule current_rule=get_particle_target_rule();
    if(TOURNAMENT_MODE == 1){
        current_rule = target_rules[tournament_home_tile(index)];
    }
    //if a few arbitrary coefficients are exactly 0, then assume target_rule is all 0s (no target) and generate a random rule instead.
    else if(current_rule.centers[0].frequency==vec4(0) && current_rule.centers[5].amplitude==vec4(0)){
        current_rule = Rule(generate_random_centers(get_particle_rule_seed()+floor(cohort)));
    }
```

- [ ] **Step 6: Scatter particles into their home tile on reset**

In `reset(uint index)`, the last line currently is (line ~384):
```glsl
    //store to persistent entity buffer
    entities[index]=Entity(pos,vel,size,cohort_val/float(cohorts),float[2](0,0),color);
```
Insert BEFORE it:
```glsl
    //Tournament: place the particle uniformly inside its home tile (with a small margin).
    if(TOURNAMENT_MODE == 1){
        int htile = tournament_home_tile(index);
        vec2 lo, hi; tournament_tile_box(htile, lo, hi);
        vec2 margin = (hi - lo) * 0.04;
        lo += margin; hi -= margin;
        vec2 r = vec2(hash(vec2(cohort_val, float(index)+0.1)),
                      hash(vec2(float(index)+0.2, cohort_val)));
        pos = mix(lo, hi, r);
    }
```

- [ ] **Step 7: Clamp sensors to the home tile**

In `main()`, the sensor reads currently are (line ~508):
```glsl
    //read the trails from canvas
    vec4 ltap = get_can(e.pos+left_sensor_offset);
    vec4 rtap = get_can(e.pos+right_sensor_offset);
```
Replace with:
```glsl
    //read the trails from canvas (tournament: keep sample points inside the home tile)
    vec2 lsample = e.pos + left_sensor_offset;
    vec2 rsample = e.pos + right_sensor_offset;
    if(TOURNAMENT_MODE == 1){
        vec2 tlo, thi; tournament_tile_box(tournament_home_tile(index), tlo, thi);
        lsample = clamp(lsample, tlo, thi);
        rsample = clamp(rsample, tlo, thi);
    }
    vec4 ltap = get_can(lsample);
    vec4 rtap = get_can(rsample);
```

- [ ] **Step 8: Per-tile bounce boundary**

In `main()`, immediately AFTER the existing boundary block closes (after the `else if(boundary_mode==2){...}` block ends, line ~580) and BEFORE `//Commit new entity state to buffers`, insert:
```glsl
    //Tournament: override world boundaries with per-tile bounce so tiles stay isolated.
    if(TOURNAMENT_MODE == 1){
        vec2 tlo, thi; tournament_tile_box(tournament_home_tile(index), tlo, thi);
        if(e.pos.x < tlo.x){ e.pos.x = tlo.x; e.vel.x = abs(e.vel.x); }
        if(e.pos.x > thi.x){ e.pos.x = thi.x; e.vel.x = -abs(e.vel.x); }
        if(e.pos.y < tlo.y){ e.pos.y = tlo.y; e.vel.y = abs(e.vel.y); }
        if(e.pos.y > thi.y){ e.pos.y = thi.y; e.vel.y = -abs(e.vel.y); }
    }
```

- [ ] **Step 9: Run the source test to verify it passes**

Run: `python -m pytest tests/test_shader_source.py::test_entity_update_has_tournament_hooks -v`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add shaders/entity_update.glsl tests/test_shader_source.py
git commit -m "feat: tournament tiling in entity_update compute shader"
```

---

### Task 6: Trail isolation in canvas.frag

**Files:**
- Modify: `shaders/canvas.frag`

**Interfaces:**
- Consumes: uniforms `TOURNAMENT_MODE`, `TOURNAMENT_GRID` (set by Task 7).
- Produces: diffusion blur that does not cross tile borders when tournament is on.

- [ ] **Step 1: Add uniforms**

In `shaders/canvas.frag`, after `uniform bool tiling_mode;` (line ~29) add:
```glsl
// Tournament isolation
uniform int TOURNAMENT_MODE;   // 0 = off, 1 = on
uniform int TOURNAMENT_GRID;   // grid side length
```

- [ ] **Step 2: Add a tile helper and isolate the blur**

In `shaders/canvas.frag`, replace the entire `getBlur` function (lines ~112–125) with:
```glsl
// Which tile (in uv space) a texcoord belongs to. -1 when tournament is off.
int tournament_tile_uv(vec2 uv){
    if(TOURNAMENT_MODE != 1) return -1;
    ivec2 t = ivec2(floor(clamp(uv, 0.0, 0.999999) * float(TOURNAMENT_GRID)));
    return t.y * TOURNAMENT_GRID + t.x;
}

vec4 getBlur(vec2 pos, sampler2D sam, float diffusion_constant) {
    ivec2 imsz = textureSize(sam, 0);
    vec3 off = vec3(1. / vec2(imsz), 0);
    vec2 np = pos + off.zy;
    vec2 sp = pos - off.zy;
    vec2 wp = pos - off.xz;
    vec2 ep = pos + off.xz;
    vec4 cc = getCan(pos, sam);
    vec4 nc = getCan(np, sam);
    vec4 sc = getCan(sp, sam);
    vec4 wc = getCan(wp, sam);
    vec4 ec = getCan(ep, sam);
    // Tournament: zero-flux at tile borders — a neighbor in another tile is
    // replaced by the center value so no trail energy crosses the seam.
    if(TOURNAMENT_MODE == 1){
        int ct = tournament_tile_uv(pos);
        if(tournament_tile_uv(np) != ct) nc = cc;
        if(tournament_tile_uv(sp) != ct) sc = cc;
        if(tournament_tile_uv(wp) != ct) wc = cc;
        if(tournament_tile_uv(ep) != ct) ec = cc;
    }
    float K = diffusion_constant;
    return (cc * K + nc + sc + wc + ec) / (4. + K);
}
```

- [ ] **Step 3: Run the canvas source test to verify it passes**

Run: `python -m pytest tests/test_shader_source.py::test_canvas_frag_has_tournament_isolation -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add shaders/canvas.frag
git commit -m "feat: per-tile trail isolation in canvas.frag"
```

---

### Task 7: Sim wiring — uniforms, rule upload, tournament toggle

**Files:**
- Modify: `sim.py`
- Test: `tests/test_sim_tournament_pack.py`

**Interfaces:**
- Consumes: `TournamentService.pack_rule_bytes()`.
- Produces on `Sim`:
  - attrs `_tournament_enabled: bool = False`, `_tournament_grid: int = 4` (init in `__init__`).
  - `apply_tournament(self, enabled: bool, grid: int = 4)` — store flags.
  - `write_tournament_rules(self, rule_bytes: bytes)` — write the packed 16 genomes into `multi_load_rule_buffer`.
  - uniform sets in `entity_update` and `can_update`: `TOURNAMENT_MODE`, `TOURNAMENT_GRID`; when enabled, force `MUTATION_SCALE_SETTING.slider_value = 0`.

- [ ] **Step 1: Write the failing packing test**

Create `tests/test_sim_tournament_pack.py`:
```python
import numpy as np
from services.tournament_service import TournamentService


def test_pack_matches_ssbo_expectations():
    svc = TournamentService(rng=np.random.default_rng(0))
    svc.init_population()
    data = svc.pack_rule_bytes()
    # 16 rules * SIZE_OF_RULE_STRUCT (320 bytes) — matches sim.py constant
    from sim import SIZE_OF_RULE_STRUCT
    assert SIZE_OF_RULE_STRUCT == 320
    assert len(data) == 16 * SIZE_OF_RULE_STRUCT
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sim_tournament_pack.py -v`
Expected: FAIL — `sim` import pulls in moderngl but the assertion on layout is the real check; if `sim` imports cleanly it will fail only if lengths mismatch. If `import sim` errors in the headless test env, mark this test to import lazily — but `sim.py` imports moderngl at top which is installed, so import succeeds. Expected failure here is the missing behavior in later steps is N/A; this test actually passes once Task 3 exists. If it already passes, proceed (it guards the constant).

- [ ] **Step 3: Add tournament attributes to `__init__`**

In `sim.py`, in `Sim.__init__`, after `self._pending_entity_id = None` (line ~34) add:
```python
        # Tournament mode
        self._tournament_enabled = False
        self._tournament_grid = 4
```

- [ ] **Step 4: Add apply_tournament and write_tournament_rules methods**

In `sim.py`, add these methods to the `Sim` class (place them just after `apply_rule`, near line ~716):
```python
    def apply_tournament(self, enabled: bool, grid: int = 4) -> None:
        """Enable/disable tournament tiling for the next update."""
        self._tournament_enabled = enabled
        self._tournament_grid = grid

    def write_tournament_rules(self, rule_bytes: bytes) -> None:
        """Upload 16 packed genomes into the (reused) multi-load rule buffer."""
        self.multi_load_rule_buffer.write(rule_bytes)
```

- [ ] **Step 5: Set the compute-shader uniforms**

In `sim.py::entity_update`, just before `num_workgroups = (self.entity_count + 63) // 64` (line ~219) add:
```python
        # Tournament tiling uniforms
        tryset(self.entity_update_program, 'TOURNAMENT_MODE', 1 if self._tournament_enabled else 0)
        tryset(self.entity_update_program, 'TOURNAMENT_GRID', self._tournament_grid)
        if self._tournament_enabled:
            # Force per-particle mutation off so all particles in a tile share the genome
            tryset(self.entity_update_program, 'MUTATION_SCALE_SETTING.slider_value', 0.0)
```

- [ ] **Step 6: Set the canvas-shader uniforms**

In `sim.py::can_update`, just before the `if strong_determinism:` block (line ~309) add:
```python
        # Tournament tiling uniforms (trail isolation)
        tryset(self.canvas_update_program, 'TOURNAMENT_MODE', 1 if self._tournament_enabled else 0)
        tryset(self.canvas_update_program, 'TOURNAMENT_GRID', self._tournament_grid)
```

- [ ] **Step 7: Run the packing test**

Run: `python -m pytest tests/test_sim_tournament_pack.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add sim.py tests/test_sim_tournament_pack.py
git commit -m "feat: sim tournament uniforms and rule upload"
```

---

### Task 8: Command routing — drive the service each frame

**Files:**
- Modify: `main.py`
- Modify: `command_handler.py`

**Interfaces:**
- Consumes: `UIState.tournament`, `TournamentService`, `Sim.apply_tournament/write_tournament_rules/reset`.
- Produces: per-frame flow that (a) applies enabled/grid to the sim, (b) processes tournament one-shot flags, (c) on any dirty change uploads genomes and clears+reseeds the canvas.

- [ ] **Step 1: Construct the service and pass it in**

In `main.py::App.__init__`, after `self.multi_load_service = MultiLoadService()` (line ~69) add:
```python
        from services import TournamentService
        self.tournament_service = TournamentService()
```
Add `tournament_service=self.tournament_service` to the `CommandHandler(...)` constructor call (line ~85) as a new keyword argument, and expose it to the UI after `self.ui.multi_load_service = self.multi_load_service` (line ~71):
```python
        self.ui.tournament_service = self.tournament_service
```

- [ ] **Step 2: Accept the service in CommandHandler**

In `command_handler.py::CommandHandler.__init__`, add `tournament_service=None` to the signature (after `param_lock_service=None`) and store it:
```python
        self.tournament_service = tournament_service
```

- [ ] **Step 3: Apply tournament state to the sim each frame**

In `main.py::orchestrate_frame`, after `self.multi_load_service.apply_state(ui_state.multi_load)` (line ~226) add:
```python
        self.sim.apply_tournament(ui_state.tournament.enabled, grid=4)
```

- [ ] **Step 4: Add the tournament command handler**

In `command_handler.py::process_commands`, after `self._handle_clipboard_commands(ui_state)` (line ~147) and before `return None` add:
```python
        # Tournament mode
        self._handle_tournament(ui_state)
```
Then add this method to the class (place after `_handle_mouse_clicks`):
```python
    def _handle_tournament(self, ui_state):
        """Drive the TournamentService from tournament one-shot flags."""
        svc = self.tournament_service
        if svc is None:
            return
        ts = ui_state.tournament

        # Sync persistent controls
        svc.mutation_strength = ts.mutation_strength
        svc.inject_randoms = ts.inject_randoms
        svc.crossover_enabled = ts.crossover_enabled

        if not ts.enabled:
            return

        # Lazy-init the population the first time tournament turns on
        if not svc.initialized:
            svc.init_population()

        if ts.clicked_tile >= 0:
            svc.toggle_select(ts.clicked_tile)
        if ts.next_gen_requested:
            svc.next_generation()
        if ts.undo_requested:
            svc.undo()
        if ts.reset_requested:
            svc.reset()
        if ts.save_requested:
            self._save_tournament_selection(ui_state)

        # Upload + clear/reseed whenever the population changed
        if svc.is_dirty():
            self.sim.write_tournament_rules(svc.pack_rule_bytes())
            self.sim.reset()   # clears canvas + frame_count=0 => reseed into tiles
            svc.clear_dirty()

    def _save_tournament_selection(self, ui_state):
        """Save each selected genome to a config JSON in the user configs dir."""
        svc = self.tournament_service
        if not svc.selected:
            print("Tournament save: no tiles selected")
            return
        for tile in sorted(svc.selected):
            config = self.config_saver.create_config(ui_state.sim, svc.population[tile])
            filepath = self.user_configs_dir / f"tournament_tile{tile}.json"
            self.config_saver.save_to_file(config, filepath)
            print(f"Saved tournament tile {tile} -> {filepath}")
```

- [ ] **Step 5: Route canvas clicks to tile selection**

In `command_handler.py::_handle_mouse_clicks`, at the very top of the method (before the existing `if ui_state.left_click_this_frame:`) add:
```python
        # Tournament mode: left click selects the tile under the cursor
        if ui_state.tournament.enabled and ui_state.left_click_this_frame:
            tex = self.camera.screen_to_tex(ui_state.mouse_pos, self.sim.view_tex.size)
            grid = 4
            tx = min(grid - 1, max(0, int(tex[0] * grid)))
            ty = min(grid - 1, max(0, int(tex[1] * grid)))
            ui_state.tournament.clicked_tile = ty * grid + tx
            return
```
(Note: the y-orientation of `screen_to_tex` vs the shader's tile math is validated visually in Task 10; if selection is vertically mirrored, change `ty` to `int((1.0 - tex[1]) * grid)`.)

- [ ] **Step 6: Reset the tournament one-shot flags each frame**

In `ui/core.py`, find `get_state` (the method that returns `self.state` and clears one-shot flags each frame). After the existing one-shot resets (where flags like `request_reset` are set back to `False`), add:
```python
        self.state.tournament.clicked_tile = -1
        self.state.tournament.next_gen_requested = False
        self.state.tournament.undo_requested = False
        self.state.tournament.reset_requested = False
        self.state.tournament.save_requested = False
```
(Match the exact place/pattern the method already uses — the existing code sets other `request_*` flags to `False` right before `return`. Do not reset `enabled`, `mutation_strength`, `inject_randoms`, or `crossover_enabled` — those are persistent.)

- [ ] **Step 7: Verify the app still imports and runs (smoke)**

Run:
```bash
python -c "import main; print('import ok')"
```
Expected: `import ok` (no import/syntax errors). Full run verified in Task 10.

- [ ] **Step 8: Commit**

```bash
git add main.py command_handler.py ui/core.py
git commit -m "feat: route tournament commands and canvas-click selection"
```

---

### Task 9: Tournament UI window + menu toggle

**Files:**
- Create: `ui/tournament_window.py`
- Modify: `ui/core.py`
- Modify: `ui/menu_bar.py`

**Interfaces:**
- Consumes: `self.state.tournament` (TournamentState), `self.tournament_service` (for selection highlight display).
- Produces: `TournamentWindowMixin.render_tournament_window()` — a window with a 4×4 selectable button grid + controls; called from the main UI render loop. A "Tournament" checkbox in the menu bar toggles `self.state.tournament.enabled`.

- [ ] **Step 1: Create the mixin**

Create `ui/tournament_window.py`:
```python
"""Tournament mode UI: 4x4 tile selector + breeding controls."""
from imgui_bundle import imgui


class TournamentWindowMixin:
    """Renders the interactive tournament window. Mixed into UI."""

    def render_tournament_window(self):
        state = self.state.tournament
        if not state.enabled:
            return

        svc = getattr(self, "tournament_service", None)
        selected = svc.selected if svc is not None else set()

        imgui.begin("Tournament")

        imgui.text("Click tiles (canvas or grid below) to select. Then breed.")
        imgui.separator()

        # 4x4 selectable grid mirroring the on-canvas tiles.
        grid = 4
        for ty in range(grid):
            for tx in range(grid):
                tile = ty * grid + tx
                is_sel = tile in selected
                if is_sel:
                    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.2, 0.7, 0.3, 1.0))
                label = f"{tile}"
                if imgui.button(label, imgui.ImVec2(40, 40)):
                    state.clicked_tile = tile
                if is_sel:
                    imgui.pop_style_color()
                if tx < grid - 1:
                    imgui.same_line()

        imgui.separator()

        changed, val = imgui.slider_float("Mutation strength", state.mutation_strength, 0.0, 0.5)
        if changed:
            state.mutation_strength = val

        changed, val = imgui.slider_int("Inject randoms", state.inject_randoms, 0, 4)
        if changed:
            state.inject_randoms = val

        changed, val = imgui.checkbox("Crossover", state.crossover_enabled)
        if changed:
            state.crossover_enabled = val

        imgui.separator()

        if imgui.button("Next Generation"):
            state.next_gen_requested = True
        imgui.same_line()
        if imgui.button("Undo"):
            state.undo_requested = True

        if imgui.button("Reset Population"):
            state.reset_requested = True
        imgui.same_line()
        if imgui.button("Save Selected"):
            state.save_requested = True

        imgui.text(f"Selected: {sorted(selected)}")
        imgui.end()
```

- [ ] **Step 2: Mix it into UI and call it in the render loop**

In `ui/core.py`:
- Add the import near the other mixin imports (line ~27): `from .tournament_window import TournamentWindowMixin`
- Add `TournamentWindowMixin,` to the `class UI(...)` base list (line ~38, alongside the other mixins).
- In the UI's main `render()` method, where the other windows are rendered (e.g. near `render_history_window()` / `render_physics_window()` calls), add:
```python
        self.render_tournament_window()
```

- [ ] **Step 3: Add the menu-bar toggle**

In `ui/menu_bar.py`, inside the menu where other feature toggles live (e.g. the "Extras" menu that toggles `show_history_window`), add a checkbox that toggles `self.state.tournament.enabled`:
```python
            clicked, new_val = imgui.checkbox("Tournament Mode", self.state.tournament.enabled)
            if clicked:
                self.state.tournament.enabled = new_val
```
(Place it alongside the existing `imgui.checkbox`/`imgui.menu_item` calls in that menu; match the surrounding style. If the file uses `imgui.menu_item` with a selected flag instead of `imgui.checkbox`, follow that idiom.)

- [ ] **Step 4: Smoke-test the import**

Run:
```bash
python -c "import ui.core; print('ui import ok')"
```
Expected: `ui import ok`.

- [ ] **Step 5: Commit**

```bash
git add ui/tournament_window.py ui/core.py ui/menu_bar.py
git commit -m "feat: tournament UI window and menu toggle"
```

---

### Task 10: Live verification — run the app and drive a tournament

**Files:** none (manual verification). Requires an NVIDIA/OpenGL-4.5 capable machine with a display.

- [ ] **Step 1: Launch**

Run:
```bash
python main.py
```
Expected: the app opens as normal (single sim) — confirms `TOURNAMENT_MODE == 0` path is intact and shaders compile.

- [ ] **Step 2: Enable tournament**

Open the menu → check "Tournament Mode". The Tournament window appears and the canvas resolves into a **4×4 grid of visibly distinct panels** (16 different genomes). Confirm particles stay within their tiles and there is **no trail bleed across tile borders**.

- [ ] **Step 3: Verify selection alignment**

Click a panel on the canvas; confirm the corresponding numbered button in the Tournament window highlights (and `Selected:` updates). If selection is vertically mirrored relative to where you clicked, apply the y-flip noted in Task 8 Step 5, reload (R / restart), and re-verify.

- [ ] **Step 4: Breed a generation**

Select 2–3 interesting panels → click **Next Generation**. Confirm: the selected panels **remain in place unchanged**, the other tiles are filled with mutations of the selection (family resemblance) plus ~`Inject randoms` fresh-random panels, and the canvas cleared and reseeded.

- [ ] **Step 5: Undo, reset, save**

- Click **Undo** → previous generation returns.
- Click **Reset Population** → 16 brand-new random panels.
- Select a panel → **Save Selected** → confirm `tournament_tile<N>.json` appears in the user configs dir (printed path in the console) and can be loaded via the normal File→Load menu.

- [ ] **Step 6: Adjust mutation strength**

Set Mutation strength to ~0.05 (subtle variety) and ~0.4 (wild variety); confirm Next Generation reflects the change.

- [ ] **Step 7: Record findings**

Note any visual issues (tile density, isolation seams, selection alignment). File follow-ups if needed. No commit (manual task).

---

### Task 11: Regression check — tournament OFF is unchanged

**Files:** none (manual verification) + run the full pytest suite.

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest`
Expected: all tests pass.

- [ ] **Step 2: Visual regression**

Launch `python main.py` with tournament **off**; load a couple of `physics_configs` presets (e.g. from Core). Confirm behavior/appearance is identical to `dist` before this branch (single canvas, cohorts, sweeps, multi-load all still work). Toggle tournament on then off again and confirm the app returns cleanly to normal single-sim mode.

- [ ] **Step 3: Final commit / branch ready**

```bash
git add -A
git commit -m "docs: tournament mode verification notes" --allow-empty
```
Branch `tournament-mode` is ready for review/merge.

---

## Execution Log (2026-08-03)

Tasks 1–9 implemented and committed on branch `tournament-mode`. 24 tests pass.
Four deviations from the plan as written, all deliberate:

1. **Environment (pre-Task 1).** The freshly-cloned repo had no usable interpreter
   (system 3.12 was bare, conda was 3.9 — too old for the codebase's `X | None`
   runtime annotations). Created a project venv at `.venv/` on Python 3.12 and
   installed `requirements.txt`. Added `.venv/`, `__pycache__/`, `*.pyc`,
   `.pytest_cache/` to `.gitignore`. Run tests with `./.venv/Scripts/python.exe -m pytest`.

2. **Task 2 — import cycle.** A cold `import services.genome` fails because of a
   *pre-existing* cycle: `services/__init__` → `config_saver` → `ui.physics_params`
   → `ui/__init__` → `ui.core` → `services.config_saver` (partially initialized).
   It only resolves when `ui` is imported before `services`, which is what `main.py`
   happens to do. Fixed test-side only: `tests/conftest.py` primes the order with
   `import ui`. No application code changed.

3. **Task 8 Step 6 — plan bug, corrected.** The plan said to clear the tournament
   one-shot flags at the end of `UI.get_state()`. That would be wrong: `get_state()`
   returns `self.state` (the live object) and only clears its *private* `_request_*`
   mirrors — never the state fields. Clearing there would wipe each flag before
   `CommandHandler` ever read it. Instead the flags are cleared on the consuming side,
   in `CommandHandler._clear_tournament_flags()`, called from `_handle_tournament()`
   after consumption. `ui/core.py`'s `get_state()` was left untouched.

4. **Task 8 Step 5 / Task 10 Step 3 — no y-flip needed.** The plan flagged the
   click→tile y-orientation as uncertain. Resolved analytically: `camera.screen_to_tex`
   returns `tex_y == 0` at the **bottom** of the canvas (its docstring claiming
   "top-left" is inaccurate; `services/entity_picker.py` uses the same
   bottom-origin convention), and `tournament_tile_box()` places tile 0 at the
   bottom-left of entity space. The two already agree, so the fallback flip was
   **not** applied. Locked in by `tests/test_tournament_tile_mapping.py` (5 tests),
   which mirrors both mappings in Python and asserts every tile round-trips.

**Automated verification performed beyond the plan:**
- Headless GLSL compile check of both modified shaders against a real GL 4.5 context
  (Intel UHD Graphics): `entity_update.glsl` and `canvas.frag` both compile.
- 25-second headless-of-interaction boot run of `main.py`: ran the full main loop
  with no errors or crashes (tournament off — regression path healthy).

**Still requires a human at the machine:** Task 10 (all steps) and Task 11 Step 2 —
the visual confirmation that 16 distinct panels appear, that tiles are isolated,
and that breeding behaves as intended.

## Self-Review Notes (author)

- **Spec coverage:** genome+population (Tasks 2–3), brain-only evolution & forced MUTATION_SCALE=0 (Tasks 5,7), GLSL tournament branch with home-tile rule/seed/bounce/sensor-clamp (Task 5), canvas.frag hard isolation (Task 6), sim uniforms + rule upload (Task 7), clear+reseed each generation (Task 8 `sim.reset()`), 4×4 pick-any-K UI + elitism + inject randoms + crossover-optional + undo + save (Tasks 3,8,9), `TOURNAMENT_MODE==0` regression (Task 11). All spec sections map to a task.
- **Home-tile vs position-tile:** the plan uses index-derived home tiles (stable) for rule/seed/bounce/sensor, and spatial (uv) tiles only in `canvas.frag`; reset scatters index-tile particles into the matching spatial box, so the two coincide. This is a deliberate refinement of the spec's "tile by position" and is strictly more robust (no cross-tile migration).
- **Placeholders:** none — every code step has complete code.
- **Type consistency:** genome `(10,8)` float32 and `pack_rule_bytes()`/`write_tournament_rules(bytes)` are consistent across Tasks 3/7/8; `TournamentState` field names match the flags read in Task 8 and written in Task 9.
