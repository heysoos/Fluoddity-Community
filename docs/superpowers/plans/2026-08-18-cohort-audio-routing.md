# Cohort Audio Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let each audio mapping name which cohorts it drives, painted by dragging across a strip, so bass can add to one half of the swarm's Sensor Gain while highs subtract from the other half.

**Architecture:** A per-mapping 144-slot mask over the normalised cohort axis. The CPU reduces every masked mapping into two per-cohort numbers — a gain and an offset — and uploads them as one SSBO. `entity_update.glsl` applies `v = v*gain + offset` after `calculate_setting`. The existing whole-swarm path through `modulate()` and `slider_value` is not touched, so this is deletable.

**Tech Stack:** Python 3.12, NumPy, ModernGL (OpenGL 4.3 compute), imgui_bundle, pytest.

**Spec:** `docs/superpowers/specs/2026-08-18-cohort-audio-routing-design.md`. Read it before Task 1.

## Global Constraints

- **This feature is EXPERIMENTAL and must stay deletable.** Section 8 of the spec is the removal procedure. Every task must leave it true. Do not refactor anything outside the files each task names.
- **Run tests with the venv interpreter:** `.venv/Scripts/python.exe -m pytest`. Bare `python` is 3.10 and has no pytest.
- **Commit each task locally. Never push.** Stage files BY NAME — `git add <path>`, never `git add -A`. Other sessions share this working tree and it already has unrelated modified files.
- **Comments state the rule, never the evidence.** No measured percentages, timings, dates or before/after comparisons in comments, docstrings or tooltips. A tooltip is ONE sentence naming what the control does. Measured facts belong in the spec.
- **Naming:** `SimState` fields and GLSL uniforms are `ALL_CAPS_UNDERSCORE`; UI labels are Title Case; private UI state is `_snake_case`.
- **Windows:** forward slashes or `os.path`; `rm` not `del` in bash.
- `MASK_SLOTS = 144`, matching `MAX_COHORTS` in `services/cohort_tiling.py`.
- `COHORT_AUDIO_PARAMS` is exactly: `SENSOR_GAIN`, `SENSOR_ANGLE`, `SENSOR_DISTANCE`, `MUTATION_SCALE`, `GLOBAL_FORCE_MULT`, `DRAG`, `AXIAL_FORCE`, `LATERAL_FORCE`, `STRAFE_POWER`, `HAZARD_RATE` — in that order. `TRAIL_PERSISTENCE`, `TRAIL_DIFFUSION` and `TIME_SCALE` are deliberately absent.
- The SSBO is **binding 5**. Bindings 0, 2, 3 and 4 are taken.

---

## File Structure

| File | Responsibility |
|---|---|
| `services/cohort_audio.py` | NEW. The mask model, the row order, and the affine reduction. All feature logic. |
| `shaders/cohort_audio.glsl` | NEW. The SSBO, the `CA_*` row constants, and `cohort_audio()`. |
| `ui/cohort_strip.py` | NEW. The paint widget and its drag state. |
| `tests/test_cohort_audio.py` | NEW. Mask maths, the reduction, the shader/Python row order. |
| `tests/test_cohort_strip.py` | NEW. Drives the mouse across the strip. |
| `tests/test_cohort_audio_gl.py` | NEW. Runs the sim with half the cohorts masked. |
| `services/audio_mapping.py` | +1 field on `Mapping`. |
| `state/audio_in_state.py` | +2 lines, mask in and out of the rig dict. |
| `services/audio_runtime.py` | +1 call, +1 attribute. |
| `sim.py` | +1 buffer, +1 setter, +1 write. USER-OWNED — additive only. |
| `shaders/entity_update.glsl` | The prepend, and `cohort_audio()` around ten call sites. |
| `ui/audio_reactive_window.py` | +1 call into the strip, +1 chip marker. |
| `main.py` | +1 line. |

---

### Task 1: The mask model

Pure Python and NumPy. No GL, no imgui, no imports from `ui` or `sim`.

**Files:**
- Create: `services/cohort_audio.py`
- Test: `tests/test_cohort_audio.py`

**Interfaces:**
- Consumes: `MAX_COHORTS` from `services/cohort_tiling.py`.
- Produces:
  - `MASK_SLOTS: int` (144)
  - `COHORT_AUDIO_PARAMS: tuple[str, ...]` (10 names, order fixed)
  - `full_mask() -> np.ndarray` — `(144,)` bool, all True
  - `slot_of(cohort: int, n: int) -> int`
  - `paint_span(cell: int, n: int) -> tuple[int, int]` — half-open `[lo, hi)`
  - `covers(mask: np.ndarray, cohort: int, n: int) -> bool`
  - `paint(mask: np.ndarray, cell: int, n: int, value: bool) -> None` — in place
  - `is_full(mask) -> bool`, `is_empty(mask) -> bool`
  - `cells_lit(mask: np.ndarray, n: int) -> np.ndarray` — `(n,)` bool, what the strip draws

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cohort_audio.py`:

```python
"""The cohort mask: a selection over the normalised cohort axis."""
import numpy as np
import pytest

from services import cohort_audio as ca
from services.cohort_tiling import MAX_COHORTS


def test_the_mask_is_as_wide_as_the_cohort_cap():
    assert ca.MASK_SLOTS == MAX_COHORTS


def test_a_fresh_mask_covers_every_cohort():
    m = ca.full_mask()
    assert m.shape == (ca.MASK_SLOTS,)
    assert m.dtype == np.bool_
    assert ca.is_full(m)
    assert not ca.is_empty(m)
    assert all(ca.covers(m, c, 64) for c in range(64))


def test_painting_a_cell_at_a_coarse_count_fills_its_whole_span():
    m = np.zeros(ca.MASK_SLOTS, dtype=bool)
    ca.paint(m, 0, 8, True)
    lo, hi = ca.paint_span(0, 8)
    assert (lo, hi) == (0, 18)
    assert m[lo:hi].all()
    assert not m[hi:].any()


def test_every_cell_paints_at_least_one_slot():
    for n in (1, 7, 63, 64, 100, 143, 144):
        for cell in range(n):
            lo, hi = ca.paint_span(cell, n)
            assert hi > lo, (n, cell)
            assert 0 <= lo < ca.MASK_SLOTS
            assert hi <= ca.MASK_SLOTS


def test_a_mask_holds_its_proportions_when_the_cohort_count_changes():
    """The whole point: the first half stays the first half."""
    m = np.zeros(ca.MASK_SLOTS, dtype=bool)
    for cell in range(32):
        ca.paint(m, cell, 64, True)

    for n in (8, 16, 64, 100, 144):
        lit = ca.cells_lit(m, n)
        assert lit[: n // 2].all(), n
        assert not lit[n // 2 :].any(), n


def test_cells_lit_is_one_entry_per_live_cohort():
    m = ca.full_mask()
    assert ca.cells_lit(m, 7).shape == (7,)
    assert ca.cells_lit(m, 144).shape == (144,)


def test_an_empty_mask_covers_nothing():
    m = np.zeros(ca.MASK_SLOTS, dtype=bool)
    assert ca.is_empty(m)
    assert not any(ca.covers(m, c, 64) for c in range(64))


def test_slot_of_never_leaves_the_mask():
    for n in (1, 3, 64, 144):
        for c in range(n):
            assert 0 <= ca.slot_of(c, n) < ca.MASK_SLOTS


def test_the_param_rows_are_the_ones_the_shader_can_reach():
    assert ca.COHORT_AUDIO_PARAMS == (
        "SENSOR_GAIN", "SENSOR_ANGLE", "SENSOR_DISTANCE", "MUTATION_SCALE",
        "GLOBAL_FORCE_MULT", "DRAG", "AXIAL_FORCE", "LATERAL_FORCE",
        "STRAFE_POWER", "HAZARD_RATE",
    )


def test_the_trail_and_the_clock_are_not_maskable():
    """canvas.frag has no cohort, and TIME_SCALE is one float for the canvas."""
    for name in ("TRAIL_PERSISTENCE", "TRAIL_DIFFUSION", "TIME_SCALE", "V_MAX"):
        assert name not in ca.COHORT_AUDIO_PARAMS


def test_every_maskable_param_is_a_real_modulation_target():
    from services.audio_mapping import physics_targets
    from state.sim_state import SimState

    keys = {t.key for t in physics_targets(SimState())}
    assert set(ca.COHORT_AUDIO_PARAMS) <= keys
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/test_cohort_audio.py -q
```

Expected: collection error, `ModuleNotFoundError: No module named 'services.cohort_audio'`.

- [ ] **Step 3: Write the module**

Create `services/cohort_audio.py`:

```python
"""Which cohorts an audio mapping drives.

A mask over the NORMALISED cohort axis, not over cohort indices: a live cohort
reads the slot its position falls in, so a mask keeps its proportions when the
cohort count moves. Changing the count writes nothing, which is what makes it
lossless; painting at a coarse count writes wide spans, which is not.
"""
from __future__ import annotations

import numpy as np

from services.cohort_tiling import MAX_COHORTS

MASK_SLOTS = MAX_COHORTS

# The parameters a mask can reach, in the row order the SSBO uses. Mirrored as
# CA_* in shaders/cohort_audio.glsl; the two are compared by a test. A
# parameter belongs here only if entity_update.glsl evaluates it per particle.
COHORT_AUDIO_PARAMS: tuple[str, ...] = (
    "SENSOR_GAIN", "SENSOR_ANGLE", "SENSOR_DISTANCE", "MUTATION_SCALE",
    "GLOBAL_FORCE_MULT", "DRAG", "AXIAL_FORCE", "LATERAL_FORCE",
    "STRAFE_POWER", "HAZARD_RATE",
)


def full_mask() -> np.ndarray:
    return np.ones(MASK_SLOTS, dtype=bool)


def slot_of(cohort: int, n: int) -> int:
    if n <= 0:
        return 0
    return min(MASK_SLOTS - 1, int((cohort + 0.5) / n * MASK_SLOTS))


def paint_span(cell: int, n: int) -> tuple[int, int]:
    """Half-open slot range one strip cell owns. Never empty."""
    if n <= 0:
        return 0, MASK_SLOTS
    lo = min(MASK_SLOTS - 1, int(cell / n * MASK_SLOTS))
    hi = min(MASK_SLOTS, max(lo + 1, int((cell + 1) / n * MASK_SLOTS)))
    return lo, hi


def paint(mask: np.ndarray, cell: int, n: int, value: bool) -> None:
    lo, hi = paint_span(cell, n)
    mask[lo:hi] = value


def covers(mask: np.ndarray, cohort: int, n: int) -> bool:
    return bool(mask[slot_of(cohort, n)])


def cells_lit(mask: np.ndarray, n: int) -> np.ndarray:
    """One entry per live cohort - what the strip draws."""
    if n <= 0:
        return np.zeros(0, dtype=bool)
    idx = np.minimum(MASK_SLOTS - 1,
                     ((np.arange(n) + 0.5) / n * MASK_SLOTS).astype(np.int64))
    return mask[idx]


def is_full(mask: np.ndarray) -> bool:
    return bool(np.all(mask))


def is_empty(mask: np.ndarray) -> bool:
    return not bool(np.any(mask))
```

- [ ] **Step 4: Run to verify it passes**

```bash
.venv/Scripts/python.exe -m pytest tests/test_cohort_audio.py -q
```

Expected: PASS, 11 tests.

- [ ] **Step 5: Commit**

```bash
git add services/cohort_audio.py tests/test_cohort_audio.py
git commit -m "feat: a cohort mask that rescales with the cohort count"
```

---

### Task 2: The mask on Mapping, and in the rig file

**Files:**
- Modify: `services/audio_mapping.py` (the `Mapping` dataclass, around line 34)
- Modify: `state/audio_in_state.py` (`_mapping_to_dict` around line 104, `_mapping_from_dict` around line 127)
- Test: `tests/test_cohort_audio.py` (append)

**Interfaces:**
- Consumes: `cohort_audio.full_mask`, `is_full`, `MASK_SLOTS` from Task 1.
- Produces: `Mapping.cohorts: np.ndarray` — always a `(144,)` bool array, never `None`. Rig key is `"cohorts"`, a list of 0/1 ints, written only when the mask is not full.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cohort_audio.py`:

```python
def test_a_new_mapping_drives_every_cohort():
    from services.audio_mapping import Mapping
    m = Mapping(signal="bass", target="SENSOR_GAIN")
    assert m.cohorts.shape == (ca.MASK_SLOTS,)
    assert ca.is_full(m.cohorts)


def test_two_mappings_do_not_share_one_mask():
    from services.audio_mapping import Mapping
    a = Mapping(signal="bass", target="SENSOR_GAIN")
    b = Mapping(signal="hi", target="SENSOR_GAIN")
    a.cohorts[:10] = False
    assert ca.is_full(b.cohorts)


def test_an_unmasked_mapping_writes_nothing_to_the_rig():
    from services.audio_in_state import _mapping_to_dict
    from services.audio_mapping import Mapping
    d = _mapping_to_dict(Mapping(signal="bass", target="SENSOR_GAIN"))
    assert "cohorts" not in d


def test_a_mask_round_trips_through_the_rig():
    from services.audio_in_state import _mapping_from_dict, _mapping_to_dict
    from services.audio_mapping import Mapping
    m = Mapping(signal="bass", target="SENSOR_GAIN")
    for cell in range(32):
        ca.paint(m.cohorts, cell, 64, False)

    back = _mapping_from_dict(_mapping_to_dict(m))
    assert back is not None
    assert np.array_equal(back.cohorts, m.cohorts)


def test_a_rig_written_before_this_feature_loads_as_all_cohorts():
    from services.audio_in_state import _mapping_from_dict
    m = _mapping_from_dict({"signal": "bass", "target": "SENSOR_GAIN",
                            "mode": "add", "depth": 0.5})
    assert m is not None
    assert ca.is_full(m.cohorts)


def test_a_malformed_mask_falls_back_to_all_cohorts():
    """One bad row must not lose the rig."""
    from services.audio_in_state import _mapping_from_dict
    for bad in ("nonsense", [1, 2, 3], [None] * ca.MASK_SLOTS, {}):
        m = _mapping_from_dict({"signal": "bass", "target": "SENSOR_GAIN",
                                "cohorts": bad})
        assert m is not None, bad
        assert ca.is_full(m.cohorts), bad


def test_a_mask_edit_is_a_rig_change():
    """_save_last_rig diffs by value, so an edit has to be visible there."""
    from services.audio_in_state import AudioInState, to_dict
    from services.audio_mapping import Mapping
    s = AudioInState()
    s.mappings = [Mapping(signal="bass", target="SENSOR_GAIN")]
    before = to_dict(s)
    ca.paint(s.mappings[0].cohorts, 0, 64, False)
    assert to_dict(s) != before
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/test_cohort_audio.py -q -k "cohort or mask or rig"
```

Expected: FAIL, `AttributeError: 'Mapping' object has no attribute 'cohorts'`.

- [ ] **Step 3: Add the field**

In `services/audio_mapping.py`, add the import at the top of the imports block:

```python
from services.cohort_audio import full_mask
```

and add this field to `Mapping`, after `uid`:

```python
    # Which cohorts this mapping drives, over the normalised cohort axis. Full
    # means every cohort, which is what a rig written before this comes to.
    # An ordinary field, so a rig diffed by value sees a mask edit.
    cohorts: "np.ndarray" = field(default_factory=full_mask)
```

Add `import numpy as np` to the top of `services/audio_mapping.py`.

`Mapping` is a plain `@dataclass` (not frozen, not `eq=False`), and NumPy arrays
make the generated `__eq__` return an array rather than a bool. Give the class
an explicit `__eq__` so `to_dict` comparisons and `in`/`remove` still work:

```python
    def __eq__(self, other) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        return (self.signal, self.target, self.mode, self.depth, self.gain,
                self.shaper, self.enabled, self.uid) == \
               (other.signal, other.target, other.mode, other.depth,
                other.gain, other.shaper, other.enabled, other.uid) and \
            bool(np.array_equal(self.cohorts, other.cohorts))
```

- [ ] **Step 4: Persist it**

In `state/audio_in_state.py`, in `_mapping_to_dict`, before the `return`:

```python
def _mapping_to_dict(m: Mapping) -> dict:
    """`uid` is in-session only, so a loaded rig mints fresh ones."""
    from services.cohort_audio import is_full

    out = {
        "signal": m.signal, "target": m.target, "mode": m.mode,
        "depth": float(m.depth), "gain": float(m.gain),
        "enabled": bool(m.enabled),
        "shaper": {
            "kind": m.shaper.kind, "attack": m.shaper.attack,
            "release": m.shaper.release, "threshold": m.shaper.threshold,
            "hold": m.shaper.hold, "rate": m.shaper.rate,
            "wave": m.shaper.wave,
        },
    }
    # Only when something is painted out, so an unmasked row adds nothing and
    # a rig stays readable.
    if not is_full(m.cohorts):
        out["cohorts"] = [int(v) for v in m.cohorts]
    return out
```

In `_mapping_from_dict`, replace the `return Mapping(...)` block inside the
`try` with:

```python
    try:
        m = Mapping(
            signal=signal, target=target, mode=mode,
            depth=float(d.get("depth", 0.5)), gain=float(d.get("gain", 1.0)),
            shaper=shaper, enabled=bool(d.get("enabled", True)),
        )
    except (TypeError, ValueError):
        return None
    _read_mask(m, d.get("cohorts"))
    return m
```

and add above `_mapping_from_dict`:

```python
def _read_mask(m, raw) -> None:
    """A malformed mask leaves the mapping on every cohort, never drops it."""
    from services.cohort_audio import MASK_SLOTS

    if not isinstance(raw, list) or len(raw) != MASK_SLOTS:
        return
    if not all(isinstance(v, (int, float, bool)) for v in raw):
        return
    m.cohorts[:] = [bool(v) for v in raw]
```

- [ ] **Step 5: Run the audio suite**

```bash
.venv/Scripts/python.exe -m pytest tests/test_cohort_audio.py tests/test_audio_mapping.py tests/test_rig_presets.py tests/test_audio_runtime.py -q
```

Expected: PASS. If `test_audio_mapping.py` fails on mapping equality, the
explicit `__eq__` in Step 3 is missing or wrong — fix that, do not change the
existing test.

- [ ] **Step 6: Commit**

```bash
git add services/audio_mapping.py state/audio_in_state.py tests/test_cohort_audio.py
git commit -m "feat: every mapping carries a cohort mask, persisted when painted"
```

---

### Task 3: The affine reduction

The heart of it. For each maskable parameter and each live cohort, reduce every
mapping that covers that cohort into one gain and one offset.

**Files:**
- Modify: `services/cohort_audio.py`
- Test: `tests/test_cohort_audio.py` (append)

**Interfaces:**
- Consumes: `ShaperState` from `services/audio_shapers.py`; `TargetDef` from `services/audio_mapping.py`.
- Produces:
  ```python
  def build_arrays(mappings, targets, signals, states, strengths,
                   global_strength, dt, deaf, n_cohorts,
                   held=(), rate_scale=1.0) -> tuple[np.ndarray, bool]
  ```
  Returns `(len(COHORT_AUDIO_PARAMS), MASK_SLOTS, 2)` float32, indexed
  `[row, COHORT]` — not by mask slot, because that is what the shader reads.
  `[..., 0]` is gain, `[..., 1]` is offset. The bool says whether any mapping is
  masked. Entries at or above `n_cohorts` are identity, as is the whole array
  when the flag is False, and the caller may then skip the upload.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cohort_audio.py`:

```python
def _targets():
    from services.audio_mapping import physics_targets
    from state.sim_state import SimState
    return physics_targets(SimState())


def _apply(arr, row, cohort, base):
    """The array is COHORT-indexed, matching what the shader reads."""
    g, o = arr[row, cohort]
    return float(g) * base + float(o)


def test_nothing_masked_is_reported_as_nothing_masked():
    from services.audio_mapping import Mapping
    m = Mapping(signal="bass", target="SENSOR_GAIN")
    arr, active = ca.build_arrays(
        [m], _targets(), {"bass": 1.0}, {}, {}, 1.0, 1 / 60.0, set(), 64)
    assert active is False


def test_a_masked_row_is_reported_as_masked():
    from services.audio_mapping import Mapping
    m = Mapping(signal="bass", target="SENSOR_GAIN")
    ca.paint(m.cohorts, 0, 64, False)
    _, active = ca.build_arrays(
        [m], _targets(), {"bass": 1.0}, {}, {}, 1.0, 1 / 60.0, set(), 64)
    assert active is True


def test_a_masked_row_moves_only_the_cohorts_it_names():
    from services.audio_mapping import Mapping
    m = Mapping(signal="bass", target="SENSOR_GAIN", mode="add", depth=0.5)
    m.cohorts[:] = False
    for cell in range(32):
        ca.paint(m.cohorts, cell, 64, True)

    arr, _ = ca.build_arrays([m], _targets(), {"bass": 1.0}, {}, {}, 1.0,
                             1 / 60.0, set(), 64)
    row = ca.COHORT_AUDIO_PARAMS.index("SENSOR_GAIN")
    assert _apply(arr, row, 0, 2.0) > 2.0
    assert _apply(arr, row, 63, 2.0) == pytest.approx(2.0)


def test_the_two_band_split_the_feature_exists_for():
    """bass ADDS over the first half, hi SUBTRACTS over the second."""
    from services.audio_mapping import Mapping
    lo = Mapping(signal="bass", target="SENSOR_GAIN", mode="add", depth=0.5)
    hi = Mapping(signal="hi", target="SENSOR_GAIN", mode="subtract", depth=0.5)
    lo.cohorts[:] = False
    hi.cohorts[:] = False
    for cell in range(32):
        ca.paint(lo.cohorts, cell, 64, True)
    for cell in range(32, 64):
        ca.paint(hi.cohorts, cell, 64, True)

    arr, _ = ca.build_arrays([lo, hi], _targets(), {"bass": 1.0, "hi": 1.0},
                             {}, {}, 1.0, 1 / 60.0, set(), 64)
    row = ca.COHORT_AUDIO_PARAMS.index("SENSOR_GAIN")
    assert _apply(arr, row, 0, 2.0) > 2.0
    assert _apply(arr, row, 63, 2.0) < 2.0


def test_a_disabled_row_contributes_nothing():
    from services.audio_mapping import Mapping
    row = ca.COHORT_AUDIO_PARAMS.index("SENSOR_GAIN")
    m = Mapping(signal="bass", target="SENSOR_GAIN", depth=0.5, enabled=False)
    ca.paint(m.cohorts, 0, 64, False)
    arr, _ = ca.build_arrays([m], _targets(), {"bass": 1.0}, {}, {}, 1.0,
                             1 / 60.0, set(), 64)
    assert _apply(arr, row, 40, 2.0) == pytest.approx(2.0)


def test_a_deaf_target_contributes_nothing():
    """A swept or muted parameter is refused here as it is in modulate()."""
    from services.audio_mapping import Mapping
    row = ca.COHORT_AUDIO_PARAMS.index("SENSOR_GAIN")
    m = Mapping(signal="bass", target="SENSOR_GAIN", depth=0.5)
    ca.paint(m.cohorts, 0, 64, False)
    arr, _ = ca.build_arrays([m], _targets(), {"bass": 1.0}, {}, {}, 1.0,
                             1 / 60.0, {"SENSOR_GAIN"}, 64)
    assert _apply(arr, row, 40, 2.0) == pytest.approx(2.0)


def test_an_empty_mask_is_idle_rather_than_everything():
    from services.audio_mapping import Mapping
    m = Mapping(signal="bass", target="SENSOR_GAIN", depth=0.5)
    m.cohorts[:] = False
    arr, active = ca.build_arrays([m], _targets(), {"bass": 1.0}, {}, {}, 1.0,
                                  1 / 60.0, set(), 64)
    assert active is True
    row = ca.COHORT_AUDIO_PARAMS.index("SENSOR_GAIN")
    for c in (0, 31, 63):
        assert _apply(arr, row, c, 2.0) == pytest.approx(2.0)


@pytest.mark.parametrize("mode", ["add", "subtract", "multiply"])
@pytest.mark.parametrize("strength", [0.0, 0.4, 1.0])
def test_the_reduction_agrees_with_modulate_when_the_mask_is_full(mode, strength):
    """The property that keeps the two paths from drifting apart."""
    from services.audio_mapping import Mapping, modulate
    from state.sim_state import SimState

    targets = _targets()
    base = float(getattr(SimState(), "SENSOR_GAIN"))
    signals = {"bass": 0.7}
    m = Mapping(signal="bass", target="SENSOR_GAIN", mode=mode, depth=0.6,
                gain=1.3)

    want = modulate({"SENSOR_GAIN": base}, targets, [m], signals, {},
                    {"SENSOR_GAIN": strength}, 1.0, 1 / 60.0, set(),
                    apply_shapers=False)["SENSOR_GAIN"]

    arr, _ = ca.build_arrays([m], targets, signals, {},
                             {"SENSOR_GAIN": strength}, 1.0, 1 / 60.0, set(),
                             64, apply_shapers=False)
    row = ca.COHORT_AUDIO_PARAMS.index("SENSOR_GAIN")
    assert _apply(arr, row, 5, base) == pytest.approx(want, rel=1e-5)


def test_cohorts_past_the_live_count_are_identity():
    from services.audio_mapping import Mapping
    m = Mapping(signal="bass", target="SENSOR_GAIN", depth=1.0)
    ca.paint(m.cohorts, 0, 8, False)
    arr, _ = ca.build_arrays([m], _targets(), {"bass": 1.0}, {}, {}, 1.0,
                             1 / 60.0, set(), 8)
    row = ca.COHORT_AUDIO_PARAMS.index("SENSOR_GAIN")
    tail = arr[row, 8:]
    assert np.allclose(tail[:, 0], 1.0)
    assert np.allclose(tail[:, 1], 0.0)
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/test_cohort_audio.py -q -k "reduction or masked or split or idle or identity"
```

Expected: FAIL, `AttributeError: module 'services.cohort_audio' has no attribute 'build_arrays'`.

- [ ] **Step 3: Implement the reduction**

Append to `services/cohort_audio.py`:

```python
def build_arrays(mappings, targets, signals, states, strengths,
                 global_strength: float, dt: float, deaf, n_cohorts: int,
                 held=(), rate_scale: float = 1.0,
                 apply_shapers: bool = True) -> tuple[np.ndarray, bool]:
    """Per-cohort gain and offset for every maskable parameter.

    The modulation chain is affine in the base value, so a cohort's whole
    contribution is one multiply and one add:

        gain   = 1 + S * (M - 1)
        offset = S * A * M

    with A the summed add/subtract terms, M the product of the multiply terms
    and S the combined strength. Nothing there reads the base, which is what
    lets this run per cohort on the CPU and land on top of a sweep.

    The second return says whether any mapping is masked at all; when it is
    False the array is identity and the caller may skip the upload.
    """
    from services.audio_shapers import ShaperState

    rows = len(COHORT_AUDIO_PARAMS)
    arr = np.empty((rows, MASK_SLOTS, 2), dtype=np.float32)
    arr[..., 0] = 1.0
    arr[..., 1] = 0.0

    by_target = {t.key: t for t in targets}
    live = []
    any_masked = False
    for m in mappings:
        if m.target not in COHORT_AUDIO_PARAMS:
            continue
        if not m.enabled or m.target in deaf or m.target not in by_target:
            continue
        if m.signal not in signals:
            continue
        if not is_full(m.cohorts):
            any_masked = True
        live.append(m)

    if not any_masked:
        return arr, False

    n = min(MASK_SLOTS, max(1, int(n_cohorts)))
    # The slot each live cohort reads. The OUTPUT is indexed by cohort, not by
    # slot: the shader has a cohort and no way to recover a slot from it.
    slots = np.minimum(MASK_SLOTS - 1,
                       ((np.arange(n) + 0.5) / n * MASK_SLOTS).astype(np.int64))

    shaped: dict[int, float] = {}
    for m in live:
        s = min(1.0, max(0.0, signals[m.signal] * m.gain))
        if apply_shapers:
            s = states.setdefault(m.uid, ShaperState()).apply(
                s, dt, m.shaper, m.signal not in held, rate_scale)
        shaped[m.uid] = s

    for row, key in enumerate(COHORT_AUDIO_PARAMS):
        bound = [m for m in live if m.target == key]
        if not bound:
            continue
        t = by_target[key]
        span = t.hi - t.lo
        strength = float(strengths.get(key, 1.0)) * float(global_strength)

        a = np.zeros(n, dtype=np.float64)
        mul = np.ones(n, dtype=np.float64)
        for m in bound:
            covered = m.cohorts[slots]
            s = shaped[m.uid]
            if m.mode == "multiply":
                mul[covered] *= 1.0 + s * m.depth
            else:
                sign = -1.0 if m.mode == "subtract" else 1.0
                a[covered] += sign * s * m.depth * span

        arr[row, :n, 0] = (1.0 + strength * (mul - 1.0)).astype(np.float32)
        arr[row, :n, 1] = (strength * a * mul).astype(np.float32)

    return arr, True
```

- [ ] **Step 4: Run to verify it passes**

```bash
.venv/Scripts/python.exe -m pytest tests/test_cohort_audio.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/cohort_audio.py tests/test_cohort_audio.py
git commit -m "feat: reduce masked mappings to a per-cohort gain and offset"
```

---

### Task 4: The shader

**Files:**
- Create: `shaders/cohort_audio.glsl`
- Modify: `shaders/entity_update.glsl` (ten `calculate_setting` call sites, lines 624–754)
- Modify: `sim.py` (the entity-update shader build, near line 190)
- Test: `tests/test_cohort_audio.py` (append)

**Interfaces:**
- Consumes: `COHORT_AUDIO_PARAMS` from Task 1.
- Produces: GLSL `float cohort_audio(float v, int row, float cohort)`, `#define CA_<NAME> <index>` per row, SSBO `CohortAudioBuffer` at binding 5, `uniform bool COHORT_AUDIO_ACTIVE`, `uniform int COHORT_AUDIO_ROWS`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cohort_audio.py`:

```python
from pathlib import Path

SHADERS = Path(__file__).resolve().parent.parent / "shaders"


def _ca_defines():
    import re
    src = (SHADERS / "cohort_audio.glsl").read_text(encoding="utf-8")
    return {m.group(1): int(m.group(2)) for m in
            re.finditer(r"#define\s+CA_(\w+)\s+(\d+)", src)}


def test_the_shader_and_python_agree_on_the_row_order():
    """Neither side can check this at runtime; a stale row reads the wrong
    parameter's modulation with nothing raising."""
    defines = _ca_defines()
    assert defines == {name: i for i, name
                       in enumerate(ca.COHORT_AUDIO_PARAMS)}


def test_the_shader_binds_a_free_slot():
    src = (SHADERS / "cohort_audio.glsl").read_text(encoding="utf-8")
    assert "binding = 5" in src
    for taken in ("binding = 0", "binding = 2", "binding = 3", "binding = 4"):
        assert taken not in src


def test_every_maskable_parameter_is_wrapped_at_its_call_site():
    """A parameter with a row but no wrapper is a control that does nothing."""
    src = (SHADERS / "entity_update.glsl").read_text(encoding="utf-8")
    for name in ca.COHORT_AUDIO_PARAMS:
        assert f"CA_{name}" in src, name


def test_the_unmaskable_parameters_are_not_wrapped():
    src = (SHADERS / "entity_update.glsl").read_text(encoding="utf-8")
    for name in ("TRAIL_PERSISTENCE", "TRAIL_DIFFUSION", "TIME_SCALE"):
        assert f"CA_{name}" not in src


def test_calculate_setting_itself_is_untouched():
    """It must stay character-identical across two shaders and sim.py."""
    src = (SHADERS / "entity_update.glsl").read_text(encoding="utf-8")
    body = src.split("float calculate_setting(")[1].split("\n}")[0]
    assert "cohort_audio" not in body
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/test_cohort_audio.py -q -k "shader or wrapped or calculate_setting"
```

Expected: FAIL, `FileNotFoundError` on `shaders/cohort_audio.glsl`.

- [ ] **Step 3: Write the shader include**

Create `shaders/cohort_audio.glsl`:

```glsl
// Per-cohort audio modulation. Prepended to entity_update.glsl.
//
// The CPU reduces every masked mapping into one gain and one offset per
// cohort, because the whole modulation chain is affine in the base value.
// Applied AFTER calculate_setting, never inside it: that function must stay
// character-identical across entity_update.glsl, canvas.frag and sim.py, and
// canvas.frag has no cohort to index with.
//
// Row order is mirrored in services/cohort_audio.COHORT_AUDIO_PARAMS and
// compared by a test. A row here that names a different parameter than the
// host thinks reads the wrong modulation with nothing raising.
#define CA_SENSOR_GAIN 0
#define CA_SENSOR_ANGLE 1
#define CA_SENSOR_DISTANCE 2
#define CA_MUTATION_SCALE 3
#define CA_GLOBAL_FORCE_MULT 4
#define CA_DRAG 5
#define CA_AXIAL_FORCE 6
#define CA_LATERAL_FORCE 7
#define CA_STRAFE_POWER 8
#define CA_HAZARD_RATE 9

#define CA_SLOTS 144

layout(std430, binding = 5) buffer CohortAudioBuffer {
    vec2 cohort_audio_ga[];   // x = gain, y = offset
};

uniform bool COHORT_AUDIO_ACTIVE;

float cohort_audio(float v, int row, float cohort) {
    if (!COHORT_AUDIO_ACTIVE) return v;
    int c = clamp(int(floor(cohort)), 0, CA_SLOTS - 1);
    vec2 ga = cohort_audio_ga[row * CA_SLOTS + c];
    return v * ga.x + ga.y;
}
```

The shader indexes by COHORT, which is what `build_arrays` already writes —
Task 3's array is `[row, cohort]`, not `[row, slot]`. Nothing to reconcile here;
if you find yourself wanting to, Task 3 was implemented wrongly.

- [ ] **Step 4: Prepend it**

In `sim.py`, in the block that assembles `self.entity_update_source` (lines
206–220), add this immediately after the `_header.glsl` prepend and before the
`fourier4_4.glsl` one:

```python
        self.entity_update_source = shader_prepend(
            self.entity_update_source, read_shader('shaders/cohort_audio.glsl'))
```

Each `shader_prepend` puts its content at the front, so this lands above
`entity_update.glsl`'s body, which is all `cohort_audio()` needs.

- [ ] **Step 5: Wrap the ten call sites**

In `shaders/entity_update.glsl`, wrap each `calculate_setting(...)` whose
parameter has a row. Every one, with its line number for orientation:

```glsl
// 624-625
force=forward*force.x*cohort_audio(calculate_setting(get_particle_axial_force(),pos,cohort),CA_AXIAL_FORCE,cohort)+left*force.y*cohort_audio(calculate_setting(get_particle_lateral_force(),pos,cohort),CA_LATERAL_FORCE,cohort);
strafe = forward*strafe.x*cohort_audio(calculate_setting(get_particle_axial_force(),pos,cohort),CA_AXIAL_FORCE,cohort) + left * strafe.y * cohort_audio(calculate_setting(get_particle_lateral_force(),pos,cohort),CA_LATERAL_FORCE,cohort);
// 647
g_brain_mut = cohort_audio(calculate_setting(get_particle_mutation_scale(),e.pos,cohort),CA_MUTATION_SCALE,cohort);
// 661
if (frame_count==0||cohort_audio(calculate_setting(get_particle_hazard_rate(),e.pos,cohort),CA_HAZARD_RATE,cohort)>hash(vec2(float(index)/float(ACTIVE_COUNT),frame_count))){reset(index);return;}
// 666
float sample_dist = 1./SQRT_WORLD_SIZE*.005 * cohort_audio(calculate_setting(get_particle_sensor_distance(),e.pos,cohort),CA_SENSOR_DISTANCE,cohort);
// 680-681
pR(left_sensor_offset,cohort_audio(calculate_setting(get_particle_sensor_angle(),e.pos,cohort),CA_SENSOR_ANGLE,cohort)*PI);
pR(right_sensor_offset,-cohort_audio(calculate_setting(get_particle_sensor_angle(),e.pos,cohort),CA_SENSOR_ANGLE,cohort)*PI);
// 697
float sensor_scaling = SQRT_WORLD_SIZE*38.855*cohort_audio(calculate_setting(get_particle_sensor_gain(),e.pos,cohort),CA_SENSOR_GAIN,cohort);
// 708-709
force *= 1./SQRT_WORLD_SIZE*cohort_audio(calculate_setting(get_particle_global_force_mult(),e.pos,cohort),CA_GLOBAL_FORCE_MULT,cohort)/400.;
strafe *= 1./SQRT_WORLD_SIZE*cohort_audio(calculate_setting(get_particle_global_force_mult(),e.pos,cohort),CA_GLOBAL_FORCE_MULT,cohort)/20.;
// 731
float drag = cohort_audio(calculate_setting(get_particle_drag(),e.pos,cohort),CA_DRAG,cohort);
// 740
vec2 hop = strafe*cohort_audio(calculate_setting(get_particle_strafe_power(),e.pos+e.vel,cohort),CA_STRAFE_POWER,cohort);
```

Line 754's `calculate_setting(vm,...)` is `V_MAX` and is deliberately NOT
wrapped — it has no row.

- [ ] **Step 6: Check it compiles**

```bash
.venv/Scripts/python.exe -m tools.shader_compile_check
```

Expected: no errors. If `cohort_audio` is reported undeclared, the prepend in
Step 4 is in the wrong order.

- [ ] **Step 7: Run the tests**

```bash
.venv/Scripts/python.exe -m pytest tests/test_cohort_audio.py tests/test_brain_shader_source.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add shaders/cohort_audio.glsl shaders/entity_update.glsl sim.py tests/test_cohort_audio.py
git commit -m "feat: the shader applies a per-cohort gain and offset"
```

---

### Task 5: The upload

`sim.py` is user-owned. Additive only: one buffer, one setter, one write. Do not
reorganise anything around it.

**Files:**
- Modify: `sim.py` (buffer creation near line 128, `entity_update` near line 363)
- Test: covered by Task 8.

**Interfaces:**
- Consumes: `MASK_SLOTS`, `COHORT_AUDIO_PARAMS` from Task 1.
- Produces: `Simulation.set_cohort_audio(arr: np.ndarray | None) -> None`.

- [ ] **Step 1: Add the buffer**

In `sim.py`, beside the other `bind_to_storage_buffer` calls (around line 137):

```python
        # Per-cohort audio gain and offset, one row per maskable parameter.
        # Zeroed is not identity here, so it is filled rather than reserved.
        from services.cohort_audio import COHORT_AUDIO_PARAMS, MASK_SLOTS
        self._cohort_audio_rows = len(COHORT_AUDIO_PARAMS)
        self.cohort_audio_buffer = self.ctx.buffer(
            reserve=self._cohort_audio_rows * MASK_SLOTS * 2 * 4)
        self._cohort_audio = None
        self.cohort_audio_buffer.bind_to_storage_buffer(5)
```

- [ ] **Step 2: Add the setter**

Next to `apply_state` in `sim.py`:

```python
    def set_cohort_audio(self, arr) -> None:
        """Per-cohort modulation for the next step, or None for no masking."""
        self._cohort_audio = arr
```

- [ ] **Step 3: Write it per step**

In `entity_update`, in the block BELOW the frame-constant uniform cache — beside
the tournament uniforms around line 363, not inside the `if
self._entity_uniforms_on is not ...` block:

```python
        # Outside the cached block: the arrays move every frame while a rig is
        # playing, and the cache is skipped whenever the program has not
        # changed.
        _ca = self._cohort_audio
        tryset(self.entity_update_program, 'COHORT_AUDIO_ACTIVE', _ca is not None)
        if _ca is not None:
            self.cohort_audio_buffer.write(np.ascontiguousarray(_ca, dtype='f4'))
```

- [ ] **Step 4: Verify the app still starts**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q -x -k "gl or gpu"
```

Expected: PASS, or the same failures as before this task — record which by
running the same command on `git stash` first if anything fails.

- [ ] **Step 5: Commit**

```bash
git add sim.py
git commit -m "feat: upload the per-cohort audio arrays each step"
```

---

### Task 6: Wire it to the runtime

**Files:**
- Modify: `services/audio_runtime.py` (`__init__` around line 41, `update` around line 100)
- Modify: `main.py` (around line 1117, after `self.sim.apply_state(_audio_sim)`)
- Test: `tests/test_cohort_audio.py` (append)

**Interfaces:**
- Consumes: `build_arrays` from Task 3, `set_cohort_audio` from Task 5.
- Produces: `AudioRuntime.cohort_audio: np.ndarray | None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cohort_audio.py`:

```python
def test_a_stopped_rig_leaves_the_cohort_arrays_off():
    from services.audio_runtime import AudioRuntime
    from state.ui_state import UIState

    rt = AudioRuntime()
    ui = UIState()
    ui.audio.enabled = False
    rt.update(ui, 1 / 60.0, None, None)
    assert rt.cohort_audio is None
    rt.close()


def test_the_runtime_exposes_the_arrays_rather_than_widening_its_return():
    """update() is unpacked into a pair at its one call site."""
    import inspect
    from services.audio_runtime import AudioRuntime
    src = inspect.getsource(AudioRuntime.update)
    assert "self.cohort_audio" in src
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/test_cohort_audio.py -q -k "runtime or stopped"
```

Expected: FAIL, `AttributeError: 'AudioRuntime' object has no attribute 'cohort_audio'`.

- [ ] **Step 3: Add the attribute**

In `services/audio_runtime.py`, in `__init__`:

```python
        self.cohort_audio = None
```

In `update`, immediately after `ast.shaped = {}` (before any early return, for
the same reason that line is there — a bypassed rig must stop driving the sim
rather than freeze on its last value):

```python
        self.cohort_audio = None
```

and after the `moved = modulate(...)` / `if moved:` block:

```python
        arr, active = build_arrays(
            ast.mappings, p_targets, signals, self._states, ast.strengths,
            ast.global_strength, dt, deaf, ui_state.sim.num_cohorts,
            held=held, rate_scale=ast.rate_scale)
        self.cohort_audio = arr if active else None
```

Add `build_arrays` to the existing `from services.audio_mapping import ...`
line's neighbour:

```python
from services.cohort_audio import build_arrays
```

- [ ] **Step 4: Hand it to the sim**

In `main.py`, directly after `self.sim.apply_state(_audio_sim)`:

```python
        self.sim.set_cohort_audio(self.audio_runtime.cohort_audio)
```

- [ ] **Step 5: Run to verify it passes**

```bash
.venv/Scripts/python.exe -m pytest tests/test_cohort_audio.py tests/test_audio_runtime.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add services/audio_runtime.py main.py tests/test_cohort_audio.py
git commit -m "feat: hand the per-cohort arrays from the rig to the sim"
```

---

### Task 7: The paint strip

**Files:**
- Create: `ui/cohort_strip.py`
- Create: `tests/test_cohort_strip.py`
- Modify: `ui/audio_reactive_window.py` (`_render_audio_band_tab` around line 615, `_render_audio_row`'s chip loop around line 580)

**Interfaces:**
- Consumes: `cohort_audio` module from Task 1, `Mapping.cohorts` from Task 2.
- Produces: `ui.cohort_strip.CohortStripMixin` with `render_cohort_strip(self, m, n_cohorts: int) -> None`, mixed into `UI` via `AudioReactiveWindowMixin`'s class. Drag state lives in `self._cohort_paint`, a `dict` or `None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cohort_strip.py`:

```python
"""The strip is painted by dragging, so it has to be DRIVEN.

A test that asserts the cells were drawn passes just as happily when the drag
does nothing - the same trap tests/test_map_wheel.py exists for.
"""
import numpy as np
import pytest
from imgui_bundle import imgui

from services import cohort_audio as ca
from services.audio_mapping import Mapping

HOST = "cohortstriphost"
N = 64


@pytest.fixture(scope="module")
def gui():
    imgui.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1280, 900)
    io.delta_time = 1.0 / 60.0
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
    for _ in range(2):
        imgui.new_frame()
        imgui.render()
    yield
    imgui.destroy_context()


class _Rig:
    def __init__(self):
        from ui.cohort_strip import CohortStripMixin

        class _Host(CohortStripMixin):
            pass

        self.ui = _Host()
        self.m = Mapping(signal="bass", target="SENSOR_GAIN")
        self.rects = None

    def frame(self, mouse, down):
        io = imgui.get_io()
        io.add_mouse_pos_event(mouse[0], mouse[1])
        io.add_mouse_button_event(0, down)
        imgui.new_frame()
        imgui.set_next_window_size(imgui.ImVec2(700, 300))
        imgui.begin(HOST)
        self.ui.render_cohort_strip(self.m, N)
        self.rects = list(self.ui._cohort_cell_rects)
        imgui.end()
        imgui.render()


def _centre(rect):
    return ((rect[0] + rect[2]) * 0.5, (rect[1] + rect[3]) * 0.5)


def test_the_strip_draws_one_cell_per_live_cohort(gui):
    r = _Rig()
    r.frame((0, 0), False)
    assert len(r.rects) == N


def test_pressing_a_lit_cell_erases_it(gui):
    r = _Rig()
    r.frame((0, 0), False)
    target = _centre(r.rects[5])
    r.frame(target, True)
    r.frame(target, False)
    assert not ca.covers(r.m.cohorts, 5, N)


def test_dragging_across_lit_cells_erases_all_of_them(gui):
    r = _Rig()
    r.frame((0, 0), False)
    r.frame(_centre(r.rects[10]), True)
    for cell in range(11, 20):
        r.frame(_centre(r.rects[cell]), True)
    r.frame(_centre(r.rects[19]), False)
    for cell in range(10, 20):
        assert not ca.covers(r.m.cohorts, cell, N), cell
    assert ca.covers(r.m.cohorts, 25, N)


def test_a_drag_started_on_a_dark_cell_draws_rather_than_erases(gui):
    r = _Rig()
    r.m.cohorts[:] = False
    r.frame((0, 0), False)
    r.frame(_centre(r.rects[3]), True)
    for cell in range(4, 9):
        r.frame(_centre(r.rects[cell]), True)
    r.frame(_centre(r.rects[8]), False)
    for cell in range(3, 9):
        assert ca.covers(r.m.cohorts, cell, N), cell


def test_releasing_ends_the_drag(gui):
    r = _Rig()
    r.frame((0, 0), False)
    r.frame(_centre(r.rects[30]), True)
    r.frame(_centre(r.rects[30]), False)
    r.frame(_centre(r.rects[31]), False)
    assert ca.covers(r.m.cohorts, 31, N)
```

- [ ] **Step 2: Run to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/test_cohort_strip.py -q
```

Expected: FAIL, `ModuleNotFoundError: No module named 'ui.cohort_strip'`.

- [ ] **Step 3: Write the widget**

Create `ui/cohort_strip.py`:

```python
"""The cohort selection strip: one cell per live cohort, painted by dragging.

The drag direction is set by the cell it starts on - press a lit one and you
erase, press a dark one and you draw - which is what makes a single press and a
sweep the same gesture.
"""
from __future__ import annotations

from imgui_bundle import imgui

from services import cohort_audio as ca

_HEIGHT = 22.0
_GAP = 1.0


class CohortStripMixin:
    _cohort_paint = None
    _cohort_cell_rects: list = []

    def render_cohort_strip(self, m, n_cohorts: int) -> None:
        n = max(1, int(n_cohorts))
        avail = imgui.get_content_region_avail().x
        w = max(2.0, (avail - _GAP * (n - 1)) / n)
        origin = imgui.get_cursor_screen_pos()
        draw = imgui.get_window_draw_list()
        lit = ca.cells_lit(m.cohorts, n)

        self._cohort_cell_rects = []
        hovered = None
        for i in range(n):
            x = origin.x + i * (w + _GAP)
            rect = (x, origin.y, x + w, origin.y + _HEIGHT)
            self._cohort_cell_rects.append(rect)
            colour = (0xFF3A4FE0 if lit[i] else 0xFF3A3A3A)
            draw.add_rect_filled(imgui.ImVec2(rect[0], rect[1]),
                                 imgui.ImVec2(rect[2], rect[3]), colour)

        imgui.invisible_button("##cohort_strip",
                               imgui.ImVec2(max(1.0, avail), _HEIGHT))
        if imgui.is_item_hovered() or self._cohort_paint is not None:
            mx = imgui.get_io().mouse_pos.x
            idx = int((mx - origin.x) / max(1e-6, w + _GAP))
            if 0 <= idx < n:
                hovered = idx

        down = imgui.is_mouse_down(0)
        if not down:
            self._cohort_paint = None
        elif hovered is not None:
            if self._cohort_paint is None:
                # The first cell sets the direction for the whole gesture.
                self._cohort_paint = {"value": not bool(lit[hovered])}
            ca.paint(m.cohorts, hovered, n, self._cohort_paint["value"])

    def render_cohort_buttons(self, m, n_cohorts: int) -> None:
        n = max(1, int(n_cohorts))
        for label, stride in (("All", 1), ("None", 0)):
            if imgui.button(f"{label}##cohort_{label}"):
                m.cohorts[:] = bool(stride)
            imgui.same_line()
        if imgui.button("Invert##cohort_invert"):
            m.cohorts[:] = ~m.cohorts
        for stride in (2, 3, 4):
            imgui.same_line()
            if imgui.button(f"Every {stride}##cohort_every{stride}"):
                m.cohorts[:] = False
                for cell in range(0, n, stride):
                    ca.paint(m.cohorts, cell, n, True)
```

- [ ] **Step 4: Run to verify it passes**

```bash
.venv/Scripts/python.exe -m pytest tests/test_cohort_strip.py -q
```

Expected: PASS, 5 tests.

- [ ] **Step 5: Put it in the drawer**

In `ui/audio_reactive_window.py`, add to the imports:

```python
from services import cohort_audio as ca
```

and add `CohortStripMixin` to `AudioReactiveWindowMixin`'s bases (or, if it has
none, make it `class AudioReactiveWindowMixin(CohortStripMixin):`).

At the end of `_render_audio_band_tab(self, m)`, after the shaper controls:

```python
        imgui.separator()
        if m.target in ca.COHORT_AUDIO_PARAMS:
            n = int(self.state.sim.num_cohorts)
            imgui.text("Cohorts")
            hints.tip("Which cohorts this band drives; drag across to paint.")
            self.render_cohort_buttons(m, n)
            self.render_cohort_strip(m, n)
            lit = int(ca.cells_lit(m.cohorts, n).sum())
            if lit == 0:
                imgui.text_disabled("no cohorts - this band is idle")
            elif lit == n:
                imgui.text_disabled(f"all {n} cohorts")
            else:
                imgui.text_disabled(f"{lit} of {n} cohorts")
        else:
            imgui.text_disabled(
                "This parameter is one value for the whole canvas, so it "
                "cannot be split by cohort.")
```

In `_render_audio_row`, inside the `for signal in SIGNAL_NAMES:` loop, mark a
chip whose mapping is masked. Replace the `colour` lines with:

```python
            existing = next((m for m in bound if m.signal == signal), None)
            colour = imgui.ImVec4(*SIGNAL_COLORS[signal])
            if existing is None:
                colour = imgui.ImVec4(colour.x, colour.y, colour.z, 0.30)
            label = SIGNAL_ABBR[signal]
            if existing is not None and not ca.is_full(existing.cohorts):
                label = f"{label}*"
```

and change the button call to use `label`:

```python
            if imgui.button(f"{label}##{signal}"):
```

- [ ] **Step 6: Run the panel's render suite**

```bash
.venv/Scripts/python.exe -m pytest tests/test_cohort_strip.py tests/test_audio_reactive_window_render.py tests/test_label_widths.py -q
```

Expected: PASS. An ID-collision failure means a `##` suffix in Step 3 collides
with an existing widget — change the suffix, never the visible text.

- [ ] **Step 7: Commit**

```bash
git add ui/cohort_strip.py ui/audio_reactive_window.py tests/test_cohort_strip.py
git commit -m "feat: paint a mapping's cohorts by dragging across a strip"
```

---

### Task 8: The end-to-end GPU check

Every source-level test above can pass while the feature does nothing on screen.
This is the one that runs the sim.

**Files:**
- Create: `tests/test_cohort_audio_gl.py`

**Interfaces:**
- Consumes: everything from Tasks 1–6.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cohort_audio_gl.py`:

```python
"""A masked mapping moves the cohorts it names and no others.

Source-level tests cannot see this: the wrapper can be present at every call
site and still read the wrong row, or the buffer can go up transposed.

Two things are load-bearing and both come from CLAUDE.md.

Only `entity_update` is stepped, never `update()`. The sim does not reproduce
itself run to run - particles splat additively into a shared texture, which
races - so comparing two full runs is the blind instrument that caveat names.
`entity_update` alone is one invocation per particle with no shared writes, and
against a constant canvas it IS bit-reproducible. Do not "fix" this test by
switching to `update()`.

The parameter under test is GLOBAL_FORCE_MULT, which scales the force directly.
SENSOR_GAIN would be the obvious pick and it is the wrong one: it scales what is
read off the TRAIL, and with no canvas pass the trail is all zeros, so every
setting of it produces identical motion and the test passes on a broken build.
"""
import numpy as np
import pytest

from services import cohort_audio as ca

moderngl = pytest.importorskip("moderngl")
N_COHORTS = 64


@pytest.fixture(scope="module")
def ctx():
    try:
        c = moderngl.create_standalone_context(require=430)
    except Exception as exc:
        pytest.skip(f"no GL 4.3 context: {exc}")
    yield c
    c.release()


def _sim(ctx):
    from sim import Simulation
    from state.sim_state import SimState
    s = SimState()
    s.num_cohorts = N_COHORTS
    return Simulation(ctx, world_size=0.05), s


def _positions(sim):
    """SIZE_OF_ENTITY_STRUCT, never a literal - a stale stride does not fail,
    it reads other fields AS positions. See test_entity_struct.py."""
    from sim import SIZE_OF_ENTITY_STRUCT
    raw = np.frombuffer(sim.entities.read(), dtype=np.float32)
    wide = raw.reshape(-1, SIZE_OF_ENTITY_STRUCT // 4)
    return wide[: sim.entity_count, :2].copy()


def _run(ctx, arr, steps=12):
    """The particles are reset by the shader on the frame_count == 0 step, so
    the first entity_update is the reset and the rest are the measurement."""
    sim, state = _sim(ctx)
    sim.frame_count = 0
    for step in range(steps + 1):
        sim.apply_state(state)
        sim.set_cohort_audio(arr)
        sim.entity_update(ctx)
        sim.frame_count = step + 1
    ctx.finish()
    return _positions(sim)


def test_the_unmasked_half_is_left_alone(ctx):
    rows = len(ca.COHORT_AUDIO_PARAMS)
    row = ca.COHORT_AUDIO_PARAMS.index("GLOBAL_FORCE_MULT")

    off = np.empty((rows, ca.MASK_SLOTS, 2), dtype=np.float32)
    off[..., 0] = 1.0
    off[..., 1] = 0.0

    on = off.copy()
    on[row, : N_COHORTS // 2, 1] = 8.0

    base = _run(ctx, off)
    moved = _run(ctx, on)

    # Cohorts are contiguous slices of the particle index, so the first half of
    # the buffer IS the first half of the cohorts.
    travel = np.abs(moved - base).sum(axis=1)
    half = travel.shape[0] // 2
    lit = float(travel[:half].mean())
    dark = float(travel[half:].mean())

    assert lit > dark * 4.0, (lit, dark)
```

- [ ] **Step 2: Run it**

```bash
.venv/Scripts/python.exe -m pytest tests/test_cohort_audio_gl.py -q -s
```

Expected on a machine with GL 4.3: PASS. If it SKIPS, note that and move on —
the suite already skips GPU tests where no context is available.

If it fails with `lit` and `dark` nearly equal, the row index or the buffer
layout disagrees between host and shader. Check two things, in order: the array
must be indexed `[row, cohort]` and not `[row, slot]`; and `CA_GLOBAL_FORCE_MULT`
must name the same row on both sides, which the source-level test in Task 4
already covers, so a failure here with that one passing points at the upload in
Task 5 rather than at the ordering.

If `dark` is not near zero, the two runs are not reproducing — check that
nothing has started calling `sim.update()` in place of `sim.entity_update(ctx)`.

- [ ] **Step 3: Run the whole suite**

```bash
.venv/Scripts/python.exe -m pytest -q
```

Expected: no new failures against the pre-task baseline. Capture the baseline
first if you have not:

```bash
git stash && .venv/Scripts/python.exe -m pytest -q ; git stash pop
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_cohort_audio_gl.py
git commit -m "test: a masked mapping moves only the cohorts it names"
```

---

## Manual verification

Automated tests cannot say whether this is nice to use. Run the app, then:

1. Extras > Audio Reactive, pick a loopback device, Start.
2. Turn on `color_by_cohort` in the physics panel — without it you are aiming blind.
3. On the Sensor Gain row, click the `B` chip, open the drawer, and paint the first half of the strip.
4. Click `H`, switch its mode to subtract, and paint the second half.
5. Play something with both bass and hats. Half the swarm should tighten on the kick while the other half loosens on the hats.
6. Move Number of Cohorts from 64 to 128 and back. The split must stay at the halfway point.
7. Open the `Trail Persistence` row's drawer. It must say the parameter cannot be split by cohort, and draw no strip.

## Removal check

Before calling this done, confirm section 8 of the spec is still true: the six
deletions listed there remove the feature and nothing else. If any task has put
feature logic into a file that section does not name, move it into
`services/cohort_audio.py` or `ui/cohort_strip.py`.
