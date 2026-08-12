# Soliton Objective Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reward spatially localized patterns (Lenia "SLPs" / solitons) — compact
structures that keep their identity against an empty background — with a stored
archive column and a fourth expedition goal kind.

**Architecture:** Three per-tile image statistics multiply into one score.
`locality` (new) is a histogram statistic: how unevenly the light is
distributed. `persistence` (new) is a temporal cross-correlation maximised over
cyclic spatial shift: is this the same form, allowing that it moved.
`structure` (existing) is a spatial autocorrelation. All three are stored per
archive entry; the product is derived. Stage 1 builds and calibrates the
measure; stage 2 lets CMA-ES climb it.

**Tech Stack:** Python 3.12, NumPy, ModernGL/GLFW (calibration tool only),
pytest. No new dependencies. CLIP is never touched by any of this.

**Spec:** `docs/superpowers/specs/2026-08-11-soliton-objective-design.md`

**Worktree:** `.claude/worktrees/integration`, branch `integration`. Every line
number below is against **that** checkout, which is 49 commits ahead of
`tournament-mode` and contains it. Do not execute this plan from the main
working directory — `services/archive.py` and `services/imgep_driver.py` differ
substantially between the two.

**What `integration` changed that this plan depends on:** the archive is now
**multi-layout**. `Archive._next_ids` is a dict keyed by brain signature,
`load_from_store` delegates to `_load_one(sig, store)` per layout directory, and
`ArchiveEntry` carries a `layout` field that is deliberately **stripped from the
index row** before writing (`row.pop("layout", None)` in `_add`) because the
directory already names it. The three new fields follow the opposite rule — they
*are* written to the index, because nothing else records them.

## Global Constraints

- **Tests run on the venv interpreter:** `.venv/Scripts/python.exe -m pytest -q`.
  Bare `python` is 3.10 with no pytest.
- **Never `git add -A`** in this repo. Stage explicit paths only.
- **Comments state the rule, never the evidence.** No measured percentages,
  timings, dates or before/after comparisons in comments, docstrings or
  tooltips. Measured facts live in `CLAUDE.md` or under `docs/`; code carries a
  one-line pointer.
- **A tooltip is ONE sentence** naming what the control does.
- **Naming:** `SimState` fields and GLSL uniforms are `ALL_CAPS_UNDERSCORE`; UI
  labels are Title Case with spaces; private UI state is `_snake_case`.
- **Windows platform:** forward slashes or `os.path`; `rm` not `del` in bash.
- **`sim.py` is user-owned.** Nothing in this plan modifies it.
- **Downsample target is 56, not 64.** `224 = 4 × 56` exactly, so the box filter
  is exact and loses no border. At 64 the block mean would crop 224→192 and
  discard the outer 16px of every edge — which is precisely where a soliton
  near a tile boundary lives.
- Every commit message ends with:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

---

## File Structure

**Created:**
- `tools/calibrate_soliton.py` — the calibration tool; sets the two constants.

**Modified:**
- `services/capture_health.py` — `_area_resize`, `_luma_small`, `locality`,
  `persistence`. Sibling of the existing `structure`.
- `services/imgep_driver.py` — `Precomputed` tuple; `precompute` computes the
  statistics; `tell` threads them into `Candidate`; `_expedition_fitness` gains
  a soliton branch; `_draw_goal` gains a share.
- `services/archive.py` — `Candidate` and `ArchiveEntry` gain three fields plus
  a derived `soliton` property; `_add` and `load_from_store` carry them.
- `services/goal_source.py` — `soliton_goal`.
- `state/archive_state.py` — `soliton_share`, added to `PERSISTED_FIELDS`.
- `ui/archive_window.py` — share slider; sort mode.
- `services/map_view.py` — colour mode.
- `CLAUDE.md` — one caveat recording the calibrated constants.

**Tests:**
- `tests/test_capture_health.py` (exists) — locality and persistence.
- `tests/test_goal_source.py` (exists) — the soliton goal.
- `tests/test_archive.py` (exists) — the three fields round-trip.
- `tests/test_imgep_driver.py` (exists) — `Precomputed` through `tell`.

---

# STAGE 1 — the measure, the column, the calibration

### Task 1: `locality` in capture_health

**Files:**
- Modify: `services/capture_health.py` (append after `structure`, line 85)
- Test: `tests/test_capture_health.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `SOLITON_PX = 56`, `MASS_FRACTION = 0.9`, `MIN_SUPPORT = 0.005`,
  `_area_resize(g: np.ndarray, px: int) -> np.ndarray`,
  `_luma_small(crops: np.ndarray, px: int = SOLITON_PX) -> np.ndarray` returning
  `(n, px, px)` float32, and
  `locality(crops: np.ndarray, mass_fraction: float = MASS_FRACTION, min_support: float = MIN_SUPPORT) -> np.ndarray`
  returning `(n,)` float32 in `[0, 1]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_capture_health.py`:

```python
import numpy as np

from services.capture_health import SOLITON_PX, locality, structure


def _blank(n=1, px=224):
    return np.zeros((n, px, px, 3), dtype=np.uint8)


def _blob(px=224, cx=112, cy=112, sigma=14.0, peak=255.0):
    """One Gaussian blob on black, as an (px, px, 3) uint8 image."""
    y, x = np.mgrid[0:px, 0:px].astype(np.float32)
    g = peak * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * sigma ** 2))
    return np.repeat(g[:, :, None], 3, axis=2).astype(np.uint8)


def test_uniform_field_has_no_locality():
    img = np.full((1, 224, 224, 3), 128, dtype=np.uint8)
    assert locality(img)[0] < 0.05


def test_a_compact_blob_is_local():
    img = _blob()[None, ...]
    assert locality(img)[0] > 0.8


def test_a_single_dot_scores_below_a_blob():
    """The degenerate the support ramp exists to close: one bright speck
    maximises concentration, and must not beat a real object."""
    dot = _blank()
    dot[0, 112, 112] = 255
    assert locality(dot)[0] < locality(_blob()[None, ...])[0]
    assert locality(dot)[0] < 0.2


def test_locality_ignores_where_the_light_is():
    """A histogram statistic: scrambling the pixels must not move it."""
    img = _blob()[None, ...]
    rng = np.random.default_rng(0)
    flat = img.reshape(1, -1, 3)
    shuffled = flat[:, rng.permutation(flat.shape[1]), :].reshape(img.shape)
    assert abs(locality(img)[0] - locality(shuffled)[0]) < 0.05


def test_locality_survives_a_brightness_scale():
    img = _blob()[None, ...]
    dim = (img.astype(np.float32) * 0.5).astype(np.uint8)
    assert abs(locality(img)[0] - locality(dim)[0]) < 0.05


def test_a_black_tile_is_zero_not_nan():
    out = locality(_blank())
    assert out[0] == 0.0


def test_locality_handles_an_empty_batch():
    assert locality(np.zeros((0, 224, 224, 3), dtype=np.uint8)).shape == (0,)


def test_scrambled_dots_are_local_but_unstructured():
    """The off-diagonal cell: mass concentrated, no spatial correlation.
    locality alone would reward this; structure is what rejects it."""
    rng = np.random.default_rng(1)
    img = _blank()
    idx = rng.choice(224 * 224, size=600, replace=False)
    flat = img.reshape(1, -1, 3)
    flat[0, idx, :] = 255
    img = flat.reshape(1, 224, 224, 3)
    assert locality(img)[0] > 0.7
    assert structure(img)[0] < 0.3


def test_downsample_target_divides_the_tile_exactly():
    assert 224 % SOLITON_PX == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_capture_health.py -q`
Expected: FAIL with `ImportError: cannot import name 'SOLITON_PX'`.

- [ ] **Step 3: Implement**

Append to `services/capture_health.py`:

```python
# 224 = 4 * SOLITON_PX exactly, so the box filter below loses no border.
SOLITON_PX = 56
MASS_FRACTION = 0.9
MIN_SUPPORT = 0.005


def _area_resize(g: np.ndarray, px: int) -> np.ndarray:
    """(n, H, W) -> (n, px, px) by area-averaging.

    Exact for an integer ratio, a box approximation otherwise, so a 224 crop
    and a 160 thumbnail reduce to the same grid.
    """
    n, h, w = g.shape
    if px >= min(h, w):
        return g
    ry = (np.arange(px + 1) * h) // px
    rx = (np.arange(px + 1) * w) // px
    rows = np.add.reduceat(g, ry[:-1], axis=1) / np.diff(ry)[None, :, None]
    return rows.__itruediv__(1.0) if False else (
        np.add.reduceat(rows, rx[:-1], axis=2) / np.diff(rx)[None, None, :]
    )


def _luma_small(crops: np.ndarray, px: int = SOLITON_PX) -> np.ndarray:
    """(n, H, W, 3) uint8 -> (n, px, px) float32 luminance.

    Area-averaged, never strided: these statistics are mass-weighted and a
    stride would discard most of the light.
    """
    a = np.asarray(crops)
    if a.ndim != 4 or a.shape[0] == 0:
        return np.zeros((0, px, px), dtype=np.float32)
    return _area_resize(a.astype(np.float32).mean(axis=3), px)


def locality(crops: np.ndarray, mass_fraction: float = MASS_FRACTION,
             min_support: float = MIN_SUPPORT) -> np.ndarray:
    """(n, H, W, 3) uint8 -> (n,) in [0, 1]. 1 is concentrated, 0 is spread.

    The fraction of pixels needed to hold `mass_fraction` of the light drives
    both terms: concentration falls as that fraction grows, and support
    discounts a bright set too small to be an object. One bright speck would
    otherwise maximise the measure.

    A histogram statistic, so it is invariant to any spatial permutation and
    the tile's torus never enters. Deliberately not a participation ratio,
    whose second moment loses to bloom. See CLAUDE.md.
    """
    g = _luma_small(crops)
    if g.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    n_px = g.shape[1] * g.shape[2]
    s = -np.sort(-g.reshape(g.shape[0], n_px), axis=1)
    total = s.sum(axis=1)
    need = float(mass_fraction) * total
    # k* is a COUNT, so the number of prefixes still short of `need`, plus one.
    kstar = (np.cumsum(s, axis=1) < need[:, None]).sum(axis=1) + 1
    a = kstar.astype(np.float32) / float(n_px)
    concentration = 1.0 - np.clip(a / float(mass_fraction), 0.0, 1.0)
    support = np.minimum(1.0, a / max(float(min_support), 1e-9))
    return np.where(total > 0, concentration * support, 0.0).astype(np.float32)
```

- [ ] **Step 4: Simplify `_area_resize`**

The `__itruediv__` line above is dead scaffolding. Replace the body of
`_area_resize` after the guard with exactly:

```python
    ry = (np.arange(px + 1) * h) // px
    rx = (np.arange(px + 1) * w) // px
    rows = np.add.reduceat(g, ry[:-1], axis=1) / np.diff(ry)[None, :, None]
    return np.add.reduceat(rows, rx[:-1], axis=2) / np.diff(rx)[None, None, :]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_capture_health.py -q`
Expected: PASS, all tests.

- [ ] **Step 6: Commit**

```bash
git add services/capture_health.py tests/test_capture_health.py
git commit -m "feat: locality measures how unevenly a tile's light is spread

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: `persistence` in capture_health

**Files:**
- Modify: `services/capture_health.py` (append after `locality`)
- Test: `tests/test_capture_health.py`

**Interfaces:**
- Consumes: `_luma_small`, `SOLITON_PX` from Task 1.
- Produces: `persistence(snapshots: list[np.ndarray]) -> np.ndarray` returning
  `(n,)` float32 in `[0, 1]`. `snapshots` is a list of `(n, H, W, 3)` uint8
  arrays, one per snapshot, all the same shape — the same list `ImgepDriver.tell`
  already receives.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_capture_health.py`:

```python
from services.capture_health import persistence


def test_a_frozen_pattern_persists():
    img = _blob()[None, ...]
    assert persistence([img, img, img])[0] > 0.99


def test_a_glider_persists_across_the_wrap_seam():
    """The torus case. A blob started near the edge and rolled past it must
    still correlate with itself; a non-cyclic shift would score it dead."""
    base = _blob(cx=8, cy=112)
    snaps = [np.roll(base, 24 * t, axis=1)[None, ...] for t in range(4)]
    assert persistence(snaps)[0] > 0.95


def test_independent_noise_does_not_persist():
    rng = np.random.default_rng(2)
    snaps = [rng.integers(0, 256, (1, 224, 224, 3), dtype=np.uint8)
             for _ in range(3)]
    assert persistence(snaps)[0] < 0.2


def test_persistence_needs_two_snapshots():
    img = _blob()[None, ...]
    out = persistence([img])
    assert out.shape == (1,) and out[0] == 0.0


def test_persistence_survives_a_brightness_change_between_frames():
    """Each frame is normalised on its own, so a global fade is not a change
    of form."""
    img = _blob()[None, ...]
    dim = (img.astype(np.float32) * 0.4).astype(np.uint8)
    assert persistence([img, dim])[0] > 0.99


def test_persistence_scores_each_tile_separately():
    blob = _blob()[None, ...]
    rng = np.random.default_rng(3)
    noise = rng.integers(0, 256, (1, 224, 224, 3), dtype=np.uint8)
    a = np.concatenate([blob, noise], axis=0)
    b = np.concatenate([blob, rng.integers(0, 256, (1, 224, 224, 3),
                                           dtype=np.uint8)], axis=0)
    out = persistence([a, b])
    assert out[0] > 0.99 and out[1] < 0.2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_capture_health.py -q`
Expected: FAIL with `ImportError: cannot import name 'persistence'`.

- [ ] **Step 3: Implement**

Append to `services/capture_health.py`:

```python
def _unit_frames(g: np.ndarray) -> np.ndarray:
    """(n, px, px) -> zero mean and unit L2 norm per frame."""
    x = g - g.mean(axis=(1, 2), keepdims=True)
    nrm = np.sqrt((x * x).sum(axis=(1, 2), keepdims=True))
    return x / np.maximum(nrm, 1e-12)


def persistence(snapshots) -> np.ndarray:
    """[(n, H, W, 3)] * S -> (n,) in [0, 1]. 1 is the same form throughout.

    The maximum over CYCLIC shifts of the cross correlation between consecutive
    snapshots, averaged over pairs. Cyclic is exact rather than an
    approximation: a tournament tile is a torus under wrap, so a structure
    leaving one edge arrives at the other.

    Including the zero shift is deliberate - a stationary pattern that keeps
    its shape is a valid target, and motion is optional.

    Unit-norm frames make the correlation a cosine, so the maximum needs no
    further scaling. NOT the complement of liveness: that compares CLIP
    embeddings at zero shift, and CLIP is position-dependent enough to read a
    move as a change. See CLAUDE.md.

    S < 2 returns zeros - one snapshot is no evidence either way - so callers
    must disable any gate on it rather than reject everything.
    """
    if len(snapshots) < 2:
        n = int(np.asarray(snapshots[0]).shape[0]) if len(snapshots) else 0
        return np.zeros(n, dtype=np.float32)

    frames = [_luma_small(s) for s in snapshots]
    n = frames[0].shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    shape = frames[0].shape[1:]
    scores = np.empty((len(frames) - 1, n), dtype=np.float32)
    for t in range(len(frames) - 1):
        a = np.fft.rfft2(_unit_frames(frames[t]))
        b = np.fft.rfft2(_unit_frames(frames[t + 1]))
        corr = np.fft.irfft2(np.conj(a) * b, s=shape)
        scores[t] = corr.reshape(n, -1).max(axis=1)
    return np.clip(scores.mean(axis=0), 0.0, 1.0).astype(np.float32)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_capture_health.py -q`
Expected: PASS, all tests.

- [ ] **Step 5: Commit**

```bash
git add services/capture_health.py tests/test_capture_health.py
git commit -m "feat: persistence asks whether a tile is the same form, moved

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: thread the statistics through `precompute` and `tell`

**Files:**
- Modify: `services/imgep_driver.py:15` (import), `:407-415` (`precompute`),
  `:423` (`tell`'s `per_snap`), `:458-462` (`coherence` and `fit`),
  `:494-505` (the `Candidate` construction)
- Modify: `services/archive.py:45-57` (`Candidate`)
- Test: `tests/test_imgep_driver.py`

**Interfaces:**
- Consumes: `locality`, `persistence` from Tasks 1–2.
- Produces: `Precomputed` — a `NamedTuple` with fields
  `embeddings: list[np.ndarray]`, `locality: np.ndarray`,
  `persistence: np.ndarray`, `structure: np.ndarray`. `ImgepDriver.precompute`
  returns it; `ImgepDriver.tell(z, snapshots, pre=None)` accepts it.
  `Candidate` gains `locality: float = 0.0`, `persistence: float = 0.0`,
  `structure: float = 0.0`.

**Why `structure` joins them:** the stored soliton score is
`locality * persistence * structure`, and `structure` is already computed inside
`tell` as `coherence`. Storing only two of the three factors would make the
browser's column and the expedition's fitness rank differently.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_imgep_driver.py`:

```python
import numpy as np

from services.imgep_driver import Precomputed


def test_precompute_returns_the_three_statistics(driver_and_snapshots):
    """precompute is the worker-thread half; it must carry everything tell
    needs that is not archive state."""
    driver, z, snapshots = driver_and_snapshots
    pre = driver.precompute(snapshots)
    assert isinstance(pre, Precomputed)
    n = len(z)
    assert len(pre.embeddings) == len(snapshots)
    for name in ("locality", "persistence", "structure"):
        arr = getattr(pre, name)
        assert arr.shape == (n,)
        assert np.all(arr >= 0.0) and np.all(arr <= 1.0)


def test_tell_accepts_a_precomputed_or_recomputes_it(driver_and_snapshots):
    driver, z, snapshots = driver_and_snapshots
    with_pre = driver.tell(z, snapshots, pre=driver.precompute(snapshots))
    assert with_pre.shape == (len(z),)


def test_tell_without_pre_still_works(driver_and_snapshots):
    driver, z, snapshots = driver_and_snapshots
    assert driver.tell(z, snapshots).shape == (len(z),)
```

Add this fixture to `tests/test_imgep_driver.py` if no equivalent exists —
match the existing fakes in that file for `scorer`, `tournament` and `spec`
rather than inventing new ones:

```python
import pytest


@pytest.fixture
def driver_and_snapshots(make_driver):
    """`make_driver` is the file's existing helper. Two snapshots of four
    tiles, so liveness and persistence both have a pair to work with."""
    driver = make_driver()
    n = 4
    rng = np.random.default_rng(0)
    snapshots = [rng.integers(0, 256, (n, 224, 224, 3), dtype=np.uint8)
                 for _ in range(2)]
    z = np.zeros((n, driver.spec.dim), dtype=np.float32)
    return driver, z, snapshots
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/test_imgep_driver.py -q`
Expected: FAIL with `ImportError: cannot import name 'Precomputed'`.

- [ ] **Step 3: Add the fields to `Candidate`**

In `services/archive.py`, in the `Candidate` dataclass (line 45), after
`goal: str = ""`:

```python
    # The soliton factors, each a static property of the capture. Stored
    # separately rather than as their product so a legacy entry can be
    # partially backfilled. See CLAUDE.md.
    locality: float = 0.0
    persistence: float = 0.0
    structure: float = 0.0
```

- [ ] **Step 4: Add `Precomputed` and rewrite `precompute`**

In `services/imgep_driver.py`, change the import on line 15 to:

```python
from services.capture_health import is_viable_tile, locality, persistence, structure
```

Add near the top of the module, after the imports:

```python
class Precomputed(NamedTuple):
    """Everything the worker thread can produce without touching the archive.

    The split is exactly this boundary: CLIP and the image statistics take only
    their own arguments and the scorer, so they need no locking. See CLAUDE.md.
    """
    embeddings: list
    locality: np.ndarray
    persistence: np.ndarray
    structure: np.ndarray
```

Add `from typing import NamedTuple` to the module's imports.

Replace the body of `precompute` (line 415) with:

```python
        return Precomputed(
            embeddings=[np.asarray(self._embed(c), dtype=np.float32)
                        for c in snapshots],
            locality=locality(snapshots[-1]),
            persistence=persistence(snapshots),
            structure=structure(snapshots[-1]),
        )
```

- [ ] **Step 5: Rewrite `tell` to unpack it**

In `services/imgep_driver.py`, replace line 423:

```python
        per_snap = pre if pre is not None else self.precompute(snapshots)
```

with:

```python
        pre = pre if pre is not None else self.precompute(snapshots)
        per_snap = pre.embeddings
```

Then delete line 458 (`coherence = structure(last)`) and its comment, and
replace the `fit = ...` call on line 461 with:

```python
        # Computed on the worker thread this generation; reused by the fitness
        # for every goal kind.
        coherence = pre.structure
        # BEFORE the admission loop, not after it, so the summit ratchet can
        # see this generation's fitness.
        fit = (self._expedition_fitness(snaps, nov, coherence)
               if source == "expedition" else None)
```

- [ ] **Step 6: Pass the three factors into each `Candidate`**

In `services/imgep_driver.py`, in the `Candidate(...)` construction at line 494,
after `goal=tile_goal,` (line 504):

```python
                    locality=float(pre.locality[i]),
                    persistence=float(pre.persistence[i]),
                    structure=float(coherence[i]),
```

- [ ] **Step 7: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS. If `tests/test_async_scoring.py` fails on the shape of
`precompute`'s return, update its assertions to read `.embeddings` — that test
exists to prove scoring runs off the frame loop, and the boundary is unchanged.

- [ ] **Step 8: Commit**

```bash
git add services/imgep_driver.py services/archive.py tests/test_imgep_driver.py
git commit -m "feat: the worker thread computes the soliton factors too

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: store the three factors on an archive entry

**Files:**
- Modify: `services/archive.py:60-79` (`ArchiveEntry`), `:384-422` (`_add`),
  `:626-641` (the `ArchiveEntry` construction inside `_load_one`)
- Test: `tests/test_archive.py`

**Interfaces:**
- Consumes: `Candidate.locality/persistence/structure` from Task 3.
- Produces: `ArchiveEntry.locality`, `.persistence`, `.structure`, each
  `float | None`, and a read-only property `ArchiveEntry.soliton -> float | None`
  returning `locality * persistence * structure`, or `None` when any factor is
  missing.

**Why `None` and not `0.0`:** an archive written before this exists has no
values, and reading "absent" as "scored zero" would sort real entries below
unmeasured ones and feed a soliton goal's seed sampler a lie.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_archive.py`:

```python
def test_an_entry_carries_the_three_soliton_factors():
    a = fresh()
    entry = a.consider(
        cand([1.0, 0.0, 0.0, 0.0], locality=0.8, persistence=0.5,
             structure=0.5),
        novelty=1.0)
    assert entry is not None
    assert entry.locality == pytest.approx(0.8)
    assert entry.soliton == pytest.approx(0.8 * 0.5 * 0.5)


def test_the_factors_round_trip_through_the_index(tmp_path):
    store = ArchiveStore(tmp_path)
    a = Archive(store=store, dim=4, liveness_min=0.0, capacity=100)
    a.consider(cand([1.0, 0.0, 0.0, 0.0], locality=0.6, persistence=0.7,
                    structure=0.9), novelty=1.0)
    a.maybe_flush(force=True)
    store.close()

    b = Archive(store=ArchiveStore(tmp_path), dim=4)
    b.load_from_store()
    e = b.entries[0]
    assert e.locality == pytest.approx(0.6)
    assert e.persistence == pytest.approx(0.7)
    assert e.structure == pytest.approx(0.9)
    assert e.soliton == pytest.approx(0.6 * 0.7 * 0.9)


def test_a_legacy_entry_has_no_soliton_score(tmp_path):
    """Absent must read as unknown, never as zero: a stored 0.0 would sort
    real entries below unmeasured ones and lie to the seed sampler."""
    import json

    store, a = _seeded_store_archive(tmp_path, n=1)
    a.maybe_flush(force=True)

    # Strip the keys from every index row, as a pre-feature archive would be.
    path = store.index_path
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        for key in ("locality", "persistence", "structure"):
            r.pop(key, None)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows),
                    encoding="utf-8")
    store.close()

    b = Archive(store=ArchiveStore(tmp_path), dim=4)
    loaded, _ = b.load_from_store()
    assert loaded == 1
    assert b.entries[0].locality is None
    assert b.entries[0].soliton is None
```

`fresh`, `cand`, `_seeded_store_archive`, `Archive` and `ArchiveStore` are all
already defined or imported at the top of `tests/test_archive.py`. Extend the
`cand(...)` helper (line 13) to accept the three new keyword arguments,
defaulting to `0.0`, and pass them through to `Candidate`:

```python
def cand(vec, liveness=1.0, viable=True, dim=4, spec="brain:80",
         locality=0.0, persistence=0.0, structure=0.0):
    e = np.zeros(dim, dtype=np.float32)
    e[: len(vec)] = vec
    return Candidate(
        brain=np.zeros((10, 8), np.float32),
        physics=np.zeros(8, np.float32),
        embedding=_unit(e[None])[0],
        liveness=liveness,
        viable=viable,
        spec=spec,
        locality=locality,
        persistence=persistence,
        structure=structure,
    )
```

**`store.index_path` is per layout directory.** If `_seeded_store_archive`'s
store writes into a signature subdirectory rather than `tmp_path` itself, take
the path from the store the helper returns — which is what the test above does —
rather than reconstructing it.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_archive.py -q`
Expected: FAIL with `AttributeError: 'ArchiveEntry' object has no attribute 'locality'`.

- [ ] **Step 3: Add the fields and the property**

In `services/archive.py`, in `ArchiveEntry` (line 61), after the `layout: str = ""`
field and its comment (line 79):

```python
    # The soliton factors. None means an entry admitted before they existed,
    # which is not the same as having scored zero.
    locality: float | None = None
    persistence: float | None = None
    structure: float | None = None

    @property
    def soliton(self) -> float | None:
        """The three factors multiplied, or None when any is missing.

        Derived, never stored: a fourth column would drift from the three it
        claims to summarise.
        """
        if (self.locality is None or self.persistence is None
                or self.structure is None):
            return None
        return float(self.locality * self.persistence * self.structure)
```

- [ ] **Step 4: Carry them through `_add`**

In `services/archive.py`, in the `ArchiveEntry(...)` construction inside `_add`
(line 407), change `ts=time.time(), thumb=thumb, layout=sig,` to:

```python
            ts=time.time(), thumb=thumb, layout=sig,
            locality=float(cand.locality), persistence=float(cand.persistence),
            structure=float(cand.structure),
```

**Leave the row writer alone.** `_add` builds the index row with
`asdict(entry)` and then `row.pop("layout", None)` — the three new fields must
NOT be popped. `layout` is excluded because the directory the row lands in
already names it; nothing else records the soliton factors, so they belong on
disk. A `@property` is not a dataclass field, so `soliton` is correctly absent
from `asdict` and never written.

- [ ] **Step 5: Read them back in `load_from_store`**

The read side is **`_load_one`, not `load_from_store`** — integration reads one
layout directory at a time and `load_from_store` only drives it. In
`services/archive.py`, in the `ArchiveEntry(...)` construction inside
`_load_one` (line 626), after `layout=sig,` (line 640):

```python
                locality=_opt_float(r.get("locality")),
                persistence=_opt_float(r.get("persistence")),
                structure=_opt_float(r.get("structure")),
```

And add this helper at module level, after `DEFAULT_MIN_SEPARATION` (line 88):

```python
def _opt_float(value) -> float | None:
    """A stored factor, or None when the archive predates it. A hand-edited
    or torn value is unknown rather than zero, for the same reason."""
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None
```

- [ ] **Step 6: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add services/archive.py tests/test_archive.py
git commit -m "feat: an archive entry records what made it a soliton

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: the calibration tool

**Files:**
- Create: `tools/calibrate_soliton.py`
- Reference: `tools/capture_presets.py` (the hidden-window pattern this copies)

**Interfaces:**
- Consumes: `locality`, `persistence`, `structure` from Tasks 1–2.
- Produces: a command-line tool. Nothing imports it.

**Why the real `Camera` through a hidden window:** `tools/capture_presets.py`
records that its first version fed `sim.can` straight to `FrameAssembler` and
reported a third of the library as near-black, because the app's view renders
particles additively over the trails. Anything that re-derives the view path
drifts from it.

**Why bloom is applied here explicitly:** `CaptureView.draw_grid` blits into a
tile-sized target and blooms *that*, per tile. This tool has one tile, so it
must do the same in the same order, or the number it reports is not the number
the optimizer sees.

- [ ] **Step 1: Write the tool**

Create `tools/calibrate_soliton.py`:

```python
"""What do locality and persistence actually score over the preset library?

    python -m tools.calibrate_soliton
    python -m tools.calibrate_soliton --no-bloom
    python -m tools.calibrate_soliton --only Zipper,Karst
    python -m tools.calibrate_soliton --contact-sheet out_dir

MIN_SUPPORT and MASS_FRACTION cannot be set on paper: they decide which end of
a real distribution counts as an object, and the distribution is a property of
the running sim. This steps each preset, grabs several snapshots and reports
where the library falls.

--no-bloom answers a separate question. Bloom is applied to every tile the
optimizer scores and its settings are user preferences the search does not
control, so how far it moves these statistics is worth knowing rather than
assuming.

USES THE REAL Camera through a hidden GLFW window, for the reason
tools/capture_presets.py records: the app's view renders particles additively
over the trails, and a re-derived view path drifts from it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# The app has a pre-existing import cycle that resolves only when `ui` is
# imported first, which is what main.py happens to do. Prime that order.
import ui  # noqa: F401,E402

VIEW_CAM_BRUSH = 2
WINDOW_PX = 1024
TILE_PX = 224


def find_presets(root: Path) -> list[Path]:
    out: list[Path] = []
    for sub in ("Core", "Advanced"):
        d = root / sub
        if d.is_dir():
            out.extend(sorted(d.glob("*.json")))
    return out


class OneTileCapture:
    """The app's per-tile bloom order, for a grid of one.

    draw_grid blits into a tile-sized target and blooms that, not the grid, so
    the same order is what makes this measurement comparable.
    """

    def __init__(self, ctx, camera, tile_px=TILE_PX):
        from services.capture_blit import CaptureBlit
        from services.tile_capture import TileCapture

        self.ctx = ctx
        self.camera = camera
        self.tile_px = tile_px
        self.cap = TileCapture(ctx, grid=1, tile_px=tile_px)
        self.blit = CaptureBlit(ctx)
        self.tex = ctx.texture((tile_px, tile_px), 4, dtype="f4")
        self.fbo = ctx.framebuffer(color_attachments=[self.tex])

    def grab(self, assembled, prefs, bloom: bool) -> np.ndarray:
        """-> one (tile_px, tile_px, 3) uint8 crop."""
        if not bloom:
            return self.cap.capture(
                lambda fbo: self.blit.draw(assembled, (0.0, 0.0), (1.0, 1.0))
            )[0]

        self.fbo.use()
        self.ctx.viewport = (0, 0, self.tile_px, self.tile_px)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.blit.draw(assembled, (0.0, 0.0), (1.0, 1.0))
        bloomed = self.camera.apply_bloom(
            self.tex, prefs.bloom_threshold, prefs.bloom_intensity,
            prefs.bloom_radius, tonemap_softness=prefs.tonemap_softness)
        return self.cap.capture(
            lambda fbo: self.blit.draw(bloomed, (0.0, 0.0), (1.0, 1.0))
        )[0]


def run_preset(ctx, sim, camera, grabber, config_saver, path, steps,
               snaps, seed, prefs, bloom):
    """-> a list of `snaps` crops, evenly spaced over the last half of the run,
    or None if the preset would not load."""
    from state import SimState

    config = config_saver.load_from_file(path)
    if config is None:
        return None

    state = SimState()
    rule = config_saver.apply_config(config, state)
    sim.apply_state(state)
    sim.apply_rule(rule)
    sim.reset_seed = float(seed)
    sim.reset()

    camera.watercolor_mode = state.watercolor_mode
    camera.ink_weight = state.ink_weight

    # Snapshots over the LAST half only: the reset transient is not the
    # pattern, and persistence would read the settling as a change of form.
    first = max(1, int(steps) // 2)
    every = max(1, (int(steps) - first) // max(1, int(snaps)))
    out = []
    for step in range(int(steps)):
        sim.update(ctx)
        if step >= first and (step - first) % every == 0 and len(out) < snaps:
            assembled = camera.frame_assembler.assemble_frame(
                camera.generate_view_texture(), total_samples=1,
                current_sample_index=0, view_mode=VIEW_CAM_BRUSH,
                screen_aspect=1.0, brightness=camera.BRIGHTNESS,
                exposure=prefs.exposure, ink_weight=state.ink_weight,
                watercolor_mode=state.watercolor_mode,
                camera_position=tuple(camera.position), camera_zoom=camera.zoom,
                canvas_resolution=sim.get_canvas_dimensions(),
                tonemap_softness=prefs.tonemap_softness,
                trail_overlay_strength=camera.trail_overlay_strength,
            )
            if assembled is not None:
                out.append(grabber.grab(assembled, prefs, bloom))
    return out if len(out) >= 2 else None


def report(name: str, values: np.ndarray) -> None:
    q = np.percentile(values, [0, 10, 25, 50, 75, 90, 100])
    print(f"  {name:<12} min {q[0]:.4f}  p10 {q[1]:.4f}  p25 {q[2]:.4f}  "
          f"med {q[3]:.4f}  p75 {q[4]:.4f}  p90 {q[5]:.4f}  max {q[6]:.4f}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--snaps", type=int, default=4)
    ap.add_argument("--only", default="", help="comma-separated preset stems")
    ap.add_argument("--limit", type=int, default=0, help="0 = every preset")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--no-bloom", action="store_true",
                    help="measure without the bloom the optimizer actually sees")
    ap.add_argument("--contact-sheet", default="",
                    help="write top and bottom decile crops to this directory")
    args = ap.parse_args(argv)

    import glfw
    import moderngl

    from camera import Camera
    from services.capture_health import locality, persistence, structure
    from services.config_saver import ConfigSaver
    from sim import Sim
    from state.preferences_state import PreferencesState
    from utilities.paths import get_app_physics_configs_dir

    presets = find_presets(get_app_physics_configs_dir())
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        presets = [p for p in presets if p.stem in want]
    if args.limit:
        presets = presets[:: max(1, len(presets) // args.limit)][: args.limit]
    if not presets:
        print("no presets found")
        return 2

    if not glfw.init():
        print("glfw init failed")
        return 2
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(WINDOW_PX, WINDOW_PX, "calibrate", None, None)
    if not window:
        glfw.terminate()
        print("could not create a hidden window")
        return 2
    glfw.make_context_current(window)
    ctx = moderngl.create_context()

    prefs = PreferencesState()
    sim = Sim(ctx, world_size=prefs.world_size, canvas_aspect_ratio="1:1",
              particle_density=prefs.particle_density)
    camera = Camera(ctx, sim, window)
    camera.cam_brush_mode = True
    camera.BRIGHTNESS = prefs.brightness
    camera.trail_overlay_strength = prefs.trail_overlay_strength
    grabber = OneTileCapture(ctx, camera)
    config_saver = ConfigSaver()

    bloom = not args.no_bloom
    print(f"{len(presets)} presets, {args.steps} steps, {args.snaps} snapshots, "
          f"bloom {'on' if bloom else 'OFF'}")
    print(f"world_size {prefs.world_size}  density {prefs.particle_density}\n")

    names, rows, keep = [], [], {}
    for path in presets:
        snaps = run_preset(ctx, sim, camera, grabber, config_saver, path,
                           args.steps, args.snaps, args.seed, prefs, bloom)
        if not snaps:
            print(f"  {path.stem:<28} skipped")
            continue
        stack = [s[None, ...] for s in snaps]
        loc = float(locality(stack[-1])[0])
        per = float(persistence(stack)[0])
        stc = float(structure(stack[-1])[0])
        names.append(path.stem)
        rows.append((loc, per, stc, loc * per * stc))
        keep[path.stem] = snaps[-1]
        print(f"  {path.stem:<28} loc {loc:.4f}  per {per:.4f}  "
              f"str {stc:.4f}  ->  {loc * per * stc:.4f}")

    if not rows:
        print("\nnothing measured")
        return 1

    arr = np.array(rows, dtype=np.float64)
    print(f"\n{len(rows)} presets measured, bloom {'on' if bloom else 'OFF'}")
    for i, label in enumerate(("locality", "persistence", "structure", "soliton")):
        report(label, arr[:, i])

    order = np.argsort(-arr[:, 3])
    n_show = max(1, len(order) // 10)
    print("\ntop decile:    " + ", ".join(names[i] for i in order[:n_show]))
    print("bottom decile: " + ", ".join(names[i] for i in order[-n_show:]))

    if args.contact_sheet:
        from PIL import Image

        out = Path(args.contact_sheet)
        out.mkdir(parents=True, exist_ok=True)
        for rank, i in enumerate(list(order[:n_show]) + list(order[-n_show:])):
            band = "top" if rank < n_show else "bot"
            Image.fromarray(keep[names[i]]).save(
                out / f"{band}-{arr[i, 3]:.4f}-{names[i]}.png")
        print(f"\ncontact sheet written to {out}")

    glfw.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 2: Verify it runs on a small sample**

Run: `.venv/Scripts/python.exe -m tools.calibrate_soliton --limit 6 --steps 400`
Expected: six lines of per-preset numbers, then a summary block with seven
percentiles per statistic. Every value in `[0, 1]`. If every `soliton` is
exactly 0.0, check that `structure` is not the factor zeroing them — that would
mean the snapshots are noise, i.e. the run is too short rather than the measure
being wrong.

- [ ] **Step 3: Commit**

```bash
git add tools/calibrate_soliton.py
git commit -m "feat: measure where the preset library falls on locality and persistence

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: calibrate, and record the constants

**Files:**
- Modify: `services/capture_health.py` (the two constants)
- Modify: `CLAUDE.md` (one caveat under "CLIP and the capture")

**Interfaces:**
- Consumes: the tool from Task 5.
- Produces: final values for `MASS_FRACTION` and `MIN_SUPPORT`.

**This is a measurement task, not a coding task.** It ends with numbers that
came off a real run, and it is the gate on stage 2.

- [ ] **Step 1: Run the full sweep, bloom on**

```bash
.venv/Scripts/python.exe -m tools.calibrate_soliton --contact-sheet C:/Users/Heysoos/AppData/Local/Temp/soliton-bloom
```

Save the summary block. This takes a while — 131 presets at 2000 steps.

- [ ] **Step 2: Run it again with bloom off**

```bash
.venv/Scripts/python.exe -m tools.calibrate_soliton --no-bloom --contact-sheet C:/Users/Heysoos/AppData/Local/Temp/soliton-nobloom
```

- [ ] **Step 3: Look at the contact sheets**

Open both directories. The question is whether the top decile looks like
localized structures and the bottom decile looks like full-field texture. If the
ranking disagrees with your eye, the measure is wrong and stage 2 must not
proceed — report which presets are misplaced and in which direction.

- [ ] **Step 4: Set `MIN_SUPPORT` from the distribution**

`MIN_SUPPORT` is the smallest light-bearing fraction of a tile that counts as an
object. Set it below the p10 of the *support fraction* among presets you judged
genuinely localized, so the ramp never discounts a real creature — its only job
is the speck. If no preset's support fraction falls near 0.005, move the
constant rather than leaving a ramp that never fires.

Leave `MASS_FRACTION` at `0.9` unless the sweep shows the library bunched
against either end of `locality`, in which case report the distribution before
changing it.

- [ ] **Step 5: Write the caveat**

Add to `CLAUDE.md` under "### CLIP and the capture", stating the rule and
pointing at the tool — no percentages in code, per the global constraints:

```markdown
- **`locality` and `persistence` are calibrated against the preset library, and
  `MIN_SUPPORT` is the only thing standing between the measure and a bright
  speck.** Over the 131 presets, locality runs <MIN>–<MAX> with median <MED>
  and persistence <MIN>–<MAX> with median <MED>; bloom moves the soliton
  product by <DELTA>. A support ramp set above the smallest genuinely localized
  preset discards real creatures, and one set too low leaves the degenerate —
  one dim dot that does not move — maximising the measure. Re-run
  `python -m tools.calibrate_soliton` before changing either constant, and
  `--no-bloom` before changing anything about the capture: bloom's threshold,
  intensity and radius are user preferences the search does not control, so
  they move what the optimizer scores without appearing in any genome.
```

Replace every `<...>` with the measured value. A caveat with a placeholder in it
is worse than no caveat.

- [ ] **Step 6: Commit**

```bash
git add services/capture_health.py CLAUDE.md
git commit -m "docs: what the preset library scores on locality and persistence

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 7: STOP and report**

Stage 1 is complete. Report the distributions and whether the contact sheets
agree with the ranking. **Do not begin stage 2 without that confirmation** — if
the library holds nothing localized, the measure describes a region the
substrate does not reach, and the next question is which physics parameters
move it, not how to climb it.

---

# STAGE 2 — the goal kind

### Task 7: `soliton_goal` in goal_source

**Files:**
- Modify: `services/goal_source.py` (append after `novelty_goal`, line 251)
- Test: `tests/test_goal_source.py`

**Interfaces:**
- Consumes: `ArchiveEntry.soliton` from Task 4.
- Produces: `soliton_goal(archive, rng, alpha: float = 4.0) -> Goal | None`,
  returning `Goal(kind="soliton", text="", embedding=None, seed_index=i)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_goal_source.py`:

```python
import numpy as np

from services.goal_source import soliton_goal


class _FakeEntry:
    def __init__(self, soliton, novelty=0.5):
        self.soliton = soliton
        self.novelty = novelty


class _FakeArchive:
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)


def test_a_soliton_goal_seeds_on_a_high_scoring_entry():
    arc = _FakeArchive([_FakeEntry(0.01), _FakeEntry(0.02), _FakeEntry(0.99)])
    rng = np.random.default_rng(0)
    seeds = [soliton_goal(arc, rng).seed_index for _ in range(50)]
    assert seeds.count(2) > 40      # p ~ soliton^4 concentrates hard


def test_a_soliton_goal_carries_no_embedding():
    arc = _FakeArchive([_FakeEntry(0.5)])
    goal = soliton_goal(arc, np.random.default_rng(0))
    assert goal.kind == "soliton"
    assert goal.embedding is None
    assert goal.seed_index == 0


def test_an_archive_with_no_column_falls_back_to_novelty():
    """An archive built before the column existed must still be able to start
    one, rather than declining and wasting a cadence interval."""
    arc = _FakeArchive([_FakeEntry(None, novelty=0.1),
                        _FakeEntry(None, novelty=0.9)])
    rng = np.random.default_rng(0)
    seeds = [soliton_goal(arc, rng).seed_index for _ in range(50)]
    assert seeds.count(1) > 40


def test_an_empty_archive_declines():
    assert soliton_goal(_FakeArchive([]), np.random.default_rng(0)) is None


def test_an_all_zero_column_still_returns_a_goal():
    """Every entry scoring zero is a flat landscape, not an error."""
    arc = _FakeArchive([_FakeEntry(0.0), _FakeEntry(0.0)])
    assert soliton_goal(arc, np.random.default_rng(0)) is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_goal_source.py -q`
Expected: FAIL with `ImportError: cannot import name 'soliton_goal'`.

- [ ] **Step 3: Implement**

Append to `services/goal_source.py`:

```python
def soliton_goal(archive, rng, alpha: float = 4.0) -> Goal | None:
    """An expedition that climbs the soliton score itself.

    The fourth goal kind, and structurally a sibling of novelty_goal: no
    embedding to point at, so the seed is drawn here rather than by
    _seed_index. Sampled with p proportional to soliton^alpha, the same way
    expansion draws a parent, so repeating the goal explores a different
    trajectory.

    Falls back to sampling on novelty when no entry carries the column. An
    archive built before the factors existed must still be able to start one:
    an expedition that declines wastes a whole cadence interval.
    """
    if len(archive) == 0:
        return None
    scores = np.array(
        [e.soliton if getattr(e, "soliton", None) is not None else np.nan
         for e in archive.entries], dtype=np.float32)
    if not np.isfinite(scores).any():
        scores = np.array([e.novelty for e in archive.entries],
                          dtype=np.float32)
    else:
        # An unmeasured entry is not a zero-scoring one, but it cannot be a
        # seed for a goal it has never been scored against.
        scores = np.where(np.isfinite(scores), scores, 0.0)
    i = int(sample_by_novelty(scores, 1, rng, alpha)[0])
    return Goal("soliton", "", None, seed_index=i)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_goal_source.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/goal_source.py tests/test_goal_source.py
git commit -m "feat: a soliton expedition seeds on the best soliton so far

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: `soliton_share`, the fitness branch, and the slider

**Files:**
- Modify: `services/imgep_driver.py:80` (setting), `:350-392` (`_draw_goal`),
  `:682-712` (`_expedition_fitness`)
- Modify: `state/archive_state.py:65` (field), `:209` (`PERSISTED_FIELDS`)
- Modify: `ui/archive_window.py` (the Explore settings panel, beside the
  existing Novelty Share and Latent Share sliders)
- Test: `tests/test_novelty_goals.py`, `tests/test_label_widths.py`

**Interfaces:**
- Consumes: `soliton_goal` from Task 7; `Precomputed` from Task 3.
- Produces: `ImgepDriver.soliton_share: float = 0.0`,
  `ArchiveState.soliton_share: float = 0.0`, and a
  `kind == "soliton"` branch in `_expedition_fitness` returning
  `locality * persistence * coherence`.

**Default is 0.0, not a share of the existing budget.** Turning a new goal kind
on by default silently re-weights every archive already in use.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_novelty_goals.py`:

```python
def test_a_soliton_expedition_climbs_the_product(make_driver):
    """The fitness for a soliton goal is the stored factors multiplied, not
    anything contrastive - there is no embedding to contrast against."""
    driver = make_driver()
    driver.start_expedition_with(None, "soliton", "", seed_index=0)
    driver._last_locality = np.array([0.9, 0.1], dtype=np.float32)
    driver._last_persistence = np.array([0.5, 0.5], dtype=np.float32)
    coherence = np.array([1.0, 1.0], dtype=np.float32)
    snaps = np.zeros((2, 2, 512), dtype=np.float32)
    fit = driver._expedition_fitness(snaps, np.zeros(2), coherence)
    assert fit[0] > fit[1]
    assert fit[0] == pytest.approx(0.45)


def test_soliton_share_zero_never_draws_one(make_driver):
    driver = make_driver()
    driver.soliton_share = 0.0
    driver.novelty_share = 1.0
    kinds = {driver._draw_goal().kind for _ in range(50)}
    assert "soliton" not in kinds


def test_soliton_share_one_draws_one(make_driver):
    driver = make_driver()
    driver.soliton_share = 1.0
    driver.novelty_share = 0.0
    driver.latent_share = 0.0
    assert driver._draw_goal().kind == "soliton"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_novelty_goals.py -q`
Expected: FAIL with `AttributeError: 'ImgepDriver' object has no attribute 'soliton_share'`.

- [ ] **Step 3: Store the per-generation factors on the driver**

In `services/imgep_driver.py`, in `__init__` after line 80
(`self.novelty_share = 0.25`):

```python
        self.soliton_share = 0.0         # off until a run asks for it
```

And in `__init__`, beside the other per-generation caches:

```python
        self._last_locality = None
        self._last_persistence = None
```

In `tell`, immediately after `per_snap = pre.embeddings`:

```python
        # The fitness needs these and is computed further down, after the
        # archive gates have run.
        self._last_locality = pre.locality
        self._last_persistence = pre.persistence
```

- [ ] **Step 4: Add the fitness branch**

In `services/imgep_driver.py`, in `_expedition_fitness`, immediately after the
`kind == "novelty"` branch (line 700):

```python
        if self._goal is not None and self._goal.kind == "soliton":
            # No embedding to contrast against: the objective IS the stored
            # factors. coherence is the third of them.
            loc = np.asarray(self._last_locality, dtype=np.float32)
            per = np.asarray(self._last_persistence, dtype=np.float32)
            return (loc * per * c).astype(np.float32)
```

- [ ] **Step 5: Add the share to `_draw_goal`**

In `services/imgep_driver.py`, replace `_draw_goal`'s share arithmetic
(lines 358–365) with:

```python
        u = float(self.rng.random())
        nov_share = max(0.0, float(self.novelty_share))
        lat_share = max(0.0, float(self.latent_share))
        sol_share = max(0.0, float(self.soliton_share))
        # Clamped rather than normalised: the sliders are independent, and
        # silently rescaling one because another moved would make none of them
        # mean what they say.
        if nov_share + sol_share > 1.0:
            sol_share = max(0.0, 1.0 - nov_share)
        if nov_share + sol_share + lat_share > 1.0:
            lat_share = max(0.0, 1.0 - nov_share - sol_share)
```

Add the maker beside the others (after line 378):

```python
        def soliton():
            return soliton_goal(self.archive, self.rng, self.alpha)
```

And replace the dispatch (lines 380–387) with:

```python
        if u < nov_share:
            order = (novelty, soliton, latent, lambda: text_goal)
        elif u < nov_share + sol_share:
            order = (soliton, novelty, latent, lambda: text_goal)
        elif u < nov_share + sol_share + lat_share:
            order = (latent, novelty, soliton, lambda: text_goal)
        elif text_goal is not None:
            return text_goal
        else:
            order = (latent, novelty, soliton, lambda: None)
```

Update the import on the `goal_source` line to include `soliton_goal`, and
update `_draw_goal`'s docstring first line to "One of four kinds, by share."

- [ ] **Step 6: Add the state field**

In `state/archive_state.py`, after line 65 (`novelty_share: float = 0.25`):

```python
    # Off by default: enabling a new goal kind silently re-weights every
    # archive already in use.
    soliton_share: float = 0.0
```

And in `PERSISTED_FIELDS` (line 209), change:

```python
    "latent_share", "novelty_share", "goal_order",
```

to:

```python
    "latent_share", "novelty_share", "soliton_share", "goal_order",
```

- [ ] **Step 7: Add the slider**

In `ui/archive_window.py`, find the existing Novelty Share slider and add
directly beneath it, matching the surrounding style exactly:

```python
        _, ast.soliton_share = imgui.slider_float(
            "Soliton Share", ast.soliton_share, 0.0, 1.0)
        _tip("How often an expedition climbs localized, persistent structure.")
```

Use whatever the file's own tooltip helper is called rather than `_tip`, and
keep the slider inside the same `layout.push_settings_width(...)` block the
neighbouring sliders use. `"Soliton Share"` must be no wider than
`WIDEST_LABEL` or `tests/test_label_widths.py` fails.

- [ ] **Step 8: Push the setting to the driver**

In `services/auto_tournament_service.py`'s `configure()` (or wherever
`novelty_share` is copied from `ArchiveState` onto the driver — grep for
`novelty_share =` outside `__init__`), add the same line for `soliton_share`.

- [ ] **Step 9: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS, including `tests/test_label_widths.py` and
`tests/test_archive_state.py`.

- [ ] **Step 10: Commit**

```bash
git add services/imgep_driver.py state/archive_state.py ui/archive_window.py services/auto_tournament_service.py tests/test_novelty_goals.py
git commit -m "feat: an expedition can chase localized structure

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: sort and colour by soliton

**Files:**
- Modify: `ui/archive_window.py:603-609` (`keyfn`), `:614-621` (`modes`)
- Modify: `services/map_view.py:28` (`COLOR_MODES`), `:93-96` (`_values`)
- Test: `tests/test_map_view.py`, `tests/test_archive_window_render.py`

**Interfaces:**
- Consumes: `ArchiveEntry.soliton` from Task 4.
- Produces: `"soliton"` as a valid `ArchiveState.sort_by` and `map_color_by`.

**An entry with no score sorts last and colours as unknown**, never as zero —
the same rule Task 4 established.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_map_view.py`:

```python
def test_soliton_is_a_colour_mode():
    from services.map_view import COLOR_MODES

    assert "soliton" in COLOR_MODES


def test_an_unmeasured_entry_colours_as_zero_not_as_a_crash():
    """None must not reach the ramp. It is drawn at the bottom of the range,
    which is honest for a colour and would be a lie for a sort key."""
    from services.map_view import _values

    class _E:
        soliton = None
        novelty = 0.5
        liveness = 0.5

    out = _values([_E()], "soliton")
    assert out.shape == (1,) and np.isfinite(out[0])
```

Append to `tests/test_archive_window_render.py`:

```python
def test_soliton_sorts_unmeasured_entries_last():
    from ui.archive_window import _sort_key_soliton

    class _E:
        def __init__(self, s):
            self.soliton = s

    keys = [_sort_key_soliton((0, _E(s))) for s in (0.9, None, 0.1)]
    assert keys[0] < keys[2] < keys[1]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_map_view.py tests/test_archive_window_render.py -q`
Expected: FAIL — `"soliton" not in COLOR_MODES`, and an ImportError for
`_sort_key_soliton`.

- [ ] **Step 3: Add the sort mode**

In `ui/archive_window.py`, add at module level:

```python
def _sort_key_soliton(pair) -> float:
    """Descending soliton score, with unmeasured entries last.

    An entry admitted before the factors existed has no score; ranking it as
    zero would put it above genuinely bad ones, which is not what absent means.
    """
    value = getattr(pair[1], "soliton", None)
    return float("inf") if value is None else -float(value)
```

In `_sorted_entries` (line 603), add to the `keyfn` dict:

```python
                 "soliton": _sort_key_soliton,
```

In `_render_gallery` (line 614), change:

```python
        modes = ["novelty", "recency", "liveness"]
```

to:

```python
        modes = ["novelty", "recency", "liveness", "soliton"]
```

and the combo's labels to
`["Novelty", "Recency", "Liveness", "Soliton"]`.

- [ ] **Step 4: Add the colour mode**

In `services/map_view.py`, change line 28 to:

```python
COLOR_MODES = ("source", "novelty", "liveness", "soliton")
```

and add to `_values`, beside the `liveness` branch (line 95):

```python
    if mode == "soliton":
        # 0.0 for an unmeasured entry: the bottom of a colour ramp is honest
        # about "nothing to show", where a sort key must say "unknown".
        return np.array([getattr(e, "soliton", None) or 0.0 for e in entries],
                        dtype=np.float32)
```

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS.

- [ ] **Step 6: Manual check**

Launch the app, open Extras > Archive Browser on an archive that has run since
stage 1, and confirm: the Sort combo offers Soliton and reorders the gallery;
the map's Colour combo offers Soliton and the ramp changes. An archive with no
measured entries should show a uniformly dark ramp rather than an error.

- [ ] **Step 7: Commit**

```bash
git add ui/archive_window.py services/map_view.py tests/test_map_view.py tests/test_archive_window_render.py
git commit -m "feat: sort and colour the archive by soliton score

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Self-Review Notes

**Spec coverage.** Every spec section maps to a task: the measure → Tasks 1–2;
`Precomputed` and storage → Tasks 3–4; the calibration tool → Task 5; the
constants and their CLAUDE.md home → Task 6; the goal kind → Tasks 7–8; browser
and map → Task 9. The spec's out-of-scope list (admission, `vectors.npz`,
`Archive._phys`, any CLIP term) is untouched by every task.

**Two decisions this plan forced, both already folded back into the spec.**
The spec first stored two factors and derived the product; it now stores
**three**, adding `structure` — which its own product already included, was
computed in `tell`, and was not being persisted, so a two-factor column and the
expedition fitness would have ranked the same tile differently. And the
downsample target is **56, not 64**: `224 = 4 × 56` exactly, while a 64 target
covers only 192 of 224 pixels and discards the outer 16px of every edge.

**Not a placeholder.** The `MASS_FRACTION` and `MIN_SUPPORT` values in Task 1
are development starting points that Task 6 replaces with measured ones. That is
the sequencing the spec asks for.

**The gate.** Task 6 Step 7 is a hard stop. Stage 2 depends on stage 1 having
shown the substrate produces localized patterns at all.
