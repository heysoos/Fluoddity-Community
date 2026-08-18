# Layout Retarget and Restore Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a brain-layout change inside one archive cheap, make an archive
remember the brain it was last searched under, and split the archive browser's
preview gate from its adopt gate.

**Architecture:** An archive already holds every layout's entries in memory —
`load_from_store` reads every signature directory — so a layout change only
alters which rows are *native*. A new `Archive.retarget(layout)` re-points the
archive instead of tearing it down, `App._apply_brain_layout` uses it, and a
new persisted `ArchiveState.layout_signature` records and restores which brain
an archive was worked on with. Separately, `_handle_archive_preview` stops
using a tab selection as a proxy for "is a grid on screen".

**Tech Stack:** Python 3.12, NumPy, ModernGL, imgui_bundle, pytest.

## Global Constraints

- **Tests run on the venv interpreter, never bare `python`:**
  `.venv/Scripts/python.exe -m pytest -q`. Bare `python` is 3.10 with no pytest.
- **Comments state the rule, never the evidence.** No percentages, timings,
  dates, or before/after comparisons in comments, docstrings or tooltips.
  Measured facts live in `CLAUDE.md` or in
  `docs/superpowers/specs/2026-08-17-brain-layout-search-design.md`; everywhere
  else points at that home.
- **A tooltip is ONE sentence** naming what the control does.
- **This tree is shared with concurrent sessions.** `git add` by explicit
  filename only. Never `git add -A` and never `git add <directory>`.
- **Never `git push`.** Commit locally; the user pushes after verifying by hand.
- **Do not restructure `sim.py`** — it is user-owned.
- **`PERSISTED_FIELDS` in `state/archive_state.py` is an explicit allowlist.**
  A new field does not persist until it is named there.
- **`_novelty_clean` may only be set by `rescore_all()`** and cleared by `_add`
  and `_remove`. Nothing else touches it.
- **Do not open an `ArchiveStore` on the user's real archives** under
  `~/Documents/Fluoddity/archives`. It `mkdir`s and opens `index.jsonl` for
  append. All tests use `tmp_path`.

## Source of truth

The spec is
`docs/superpowers/specs/2026-08-17-brain-layout-search-design.md`.
This plan implements its sections **0**, **1** and **1b** only. Sections 2–7
(layout mutation, genome transfer, layout expeditions, the ledger, the settings
panel) are explicitly **out of scope** and must not be started here.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `services/archive.py` | gains `retarget()` — re-point without reload | 1 |
| `tests/test_archive_retarget.py` | *new* — retarget equals reload | 1 |
| `main.py` | `_apply_brain_layout` fast path | 2 |
| `tests/test_brain_layout_retarget_wiring.py` | *new* — no teardown on a layout change | 2 |
| `command_handler.py` | preview gate vs adopt gate | 3 |
| `ui/archive_window.py` | Live preview checkbox follows the preview gate | 3 |
| `tests/test_foreign_preview.py` | extended — the two gates | 3 |
| `state/archive_state.py` | `layout_signature` field + `PERSISTED_FIELDS` | 4 |
| `tests/test_archive_settings.py` | extended — the field round-trips | 4 |
| `main.py` | write the live layout on save; restore it on open/switch | 5 |
| `tests/test_archive_layout_restore.py` | *new* — reopen under the recorded brain | 5 |

---

### Task 1: `Archive.retarget(layout)`

**Files:**
- Modify: `services/archive.py` (add a method after `load_from_store`, and its
  helpers `_widen`/`_widths` are already present)
- Test: `tests/test_archive_retarget.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `Archive.retarget(layout: BrainLayout) -> None`. After the call,
  `archive.layout is layout`, `archive.signature == layout.signature()`,
  `archive.store` is the `ArchiveStore` owning that signature's directory, and
  `archive.native_rows()` selects the rows whose `layout` field equals that
  signature. Entries, embeddings, novelty and `_novelty_clean` are unchanged.
  Task 2 and Task 5 both call it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_archive_retarget.py`:

```python
"""Re-pointing an archive at another brain, without reloading it.

Every layout's entries are already in memory - load_from_store reads every
signature directory - so a layout change inside one archive changes only which
rows are NATIVE. Everything else in an archive is about PICTURES.
"""
from __future__ import annotations

import numpy as np

from services.archive import Archive, Candidate
from services.archive_io import ArchiveStore
from services.brains import BrainLayout, default_layout

DIM = 8
FOURIER = default_layout()
GABOR = BrainLayout("gabor", (12,), 168)
WIDE = BrainLayout("gabor", (25,), 350)


def _cand(vec, brain_len):
    e = np.asarray(vec, dtype=np.float32)
    e = e / max(float(np.linalg.norm(e)), 1e-8)
    return Candidate(embedding=e, brain=np.arange(brain_len, dtype=np.float32),
                     physics=np.zeros(8, np.float32), liveness=0.5,
                     spec="brain", goal="", run_id="r", gen=0, tile=0,
                     viable=True)


def _axis(i):
    v = np.zeros(DIM, np.float32)
    v[i] = 1.0
    return v


def _fill(root, layout, n, start_axis):
    store = ArchiveStore(root, layout)
    arc = Archive(store=store, layout=layout, dim=DIM)
    for j in range(n):
        arc.consider(_cand(_axis(start_axis + j), layout.length),
                     novelty=1.0, force=True)
    arc.maybe_flush(force=True)
    store.close()


def _mixed(tmp_path):
    """An archive holding both brains, opened under Fourier."""
    root = tmp_path / "mixed"
    _fill(root, FOURIER, 2, 0)
    _fill(root, GABOR, 3, 4)
    arc = Archive(store=ArchiveStore(root, FOURIER), layout=FOURIER, dim=DIM)
    arc.load_from_store()
    return root, arc


def _snapshot(arc):
    """Everything a retarget must leave alone."""
    return {
        "ids": [e.id for e in arc.entries],
        "layouts": [arc.layout_at(i) for i in range(len(arc.entries))],
        "emb": arc.embeddings.copy(),
        "novelty": np.array([e.novelty for e in arc.entries], dtype=np.float64),
        "brains": [arc.brain_at(i).copy() for i in range(len(arc.entries))],
    }


def test_retarget_moves_only_which_rows_are_native(tmp_path):
    _root, arc = _mixed(tmp_path)
    before = _snapshot(arc)
    assert len(arc.native_rows()) == 2

    arc.retarget(GABOR)

    assert arc.signature == GABOR.signature()
    assert len(arc.native_rows()) == 3
    after = _snapshot(arc)
    assert after["ids"] == before["ids"]
    assert after["layouts"] == before["layouts"]
    np.testing.assert_array_equal(after["emb"], before["emb"])
    np.testing.assert_array_equal(after["novelty"], before["novelty"])
    for a, b in zip(after["brains"], before["brains"]):
        np.testing.assert_array_equal(a, b)


def test_retarget_agrees_with_release_and_reload(tmp_path):
    """The load-bearing case: the fast path must be indistinguishable from the
    teardown it replaces."""
    root, arc = _mixed(tmp_path)
    arc.retarget(GABOR)
    fast = _snapshot(arc)
    fast_native = sorted(int(i) for i in arc.native_rows())

    slow_arc = Archive(store=ArchiveStore(root, GABOR), layout=GABOR, dim=DIM)
    slow_arc.load_from_store()
    slow = _snapshot(slow_arc)
    slow_native = sorted(int(i) for i in slow_arc.native_rows())

    assert sorted(fast["ids"]) == sorted(slow["ids"])
    assert fast_native == slow_native
    assert sorted(fast["layouts"]) == sorted(slow["layouts"])


def test_retarget_does_not_dirty_the_novelty_column(tmp_path):
    """Nothing was admitted and nothing removed, so every entry is still
    scored against exactly the set now held. Dirtying it would buy a full
    rescore per layout move for no information."""
    _root, arc = _mixed(tmp_path)
    arc.rescore_all()
    assert arc._novelty_clean is True
    arc.retarget(GABOR)
    assert arc._novelty_clean is True


def test_retarget_to_a_layout_the_archive_has_never_held(tmp_path):
    """A brand new brain has no directory yet, and admitting under it must
    still find a store to write through."""
    _root, arc = _mixed(tmp_path)
    arc.retarget(WIDE)
    assert arc.signature == WIDE.signature()
    assert len(arc.native_rows()) == 0
    assert arc.stores.get(WIDE.signature()) is not None


def test_retarget_to_a_wider_layout_pads_without_reinterpreting(tmp_path):
    """brain_at() trims each row back to its own width, so widening the pooled
    column must leave every stored genome reading exactly as before."""
    _root, arc = _mixed(tmp_path)
    before = [arc.brain_at(i).copy() for i in range(len(arc.entries))]
    arc.retarget(WIDE)
    for i, b in enumerate(before):
        np.testing.assert_array_equal(arc.brain_at(i), b)


def test_retarget_is_visible_to_the_thumbnail_loader(tmp_path):
    """gl_loader closes over the LIVE stores dict, so a store added by a
    retarget has to be reachable through it without rebuilding the cache."""
    _root, arc = _mixed(tmp_path)
    stores = arc.stores
    arc.retarget(WIDE)
    assert stores.get(WIDE.signature()) is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_archive_retarget.py -q`
Expected: FAIL, every test erroring with `AttributeError: 'Archive' object has no attribute 'retarget'`.

- [ ] **Step 3: Implement `retarget`**

In `services/archive.py`, add this method immediately after `load_from_store`
(before `_load_one`):

```python
    def retarget(self, layout) -> None:
        """Point the archive at a different brain WITHOUT reloading it.

        Every layout's entries are already here - load_from_store reads every
        signature directory - so this changes only which rows are NATIVE. The
        embeddings, the novelty column, the rejects ring and the thumbnails are
        about PICTURES, and a picture does not stop being one because a
        different brain is running.

        `_novelty_clean` is deliberately left alone: nothing was admitted and
        nothing removed, so every entry is still scored against exactly the set
        now held.

        A signature the archive has never held gets a store, because the next
        admission writes its index row and thumbnail through it. The width is
        recorded with setdefault - a directory that already loaded rows knows
        its own width, and the running layout must not overwrite it.
        """
        from services.archive_io import ArchiveStore

        sig = layout.signature()
        if self.store is not None:
            store = self._stores.get(sig)
            if store is None:
                store = ArchiveStore(self.store.base, signature=sig)
                self._stores[sig] = store
            self.store = store
        self.layout = layout
        self._widths.setdefault(sig, int(layout.length))
        self._widen(int(layout.length))
        self.revision += 1
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_archive_retarget.py -q`
Expected: PASS, 6 passed.

- [ ] **Step 5: Run the existing archive suite for regressions**

Run: `.venv/Scripts/python.exe -m pytest tests/test_mixed_archive.py tests/test_foreign_preview.py tests/test_archive_id_reuse.py -q`
Expected: PASS, no failures.

- [ ] **Step 6: Commit**

```bash
git add services/archive.py tests/test_archive_retarget.py
git commit -m "feat: Archive.retarget points at another brain without reloading"
```

---

### Task 2: `_apply_brain_layout` uses `retarget`

**Files:**
- Modify: `main.py` — the `_apply_brain_layout` body below the scales-only branch
- Test: `tests/test_brain_layout_retarget_wiring.py` (create)

**Interfaces:**
- Consumes: `Archive.retarget(layout)` from Task 1.
- Produces: no new API. `App._apply_brain_layout(layout, ui_state) -> bool`
  keeps its signature and its return meaning ("did anything change"). Task 5
  calls it.

**Behaviour deliberately preserved:** a layout change still STOPS the search
(`ast.running = False` and `auto_service.pause()`) and still resets the
optimizer, because the search space moved under it. Only the archive teardown
goes. A non-pausing variant belongs to spec section 4 and is out of scope here.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_brain_layout_retarget_wiring.py`:

```python
"""A layout change re-points the archive; it does not tear it down.

The archive is keyed by signature only in its DIRECTORY layout - every
layout's entries load into one Archive - so the switch that used to flush,
close, reload and refit now moves one pointer.
"""
from __future__ import annotations

import pytest

from services.brains import BrainLayout, default_layout
from state.archive_state import ArchiveState
from state.auto_tournament_state import AutoTournamentState
from state.preferences_state import PreferencesState

FOURIER = default_layout()
GABOR = BrainLayout("gabor", (12,), 168)


class _Bag:
    pass


class _Log(list):
    def note(self, what):
        self.append(what)


class _Sim:
    def __init__(self, log):
        self.brain_layout = FOURIER
        self._log = log

    def realloc_brain_buffers(self, layout):
        self.brain_layout = layout
        self._log.note("realloc")

    def apply_rule(self, rule):
        self._log.note("apply_rule")

    def set_brain_scales(self, layout):
        self._log.note("set_brain_scales")


class _Archive:
    def __init__(self, log):
        self._log = log
        self.layout = FOURIER

    def retarget(self, layout):
        self.layout = layout
        self._log.note("retarget")


class _Service:
    def __init__(self, log):
        self._log = log
        self.driver = None

    def set_layout(self, layout):
        self._log.note("set_layout")

    def pause(self):
        self._log.note("pause")


class _App:
    def __init__(self, log):
        self._log = log
        self.sim = _Sim(log)
        self.archive = _Archive(log)
        self.archive_store = _Bag()
        self.tournament_service = _Service(log)
        self.auto_service = _Service(log)
        self.imgep_driver = None
        self.command_handler = None
        self.goal_list = None
        self.thumb_cache = None

    def _refresh_driver_specs(self, layout, reset=False):
        self._log.note(f"refresh_specs(reset={bool(reset)})")

    def _save_archive_settings(self, ui_state):
        self._log.note("save_settings")

    def _release_archive(self, ui_state):
        self._log.note("RELEASE")

    def _build_archive_set(self, path):
        self._log.note("REBUILD")

    def _load_archive_settings(self, ui_state):
        self._log.note("load_settings")

    def _apply_brain_layout(self, layout, ui_state):
        from main import App

        return App._apply_brain_layout(self, layout, ui_state)


class _UI:
    def __init__(self):
        self.archive = ArchiveState()
        self.auto_tournament = AutoTournamentState()
        self.preferences = PreferencesState()


def test_a_layout_change_retargets_instead_of_rebuilding():
    log = _Log()
    app, ui = _App(log), _UI()
    assert app._apply_brain_layout(GABOR, ui) is True
    assert "retarget" in log
    assert "RELEASE" not in log, "the archive must not be released"
    assert "REBUILD" not in log, "the archive must not be reloaded from disk"


def test_the_search_is_still_stopped_and_the_optimizer_still_reset():
    """The search SPACE moved, so the covariance and the population are
    meaningless whatever the archive cost is."""
    log = _Log()
    app, ui = _App(log), _UI()
    ui.archive.running = True
    app._apply_brain_layout(GABOR, ui)
    assert ui.archive.running is False
    assert "pause" in log
    assert "refresh_specs(reset=True)" in log


def test_the_settings_are_written_after_the_switch_not_before():
    """They record the layout the archive is now ON, so a save taken before
    the switch would file the outgoing one."""
    log = _Log()
    app, ui = _App(log), _UI()
    app._apply_brain_layout(GABOR, ui)
    assert log.index("retarget") < log.index("save_settings")


def test_a_scales_only_change_still_takes_the_light_path():
    """What a z MEANS changed, not how wide it is, so there is nothing to
    retarget and nothing to reset."""
    log = _Log()
    app, ui = _App(log), _UI()
    scaled = BrainLayout(FOURIER.modality, FOURIER.shape, FOURIER.length,
                         scales=(("freq_scale", 4.0),))
    assert app._apply_brain_layout(scaled, ui) is True
    assert "set_brain_scales" in log
    assert "retarget" not in log
    assert "realloc" not in log


def test_an_unchanged_layout_does_nothing_at_all():
    log = _Log()
    app, ui = _App(log), _UI()
    assert app._apply_brain_layout(FOURIER, ui) is False
    assert log == []


def test_a_layout_change_with_no_archive_open_still_works():
    """The Brain window works before Explore has ever been opened."""
    log = _Log()
    app, ui = _App(log), _UI()
    app.archive = None
    app.archive_store = None
    assert app._apply_brain_layout(GABOR, ui) is True
    assert "realloc" in log
    assert "retarget" not in log
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_layout_retarget_wiring.py -q`
Expected: FAIL — `test_a_layout_change_retargets_instead_of_rebuilding` fails on
`"RELEASE" not in log`, and `test_the_settings_are_written_after_the_switch_not_before`
fails because `save_settings` currently runs first.

- [ ] **Step 3: Rewrite the body of `_apply_brain_layout`**

In `main.py`, replace everything from the `# Before the release` comment to the
end of the method (the block that calls `_save_archive_settings`,
`_release_archive`, `_build_archive_set` and `_load_archive_settings`) with:

```python
        # A layout change is an archive RE-POINT, not an archive switch: every
        # layout's entries are already in memory and only which of them are
        # native differs. The search still stops - its space just moved - but
        # the teardown that used to come with that does not.
        ui_state.archive.running = False
        if self.auto_service is not None:
            self.auto_service.pause()

        # The GPU side first: the per-particle readback buffer is sized by the
        # active length, and slot 0 is re-uploaded from whatever rule is live.
        self.sim.realloc_brain_buffers(layout)
        # The old genome's floats mean something else under a new layout, so it
        # is dropped - UNLESS a config naming this very layout was loaded in the
        # same frame, because the switch is that config's own doing and its
        # creature is what the user asked for.
        pending = None
        if self.command_handler is not None:
            pending = self.command_handler.take_pending_brain_rule(
                layout.signature())
        self.sim.apply_rule(pending)

        # The interactive tournament breeds genomes of the layout it is told
        # about; without this it keeps producing the old width and the tiles are
        # uploaded into slots that expect the new one.
        self.tournament_service.set_layout(layout)
        if self.auto_service is not None:
            self.auto_service.set_layout(layout)

        # The optimizer searches a different number of dimensions now, so its
        # covariance and population are meaningless. Reset rather than resize.
        self._refresh_driver_specs(layout, reset=True)

        if self.archive is not None:
            self.archive.retarget(layout)
        # AFTER the switch, so what is written names the layout the archive is
        # now on rather than the one it just left.
        self._save_archive_settings(ui_state)
        return True
```

Also update the method's docstring first paragraph, replacing the claim that a
layout change is an archive switch:

```python
    def _apply_brain_layout(self, layout, ui_state) -> bool:
        """Switch the brain layout. A hard reset of the SEARCH, never of the
        archive.

        One archive holds every layout, so this re-points it rather than
        switching to a sibling: the entries, their embeddings, their novelty
        and their thumbnails are all about pictures and survive the change.
        Only which rows are native moves. See
        docs/superpowers/specs/2026-08-17-brain-layout-search-design.md.
        """
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_brain_layout_retarget_wiring.py -q`
Expected: PASS, 6 passed.

- [ ] **Step 5: Run the surrounding suites for regressions**

Run: `.venv/Scripts/python.exe -m pytest tests/test_archive_switch.py tests/test_archive_settings.py tests/test_search_layout_wiring.py tests/test_layout_persistence.py -q`
Expected: PASS, no failures.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_brain_layout_retarget_wiring.py
git commit -m "perf: a brain layout change re-points the archive instead of reloading it"
```

---

### Task 3: Split the browser's preview gate from its adopt gate

**Files:**
- Modify: `command_handler.py` — `_handle_archive_preview`, the `tournament` gate
- Modify: `ui/archive_window.py` — `_render_live_preview_toggle`
- Test: `tests/test_foreign_preview.py` (extend)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: no new API. `_handle_archive_preview` now reads
  `ui_state.tournament.enabled`, so every caller and test must supply a
  `ui_state.tournament` carrying a `TournamentState`.

- [ ] **Step 1: Add `tournament` to the test UI helper**

In `tests/test_foreign_preview.py`, in `_ui()`, add the tournament state — the
handler now consults it, and a helper that omits it would raise:

```python
def _ui():
    from state.archive_state import ArchiveState
    from state.auto_tournament_state import AutoTournamentState
    from state.brain_state import BrainState
    from state.sim_state import SimState
    from state.tournament_state import TournamentState

    class _UI:
        pass

    u = _UI()
    u.sim = SimState()
    u.archive = ArchiveState()
    u.auto_tournament = AutoTournamentState()
    u.tournament = TournamentState()
    u.brain = BrainState()
    u.archive.live_preview = True
    return u
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_foreign_preview.py`:

```python
# ---- the preview gate and the adopt gate are not the same gate ----------

def test_a_manual_tournament_grid_is_not_written_into_by_a_hover(tmp_path):
    """Slot 0 under a grid is TILE 0, so a hover would change one square of a
    running grid - the defect _grid_owner() fixes for the Z and G keys. The
    Manual tab sets neither sub-mode, so a tab-selection gate misses it."""
    arc, _foreign, native = _mixed(tmp_path)
    sim = _Sim(FOURIER)
    h = _handler(arc, sim)
    ui = _ui()
    ui.tournament.enabled = True          # the window is open: a grid is up
    depth = len(h.rule_manager.stack)

    ui.archive.preview_entry_id = native
    h._handle_archive_preview(ui)

    assert len(h.rule_manager.stack) == depth, "a grid must not be previewed into"


def test_clicking_a_foreign_entry_adopts_its_brain_with_the_grid_up(tmp_path):
    """Adopting a LAYOUT is not a single-sim operation: the Brain window's
    modality combo already switches layout while a tournament runs."""
    arc, foreign, _native = _mixed(tmp_path)
    sim = _Sim(FOURIER)
    h = _handler(arc, sim)
    ui = _ui()
    ui.tournament.enabled = True
    ui.archive.enabled = True             # the Explore tab is selected
    seen = []
    h.apply_brain_layout = lambda lay, _u: seen.append(lay)

    ui.archive.load_entry_id = foreign
    h._handle_archive_preview(ui)

    assert [l.signature() for l in seen] == [GABOR.signature()]
    assert ui.brain.modality == "gabor"


def test_clicking_a_native_entry_under_a_grid_does_not_push_a_rule(tmp_path):
    """Nothing to adopt and nowhere to run it: the grid owns slot 0."""
    arc, _foreign, native = _mixed(tmp_path)
    sim = _Sim(FOURIER)
    h = _handler(arc, sim)
    ui = _ui()
    ui.tournament.enabled = True
    depth = len(h.rule_manager.stack)

    ui.archive.load_entry_id = native
    h._handle_archive_preview(ui)

    assert len(h.rule_manager.stack) == depth


def test_with_no_grid_the_explore_tab_no_longer_blocks_anything(tmp_path):
    """A tab selection is not a grid. With the tournament window shut the
    browser works whichever tab was last on screen."""
    arc, _foreign, native = _mixed(tmp_path)
    sim = _Sim(FOURIER)
    h = _handler(arc, sim)
    ui = _ui()
    ui.archive.enabled = True             # tab selected, window shut
    depth = len(h.rule_manager.stack)

    ui.archive.preview_entry_id = native
    h._handle_archive_preview(ui)

    assert len(h.rule_manager.stack) > depth
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_foreign_preview.py -q`
Expected: FAIL — `test_a_manual_tournament_grid_is_not_written_into_by_a_hover`
fails because a rule IS pushed, and both
`test_clicking_a_foreign_entry_adopts_its_brain_with_the_grid_up` and
`test_with_no_grid_the_explore_tab_no_longer_blocks_anything` fail because the
handler returns early.

- [ ] **Step 4: Split the gate in `command_handler.py`**

In `_handle_archive_preview`, replace the opening gate:

```python
        ast = ui_state.archive
        tournament = ast.enabled or ui_state.auto_tournament.enabled
        if self.archive is None or tournament:
            self._end_archive_preview(ui_state)
            ast.load_entry_id = -1
            return
```

with:

```python
        ast = ui_state.archive
        # A GRID, not a tab selection. What puts one on the canvas is the
        # tournament WINDOW being open, so a sub-mode flag misses the Manual
        # tab entirely - and slot 0 under a grid is tile 0, which is the split
        # _grid_owner() already makes for the Z and G keys.
        grid = bool(getattr(ui_state.tournament, "enabled", False))
        if self.archive is None:
            self._end_archive_preview(ui_state)
            ast.load_entry_id = -1
            return
        if grid:
            # The RULE half is what a grid owns. Adopting another brain's
            # LAYOUT is not a single-sim operation and stays available: the
            # Brain window's modality combo already does it under a grid.
            self._end_archive_preview(ui_state)
            entry_id, ast.load_entry_id = ast.load_entry_id, -1
            row = self._archive_index(entry_id) if entry_id >= 0 else None
            if row is not None and not self.archive.is_native(row):
                self._return_layout("gallery")
                self._adopt_foreign_entry(ui_state, row)
            return
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_foreign_preview.py -q`
Expected: PASS, all tests in the file.

- [ ] **Step 6: Follow the same gate in the checkbox**

In `ui/archive_window.py`, `_render_live_preview_toggle`, replace the body's
gate and messages:

```python
    def _render_live_preview_toggle(self, ast):
        """Run the hovered entry in the live sim, like hovering File > Load.

        Unavailable under a grid: the canvas is many simulations there, so
        there is no single sim for one entry to run in. Clicking an entry of
        another brain still switches to it.
        """
        busy = bool(self.state.tournament.enabled)
        imgui.begin_disabled(busy)
        _, ast.live_preview = imgui.checkbox("Live preview", ast.live_preview)
        imgui.end_disabled()
        if not busy and imgui.is_item_hovered():
            imgui.set_tooltip("Hover an entry to run it; click to keep it.")
        if busy:
            imgui.same_line()
            imgui.text_disabled("(no single sim under a grid - "
                                "clicking still switches brain)")
        elif ast.live_preview:
            imgui.same_line()
            imgui.text_colored(imgui.ImVec4(*_OK),
                               "hover to try, click to keep")
```

- [ ] **Step 7: Run the UI render suite**

Run: `.venv/Scripts/python.exe -m pytest tests/test_archive_window_render.py tests/test_label_widths.py -q`
Expected: PASS, no failures.

- [ ] **Step 8: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS. If `tests/test_menu_cross_brain_load.py` or
`tests/test_archive_preview.py` fail because their `_ui()` helper has no
`tournament` attribute, add `u.tournament = TournamentState()` to that helper
exactly as Step 1 did — the handler now reads it.

- [ ] **Step 9: Commit**

```bash
git add command_handler.py ui/archive_window.py tests/test_foreign_preview.py
git commit -m "fix: a grid owns slot 0, but clicking still adopts another brain"
```

---

### Task 4: An archive records the brain it was searched under

**Files:**
- Modify: `state/archive_state.py` — new field, and `PERSISTED_FIELDS`
- Modify: `main.py` — `_save_archive_settings`
- Test: `tests/test_archive_settings.py` (extend)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `ArchiveState.layout_signature: str = ""`, persisted. Written by
  `App._save_archive_settings` from `self.sim.brain_layout.signature()` at save
  time. `""` means "not recorded", which is every archive written before this.
  Task 5 consumes it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_archive_settings.py`:

```python
# ---- which brain this archive was being searched under -------------------

def test_the_layout_signature_travels_with_the_archive():
    """The app remembers which archive you were in; without this it does not
    remember which brain you were working on in it."""
    assert "layout_signature" in PERSISTED_FIELDS
    a = ArchiveState()
    a.layout_signature = "mlp-n16.8.8-a0.0.0"
    b = ArchiveState()
    b.apply_settings(a.to_settings())
    assert b.layout_signature == "mlp-n16.8.8-a0.0.0"


def test_an_archive_with_no_recorded_layout_reads_as_empty():
    """Every archive written before this one has no such key, and must open
    exactly as it always did."""
    s = ArchiveState()
    assert s.layout_signature == ""
    s.apply_settings({"alpha": 3.0})
    assert s.layout_signature == ""


def test_saving_records_the_LIVE_layout_not_a_stale_field(tmp_path, monkeypatch):
    """The field is a readout on the way out. Trusting whatever was last put
    in it files the layout the archive just left."""
    from main import App
    from services.archive_io import ArchiveStore
    from services.brains import BrainLayout, default_layout
    from state.preferences_state import PreferencesState

    gabor = BrainLayout("gabor", (12,), 168)

    class _Bag:
        pass

    app = _Bag()
    app.sim = _Bag()
    app.sim.brain_layout = gabor
    app.archive_store = ArchiveStore(tmp_path / "arc", default_layout())
    app.archive = None
    app.imgep_driver = None

    ui = _Bag()
    ui.archive = ArchiveState()
    ui.archive.layout_signature = "fourier-n10"      # stale
    ui.preferences = PreferencesState()

    App._save_archive_settings(app, ui)

    written = app.archive_store.load_settings()
    app.archive_store.close()
    assert written["layout_signature"] == "gabor-n12"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_archive_settings.py -q`
Expected: FAIL with `AttributeError: 'ArchiveState' object has no attribute 'layout_signature'`.

- [ ] **Step 3: Add the field**

In `state/archive_state.py`, add the field to `ArchiveState` beside the other
persisted view state (immediately after `live_preview`):

```python
    # The brain this archive was last SEARCHED under. A readout on the way out
    # and a command on the way in: the app remembers which archive you were in,
    # and without this it does not remember which brain you were working on in
    # it - so an archive of one modality reopens under another with no native
    # entries. Empty means not recorded, which is every archive written before
    # this one.
    layout_signature: str = ""
```

- [ ] **Step 4: Name it in the allowlist**

In `state/archive_state.py`, add it to the browser group of `PERSISTED_FIELDS`:

```python
    "show_browser", "sort_by", "pinned_only", "live_preview", "layout_signature",
```

- [ ] **Step 5: Write the live layout at save time**

In `main.py`, `_save_archive_settings`, set the field from the sim before the
settings dict is built:

```python
    def _save_archive_settings(self, ui_state):
        """Persist the Explore settings into the archive's own folder.

        Called before anything lets go of an archive - a switch, and quitting.
        """
        if self.archive_store is None:
            return
        # From the SIM, not from whatever was last put in the field: this is a
        # readout of the brain the archive is on, and a stale one files the
        # layout it just left.
        layout = getattr(getattr(self, "sim", None), "brain_layout", None)
        if layout is not None:
            ui_state.archive.layout_signature = layout.signature()
        settings = ui_state.archive.to_settings()
        # A change made after the last generation would otherwise never reach
        # the log, since the per-generation hook has stopped firing.
        self.archive_store.save_settings(settings)
        if self.archive is not None:
            gen = int(getattr(self.imgep_driver, "gen", 0) or 0)
            self.archive.record_settings(settings, gen)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_archive_settings.py -q`
Expected: PASS, no failures.

- [ ] **Step 7: Run the undo field audit**

`tests/test_undo_fields.py` derives its cases from the dataclass fields, so a
new field must be classified or the suite fails.

Run: `.venv/Scripts/python.exe -m pytest tests/test_undo_fields.py -q`
Expected: PASS. If it fails naming `layout_signature`, the field belongs in
that module's `NOT_UNDOABLE` map with the reason
`"names the brain an archive was searched under; the Brain window owns the layout"`.

- [ ] **Step 8: Commit**

```bash
git add state/archive_state.py main.py tests/test_archive_settings.py
git commit -m "feat: an archive records the brain it was last searched under"
```

---

### Task 5: Restore that layout when an archive is opened or switched to

**Files:**
- Modify: `main.py` — a new `_restore_archive_layout`, called from
  `_open_archive` and `_switch_archive`
- Test: `tests/test_archive_layout_restore.py` (create)

**Interfaces:**
- Consumes: `ArchiveState.layout_signature` (Task 4),
  `App._apply_brain_layout` (Task 2), `Archive.retarget` (Task 1).
- Produces: `App._restore_archive_layout(ui_state) -> bool` — did it switch.
  Called only from the two paths that OPEN an archive, never from
  `_apply_brain_layout`, which would loop.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_archive_layout_restore.py`:

```python
"""Reopening an archive under the brain it was being searched under.

The app remembers which archive you were in - archive_name is in preferences -
and used to forget which brain you were working on in it, so an archive whose
entries are all one modality reopened under another with no native rows at all.
"""
from __future__ import annotations

import pytest

from services.brains import BrainLayout, default_layout
from state.archive_state import ArchiveState
from state.brain_state import BrainState
from state.preferences_state import PreferencesState

FOURIER = default_layout()
GABOR = BrainLayout("gabor", (12,), 168)


class _Bag:
    pass


class _Sim:
    def __init__(self):
        self.brain_layout = FOURIER


class _App:
    def __init__(self):
        self.sim = _Sim()
        self.applied = []

    def _apply_brain_layout(self, layout, ui_state):
        self.sim.brain_layout = layout
        self.applied.append(layout)
        return True

    def _restore_archive_layout(self, ui_state):
        from main import App

        return App._restore_archive_layout(self, ui_state)


def _ui(signature):
    u = _Bag()
    u.archive = ArchiveState()
    u.archive.layout_signature = signature
    u.brain = BrainState()
    u.preferences = PreferencesState()
    return u


def test_an_archive_recorded_under_another_brain_switches_to_it():
    app, ui = _App(), _ui("gabor-n12")
    assert app._restore_archive_layout(ui) is True
    assert app.sim.brain_layout.signature() == "gabor-n12"


def test_the_brain_window_moves_too():
    """_handle_brain_layout applies ui_state.brain every frame, so a restore
    that moves only the sim is undone by the very next frame."""
    app, ui = _App(), _ui("gabor-n12")
    app._restore_archive_layout(ui)
    assert ui.brain.modality == "gabor"
    assert ui.brain.settings.get("filters") == 12


def test_an_archive_already_on_that_brain_does_nothing():
    app, ui = _App(), _ui("fourier-n10")
    assert app._restore_archive_layout(ui) is False
    assert app.applied == []


def test_an_archive_with_no_recorded_layout_is_left_alone():
    """Every archive written before this one. No migration runs."""
    app, ui = _App(), _ui("")
    assert app._restore_archive_layout(ui) is False
    assert app.applied == []


def test_a_signature_this_build_cannot_rebuild_keeps_the_current_brain():
    """Guessing a plausible layout of the wrong width is the one outcome worse
    than refusing."""
    app, ui = _App(), _ui("mlp-n99999999-aZZZ")
    assert app._restore_archive_layout(ui) is False
    assert app.sim.brain_layout is FOURIER
    assert "mlp-n99999999-aZZZ" in ui.archive.warning


def test_the_restore_is_not_wired_into_the_layout_change_itself():
    """_apply_brain_layout writes the settings that _restore reads. Calling
    one from the other would put the outgoing layout straight back."""
    import inspect

    from main import App

    src = inspect.getsource(App._apply_brain_layout)
    assert "_restore_archive_layout" not in src
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_archive_layout_restore.py -q`
Expected: FAIL with `AttributeError: type object 'App' has no attribute '_restore_archive_layout'`.

- [ ] **Step 3: Implement the restore**

In `main.py`, add this method immediately after `_load_archive_settings`.

`_put_brain_window` is a `@staticmethod` on `CommandHandler`, so it is called
on the CLASS rather than through `self.command_handler` — this runs during
`_open_archive`, which can happen before a handler is wired, and a `None`
handler would silently skip the half that makes the restore stick:

```python
    def _restore_archive_layout(self, ui_state) -> bool:
        """Put back the brain this archive was last searched under. -> did it.

        Called only from the paths that OPEN an archive. Never from
        _apply_brain_layout, which WRITES the setting this reads - calling one
        from the other would put the outgoing layout straight back.

        A signature this build cannot rebuild keeps the current brain and says
        so: a plausible layout of the wrong width is worse than refusing.
        """
        from command_handler import CommandHandler
        from services.brains import layout_from_signature

        ast = ui_state.archive
        sig = str(ast.layout_signature or "")
        live = getattr(getattr(self, "sim", None), "brain_layout", None)
        if not sig or live is None or sig == live.signature():
            return False
        layout = layout_from_signature(sig)
        if layout is None:
            ast.warning = (
                f"this archive was searched under {sig}, which this build "
                f"cannot rebuild; the brain is unchanged.")
            return False
        # The WINDOW as well as the sim: _handle_brain_layout applies whatever
        # it finds in ui_state.brain every frame, so a restore that moves only
        # the sim is undone by the next one.
        CommandHandler._put_brain_window(layout, ui_state)
        self._apply_brain_layout(layout, ui_state)
        return True
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_archive_layout_restore.py -q`
Expected: PASS, 6 passed.

- [ ] **Step 5: Wire it into the two paths that open an archive**

In `main.py`, `_open_archive`, after `self._load_archive_settings(ui_state)`:

```python
        self._load_archive_settings(ui_state)
        # After the settings, which is where layout_signature arrives.
        self._restore_archive_layout(ui_state)
```

In `main.py`, `_switch_archive`, after the `show_browser` juggling:

```python
        was_open = ast.show_browser
        self._load_archive_settings(ui_state)
        ast.show_browser = ast.show_browser or was_open
        self._restore_archive_layout(ui_state)
```

- [ ] **Step 6: Run the archive switch and open suites**

Run: `.venv/Scripts/python.exe -m pytest tests/test_archive_switch.py tests/test_archive_browser_menu.py tests/test_missing_archive_encoder.py -q`
Expected: PASS. `_FakeApp` in `test_archive_switch.py` calls `_switch_archive`
unbound, so it needs a `_restore_archive_layout` forwarding to the real method
exactly as its neighbours do:

```python
    def _restore_archive_layout(self, ui_state):
        from main import App

        return App._restore_archive_layout(self, ui_state)
```

and a `self.sim = _Bag()` with `self.sim.brain_layout = None` in `_FakeApp.__init__`,
so the restore's `live is None` branch returns False without touching anything.

- [ ] **Step 7: Run the full suite**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS, no failures.

- [ ] **Step 8: Commit**

```bash
git add main.py tests/test_archive_layout_restore.py tests/test_archive_switch.py
git commit -m "fix: an archive reopens under the brain it was searched under"
```

---

### Task 6: Correct the stale comment and record the tranche

**Files:**
- Modify: `main.py` — the wrong comment about where `settings.json` lives
- Modify: `CLAUDE.md` — one caveat for the retarget rule

**Interfaces:**
- Consumes: everything above. Produces nothing.

- [ ] **Step 1: Find the stale comment**

Run: `grep -n "settings.json lives inside the signature directory" main.py`
Expected: one hit, inside the block Task 2 already replaced. If Task 2 removed
it, this step is a no-op — confirm with the grep and move on.

Run: `grep -rn "signature directory" main.py services/archive_io.py`
Expected: any surviving claim that `settings.json` is per-signature is wrong —
`ArchiveStore.settings_path` is `self.base / "settings.json"`, at the archive
root, shared across layouts. Correct or delete each one.

- [ ] **Step 2: Add the caveat to CLAUDE.md**

In `CLAUDE.md`, under **The archive and admission**, add:

```markdown
- **A layout change RE-POINTS the archive; only an archive NAME change tears
  one down.** Every layout's entries are already in memory — `load_from_store`
  reads every signature directory — so `Archive.retarget()` moves `layout`,
  `store` and the width map and nothing else. The embeddings, the novelty
  column, the rejects ring, the projection and the thumbnail cache are about
  PICTURES and survive it; `gl_loader` closes over the LIVE `stores` dict, so a
  store added by a retarget is reachable without rebuilding the cache. It must
  not clear `_novelty_clean` — nothing was admitted or removed, so a rescore
  would buy nothing. `settings.json`, `goals.json`, `settings_history.jsonl`,
  `runs/` and `encoder.json` all live at the archive ROOT and are shared across
  layouts; only `index.jsonl`, `vectors.npz` and `thumbs/` are per-signature.

- **An archive records the brain it was last searched under, and nothing else
  did.** `archive_name` is in preferences, so the archive reopens; without
  `ArchiveState.layout_signature` the brain did not, and an archive whose
  entries are all one modality reopened under another with ZERO native rows —
  where Start bootstraps the wrong brain into it. Written from
  `sim.brain_layout` at save time rather than trusted from the field, which
  would file the layout the archive just left. Restored only by the paths that
  OPEN an archive: `_apply_brain_layout` WRITES what `_restore_archive_layout`
  reads, so wiring one into the other puts the outgoing layout straight back. A
  missing key means "leave the brain alone", so every archive written before
  this opens untouched.

- **The archive browser's PREVIEW and its ADOPT are two actions with two
  gates.** What owns slot 0 is a GRID, which is `ui_state.tournament.enabled` —
  the window being open — not a sub-mode flag; gating on the flag missed the
  Manual tab, where a hover wrote a rule into tile 0. Adopting another brain's
  LAYOUT is not a single-sim operation and stays available under a grid,
  because the Brain window's modality combo already does exactly that while a
  tournament runs. This is the same split `_grid_owner()` makes for Z and G.
```

- [ ] **Step 3: Run the full suite one last time**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: PASS, no failures.

- [ ] **Step 4: Commit**

```bash
git add main.py CLAUDE.md
git commit -m "docs: record the retarget, restore and browser-gate rules"
```

---

## Manual verification

The sim itself is verified by hand — see `docs/testing_checklist.md`. After
Task 6, run `python main.py` and confirm:

1. **Retarget is invisible except in speed.** Open Explore on an archive with a
   few thousand entries, change Modality in the Brain window, and the gallery
   should keep its thumbnails and its scroll position with no stall. The mixed
   note under the entry count should update its native figure.
2. **The Manual-tab hole is closed.** Open the Tournament window on Manual,
   open the Archive Browser, hover entries. No tile of the grid may change.
3. **Adopt still works under a grid.** With the Tournament window open on
   Explore and the search stopped, click an entry of another brain. The Brain
   window's Modality must follow it, and the notice should name the switch.
4. **The resume case.** Open an archive built under a non-default brain, quit,
   relaunch. The Brain window must come back on that archive's brain, and the
   Explore tab's regime must not read "bootstrap" with a full archive.

## Out of scope

Sections 2–7 of the spec — layout mutation operators, genome transfer, layout
expeditions, keep-or-revert, the ledger, and the `layout_search` settings panel.
None of them may be started in this plan. In particular, `_apply_brain_layout`
keeps STOPPING the search; the non-pausing variant a layout expedition needs
belongs to spec section 4.
