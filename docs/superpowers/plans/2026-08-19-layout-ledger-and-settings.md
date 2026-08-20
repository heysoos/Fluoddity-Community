# The layout ledger and its settings — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.

**Goal:** Sections 6 and 7 of
`docs/superpowers/specs/2026-08-17-brain-layout-search-design.md` — an
append-only record of every layout move at the archive root, and the settings
and panel that let a user turn layout search on and bound where it may roam.

**Architecture:** The ledger belongs to the ARCHIVE, not to the run, so the ban
set moves off the driver and is derived from the ledger. The settings follow
every other Explore setting: fields on `ArchiveState`, named in
`PERSISTED_FIELDS`, pushed at the driver by `_handle_explore`.

**Tech Stack:** NumPy, imgui_bundle, pytest.

## Global Constraints

- Tests run with `.venv/Scripts/python.exe -m pytest -q` (the worktree has no
  venv of its own; use the main checkout's interpreter).
- Comments, docstrings and tooltips state the RULE, never the evidence.
- A tooltip is ONE sentence naming what the control does.
- Stage commits by explicit filename. Never `git add -A`.
- Commit each task locally. Never push.
- `PERSISTED_FIELDS` is an explicit allowlist: a new field does not persist
  until it is named there.
- A widget's identity IS its label. No duplicate labels in one window.
- Layout search still ships OFF.

## Deviation from the spec, stated up front

Spec §7 asks for the three bounds to be "seeded from the LIVE layout rather
than from the hard maxima". That is NOT implemented as a seeding rule, because
every concrete seeding constant (live width + 4? live floats x 2?) would be
invented rather than measured, and an invented number buried in a default is
the thing this codebase's writing rules exist to stop. The spec's REASON — that
the user should not walk into the expensive end of the space without noticing —
is served instead by making the cost visible where the bound is set: the width
bound names the `MAX_MLP_WIDTH` scratch bucket it implies, and the float bound
shows what it allows against what is running. Bounds default to each modality's
own declared limits.

---

### Task 1: The store can hold a ledger

**Files:**
- Modify: `services/archive_io.py`
- Test: `tests/test_layout_ledger.py` (create)

**Interfaces:**
- Produces: `ArchiveStore.layouts_path`, `ArchiveStore.append_layout_move(row)`,
  `ArchiveStore.load_layout_moves() -> list[dict]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_layout_ledger.py`:

```python
"""Every layout move a run makes, at the archive root.

Beside settings_history.jsonl and for the same reason: a run that changes brain
silently redirects where its results are filed, and nothing else records that
it happened.
"""
from __future__ import annotations

from services.archive_io import ArchiveStore


def a_store(tmp_path):
    return ArchiveStore(tmp_path / "arc", signature="fourier-n10")


def test_the_ledger_sits_at_the_archive_root_not_under_a_layout(tmp_path):
    """One archive holds every brain, and a move is BETWEEN two of them - it
    belongs to neither signature directory."""
    s = a_store(tmp_path)
    assert s.layouts_path.parent == s.base
    assert s.layouts_path.name == "layouts.jsonl"


def test_an_archive_with_no_moves_reads_as_empty(tmp_path):
    assert a_store(tmp_path).load_layout_moves() == []


def test_a_row_survives_the_round_trip(tmp_path):
    s = a_store(tmp_path)
    row = {"parent": "fourier-n10", "child": "fourier-n11", "op": "grow",
           "gens": 50, "admitted": 3, "kept": True, "gen": 120}
    s.append_layout_move(row)
    assert s.load_layout_moves() == [row]


def test_rows_append_in_order(tmp_path):
    s = a_store(tmp_path)
    for i in range(3):
        s.append_layout_move({"op": "grow", "gen": i})
    assert [r["gen"] for r in s.load_layout_moves()] == [0, 1, 2]


def test_a_torn_line_costs_one_row_not_the_file(tmp_path):
    """Same discipline load_history already applies: a crash mid-write must not
    make the whole ledger unreadable."""
    s = a_store(tmp_path)
    s.append_layout_move({"op": "grow", "gen": 0})
    with open(s.layouts_path, "a", encoding="utf-8") as fh:
        fh.write('{"op": "shr')
    assert [r["gen"] for r in s.load_layout_moves()] == [0]


def test_a_disabled_store_writes_nothing_and_does_not_raise(tmp_path):
    s = a_store(tmp_path)
    s.enabled = False
    s.append_layout_move({"op": "grow"})
    assert not s.layouts_path.exists()
```

- [ ] **Step 2: Run and watch it fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_layout_ledger.py -q
```

Expected: `AttributeError: 'ArchiveStore' object has no attribute 'layouts_path'`.

- [ ] **Step 3: Implement**

In `services/archive_io.py`, beside `history_path`:

```python
    @property
    def layouts_path(self) -> Path:
        return self.base / "layouts.jsonl"
```

and beside `append_history` / `load_history`:

```python
    def append_layout_move(self, row: dict) -> None:
        """One layout move. Append-only, like the settings history.

        At the archive ROOT: a move is BETWEEN two signatures and belongs to
        neither of their directories.
        """
        if not self.enabled:
            return
        try:
            self.layouts_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.layouts_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        except (OSError, TypeError) as exc:
            print(f"[Archive] layout move not recorded ({exc})")

    def load_layout_moves(self) -> list[dict]:
        """-> every move row, oldest first. Empty for an archive with none."""
        rows: list[dict] = []
        try:
            text = self.layouts_path.read_text(encoding="utf-8")
        except OSError:
            return rows
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue        # a torn trailing line is one lost row
        return rows
```

- [ ] **Step 4: Run**

```bash
.venv/Scripts/python.exe -m pytest tests/test_layout_ledger.py -q
```

- [ ] **Step 5: Commit**

```bash
git add services/archive_io.py tests/test_layout_ledger.py
git commit -m "feat: an archive records every layout move it was searched across"
```

---

### Task 2: The ban set is the ledger, and the archive owns it

**Files:**
- Modify: `services/archive.py`
- Modify: `services/imgep_driver.py`
- Test: `tests/test_layout_ledger.py`, `tests/test_layout_verdict.py`

**Interfaces:**
- Produces: `Archive.record_layout_move(parent, child, op, gens, admitted,
  kept, gen)`, `Archive.layout_moves() -> list[dict]`,
  `Archive.reverted_pairs() -> set[tuple[str, str]]`.
- Removes: `ImgepDriver._reverted` as owned state — it reads
  `archive.reverted_pairs()` instead.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_layout_ledger.py`:

```python
# ---- the archive's view of it -------------------------------------------

def an_archive(tmp_path=None):
    from services.archive import Archive
    from services.archive_io import ArchiveStore

    store = (ArchiveStore(tmp_path / "arc", signature="fourier-n10")
             if tmp_path is not None else None)
    return Archive(store=store, dim=8)


def test_a_kept_move_bans_nothing():
    arc = an_archive()
    arc.record_layout_move("fourier-n10", "fourier-n11", "grow", 50, 3, True, 7)
    assert arc.reverted_pairs() == set()


def test_a_reverted_move_bans_its_pair():
    arc = an_archive()
    arc.record_layout_move("fourier-n10", "fourier-n11", "grow", 50, 0, False, 7)
    assert arc.reverted_pairs() == {("fourier-n10", "fourier-n11")}


def test_the_ban_is_directional():
    """Growing n10 -> n11 failing says nothing about shrinking n11 -> n10."""
    arc = an_archive()
    arc.record_layout_move("fourier-n10", "fourier-n11", "grow", 50, 0, False, 7)
    assert ("fourier-n11", "fourier-n10") not in arc.reverted_pairs()


def test_a_store_less_archive_still_bans():
    """Every test harness and every browse-only session runs without one."""
    arc = an_archive()
    arc.record_layout_move("a", "b", "grow", 1, 0, False, 0)
    assert arc.reverted_pairs() == {("a", "b")}


def test_a_ban_survives_reopening_the_archive(tmp_path):
    """The whole reason it is on disk: a move that produced nothing in this
    archive will not be re-proposed in the next session either."""
    arc = an_archive(tmp_path)
    arc.record_layout_move("fourier-n10", "fourier-n11", "grow", 50, 0, False, 7)
    again = an_archive(tmp_path)
    assert again.reverted_pairs() == {("fourier-n10", "fourier-n11")}


def test_the_row_records_what_the_move_was(tmp_path):
    arc = an_archive(tmp_path)
    arc.record_layout_move("fourier-n10", "fourier-n11", "grow", 50, 3, True, 7)
    row = arc.layout_moves()[-1]
    assert row["parent"] == "fourier-n10"
    assert row["child"] == "fourier-n11"
    assert row["op"] == "grow"
    assert row["gens"] == 50
    assert row["admitted"] == 3
    assert row["kept"] is True
    assert row["gen"] == 7
```

and append to `tests/test_layout_verdict.py`:

```python
def test_the_verdict_reaches_the_archives_ledger():
    d, arc, _ts = a_move(expedition_gens=2)
    mv = d.layout_move
    run_expedition(d, admit=False)
    row = arc.layout_moves()[-1]
    assert (row["parent"], row["child"]) == mv.pair
    assert row["kept"] is False


def test_a_ban_already_in_the_archive_is_honoured():
    """The ledger outlives the run, so a driver that has proposed nothing yet
    still knows what did not work."""
    from services.brains import default_layout
    from tests.test_layout_expedition import a_driver

    d, arc, _ts = a_driver()
    for child in ("fourier-n9", "fourier-n11"):
        arc.record_layout_move(default_layout().signature(), child,
                               "grow", 1, 0, False, 0)
    assert d.start_expedition() is True
    assert d.requested_layout is None      # every move from here is banned


def test_a_reset_does_not_forgive_a_reverted_move():
    """Reset clears the SEARCH, never the archive - and the ledger is a fact
    about the archive."""
    d, arc, _ts = a_move(expedition_gens=2)
    run_expedition(d, admit=False)
    d.reset()
    assert arc.reverted_pairs()
```

- [ ] **Step 2: Run and watch them fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_layout_ledger.py tests/test_layout_verdict.py -q
```

- [ ] **Step 3: Implement on the archive**

In `services/archive.py`, beside `record_settings`:

```python
    def record_layout_move(self, parent: str, child: str, op: str, gens: int,
                           admitted: int, kept: bool, gen: int) -> None:
        """File one layout move, and ban its pair if it was reverted.

        The ledger belongs to the ARCHIVE rather than to the run: a move that
        produced nothing here will not produce anything here next session
        either, and re-proposing it spends a cadence interval that an ordinary
        expedition would have used. Deleting layouts.jsonl is how a user
        forgives one.
        """
        row = {"ts": time.time(), "gen": int(gen), "parent": str(parent),
               "child": str(child), "op": str(op), "gens": int(gens),
               "admitted": int(admitted), "kept": bool(kept)}
        self._layout_moves.append(row)
        if self.store is not None:
            self.store.append_layout_move(row)

    def layout_moves(self) -> list[dict]:
        """-> every move filed against this archive, oldest first."""
        return list(self._layout_moves)

    def reverted_pairs(self) -> set[tuple[str, str]]:
        """-> (parent, child) pairs the archive has already refused.

        Directional: growing failing says nothing about shrinking back.
        """
        return {(str(r.get("parent", "")), str(r.get("child", "")))
                for r in self._layout_moves if not r.get("kept", True)}
```

In `Archive.__init__`, beside the other per-archive caches, seeded from the
store so a reopened archive knows what it already refused:

```python
        self._layout_moves: list[dict] = (
            store.load_layout_moves() if store is not None else [])
```

Place it AFTER `self.store` is assigned. `time` is already imported (used by
`record_settings`); check before adding it.

- [ ] **Step 4: Implement on the driver**

In `services/imgep_driver.py`, delete the `_reverted` field from `__init__`
along with its comment, and delete `self._reverted.clear()` from `reset()` —
leaving the two lines above it:

```python
        # A move that has not landed belongs to the run being abandoned. What
        # the ARCHIVE learned about which moves are dead ends is not the run's
        # to forget: reset() clears the search, and the ledger is a fact about
        # the archive.
        self.requested_layout = None
        self._pending = None
```

In `_propose_layout_move`, read the ban set from the archive:

```python
        mv = propose_layout_move(parent, self.layout_bounds, self.rng,
                                 banned=self.archive.reverted_pairs())
```

In `_finish_layout_move`, replace the `self._reverted.add(mv.pair)` line and
file the row in BOTH outcomes:

```python
        self.archive.record_layout_move(
            mv.parent.signature(), mv.child.signature(), mv.operator,
            int(self.expedition_gens), int(admitted), kept, int(self.gen))
        if kept:
            # It has native entries now, so ordinary expansion breeds from it
            # next generation with no special case anywhere.
            return
        self.requested_layout = mv.parent
```

(The `print` stays where it is, above this.)

- [ ] **Step 5: Run**

```bash
.venv/Scripts/python.exe -m pytest tests/test_layout_ledger.py tests/test_layout_verdict.py tests/test_layout_expedition.py tests/test_imgep_driver.py -q
```

- [ ] **Step 6: Commit**

```bash
git add services/archive.py services/imgep_driver.py tests/test_layout_ledger.py tests/test_layout_verdict.py
git commit -m "feat: the ledger is what bans a move, and it belongs to the archive"
```

---

### Task 3: Bounds from settings

**Files:**
- Modify: `services/brains/layout_moves.py`
- Test: `tests/test_layout_moves.py`

**Interfaces:**
- Produces: `parse_modalities(text) -> tuple[str, ...]` and
  `bounds_from(max_depth, max_width, max_floats, modalities, running) ->
  LayoutBounds`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_layout_moves.py`:

```python
# ---- bounds off a settings block ----------------------------------------

def test_a_modality_bound_is_one_string_not_four_booleans():
    """So a fifth modality needs no new field."""
    from services.brains.layout_moves import parse_modalities

    assert parse_modalities("mlp, gabor") == ("gabor", "mlp")


def test_an_unknown_key_is_dropped_rather_than_raising():
    """The bound is written to disk and outlives the build that wrote it."""
    from services.brains.layout_moves import parse_modalities

    assert parse_modalities("mlp,nonesuch") == ("mlp",)


def test_an_empty_string_names_no_modality():
    from services.brains.layout_moves import parse_modalities

    assert parse_modalities("") == ()
    assert parse_modalities(None) == ()


def test_zero_means_the_modality_decides():
    """0 is 'unbounded within the modality', which is what a fresh settings
    block holds - never a bound of zero, which forbids every layout."""
    from services.brains.layout_moves import bounds_from

    b = bounds_from(0, 0, 1024, "", FOURIER)
    assert b.max_depth is None
    assert b.max_width is None
    assert candidate_moves(FOURIER, b)


def test_a_bound_narrows_and_never_widens():
    from services.brains.layout_moves import bounds_from

    b = bounds_from(2, 6, 400, "", FOURIER)
    assert (b.max_depth, b.max_width, b.max_floats) == (2, 6, 400)


def test_the_running_modality_is_never_a_jump_target():
    """It is where the search already is; naming it would offer a move to the
    layout it is standing on."""
    from services.brains.layout_moves import bounds_from

    b = bounds_from(0, 0, 1024, "fourier,mlp", FOURIER)
    assert "fourier" not in b.modalities
    assert "mlp" in b.modalities
```

- [ ] **Step 2: Run and watch them fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_layout_moves.py -q
```

- [ ] **Step 3: Implement**

Append to `services/brains/layout_moves.py`:

```python
def parse_modalities(text) -> tuple[str, ...]:
    """A comma-separated bound into keys the registry knows. -> sorted keys.

    ONE string rather than one boolean per modality, so a fifth modality needs
    no new settings field and no second list of them. An unknown key is dropped
    rather than raising: the bound is written to disk and outlives the build
    that understood it.
    """
    if not text:
        return ()
    keys = {k.strip() for k in str(text).split(",")}
    return tuple(sorted(k for k in keys if k in REGISTRY))


def bounds_from(max_depth, max_width, max_floats, modalities,
                running) -> LayoutBounds:
    """Build the bounds a settings block describes.

    0 means "whatever the modality itself allows", which is what a fresh
    settings block holds - a literal bound of zero would forbid every layout.
    The RUNNING modality is dropped from the jump list: it is where the search
    already is.
    """
    return LayoutBounds(
        max_depth=int(max_depth) or None,
        max_width=int(max_width) or None,
        max_floats=min(int(max_floats), MAX_BRAIN_FLOATS),
        modalities=tuple(k for k in parse_modalities(modalities)
                         if k != running.modality),
    )
```

- [ ] **Step 4: Run**

```bash
.venv/Scripts/python.exe -m pytest tests/test_layout_moves.py -q
```

- [ ] **Step 5: Commit**

```bash
git add services/brains/layout_moves.py tests/test_layout_moves.py
git commit -m "feat: a settings block describes where a layout search may roam"
```

---

### Task 4: The settings, and the driver reads them

**Files:**
- Modify: `state/archive_state.py`
- Modify: `command_handler.py`
- Test: `tests/test_layout_search_settings.py` (create)

**Interfaces:**
- Produces: `ArchiveState.layout_search`, `layout_move_chance`,
  `layout_max_depth`, `layout_max_width`, `layout_max_floats`,
  `layout_modalities`, all in `PERSISTED_FIELDS`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_layout_search_settings.py`:

```python
"""The settings that turn layout search on and bound where it may roam."""
from __future__ import annotations

import pytest

from state.archive_state import PERSISTED_FIELDS, ArchiveState

FIELDS = ("layout_search", "layout_move_chance", "layout_max_depth",
          "layout_max_width", "layout_max_floats", "layout_modalities")


@pytest.mark.parametrize("name", FIELDS)
def test_every_bound_persists(name):
    """PERSISTED_FIELDS is an explicit allowlist: a field not named there
    silently does not survive reopening the archive."""
    assert name in PERSISTED_FIELDS


def test_layout_search_ships_off():
    """Opening the app must never start changing brain under anyone."""
    assert ArchiveState().layout_search is False


def test_the_bounds_default_to_the_modalitys_own_limits():
    """0 is 'whatever the modality allows'. A literal bound of zero would
    forbid every layout, so it can never be the default."""
    ast = ArchiveState()
    assert ast.layout_max_depth == 0
    assert ast.layout_max_width == 0


def test_the_float_ceiling_defaults_to_the_hard_one():
    from services.brains import MAX_BRAIN_FLOATS

    assert ArchiveState().layout_max_floats == MAX_BRAIN_FLOATS


def test_a_cross_modality_jump_is_opt_in():
    """It is a restart, so opting into one should be a decision."""
    assert ArchiveState().layout_modalities == ""


def test_the_settings_round_trip():
    ast = ArchiveState()
    ast.layout_search = True
    ast.layout_modalities = "mlp"
    ast.layout_move_chance = 0.4
    other = ArchiveState()
    other.apply_settings(ast.to_settings())
    assert other.layout_search is True
    assert other.layout_modalities == "mlp"
    assert other.layout_move_chance == pytest.approx(0.4)
```

Then append the driver-wiring tests to the same file:

```python
# ---- reaching the driver ------------------------------------------------

class _Drv:
    """Only what _handle_explore writes."""
    name = "imgep"

    def __init__(self):
        from services.brains.layout_moves import LayoutBounds

        self.layout_search = False
        self.layout_move_chance = 0.0
        self.layout_bounds = LayoutBounds()
        self.spec = None


def _push(ast, layout):
    """Run the settings half of _handle_explore against a bare driver."""
    from command_handler import CommandHandler

    drv = _Drv()
    CommandHandler._push_layout_search(drv, ast, layout)
    return drv


def test_the_checkbox_reaches_the_driver():
    from services.brains import default_layout

    ast = ArchiveState()
    ast.layout_search = True
    assert _push(ast, default_layout()).layout_search is True


def test_the_bounds_reach_the_driver():
    from services.brains import default_layout

    ast = ArchiveState()
    ast.layout_max_width = 12
    ast.layout_max_floats = 400
    ast.layout_modalities = "mlp"
    b = _push(ast, default_layout()).layout_bounds
    assert b.max_width == 12
    assert b.max_floats == 400
    assert b.modalities == ("mlp",)


def test_the_running_modality_is_dropped_from_the_jump_list():
    from services.brains import REGISTRY

    ast = ArchiveState()
    ast.layout_modalities = "mlp,gabor"
    b = _push(ast, REGISTRY["mlp"].layout_from_settings({})).layout_bounds
    assert b.modalities == ("gabor",)
```

- [ ] **Step 2: Run and watch them fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_layout_search_settings.py -q
```

- [ ] **Step 3: Add the fields**

In `state/archive_state.py`, after the `seed_ess_max` line:

```python
    # brain layout search (spec 7). OFF: opening the app must never start
    # changing brain under anyone.
    layout_search: bool = False
    layout_move_chance: float = 0.2
    # 0 is "whatever the modality itself allows" - a literal bound of zero
    # would forbid every layout, so it can never be the default. Only the float
    # budget has a real ceiling to sit under.
    layout_max_depth: int = 0
    layout_max_width: int = 0
    layout_max_floats: int = MAX_BRAIN_FLOATS
    # Comma-separated registry keys, so a fifth modality needs no new field.
    # Empty is the running modality alone: a jump is a restart, and opting into
    # one should be a decision.
    layout_modalities: str = ""
```

with `from services.brains import MAX_BRAIN_FLOATS` at the top of the module —
check whether `state/archive_state.py` already imports from `services` and
follow whatever it does (a module-level import is fine if nothing else there
defers).

Add to `PERSISTED_FIELDS`, after the expeditions block:

```python
    # brain layout search
    "layout_search", "layout_move_chance", "layout_max_depth",
    "layout_max_width", "layout_max_floats", "layout_modalities",
```

- [ ] **Step 4: Push them at the driver**

In `command_handler.py`, add the static helper beside `_handle_explore`:

```python
    @staticmethod
    def _push_layout_search(drv, ast, layout) -> None:
        """Point the driver's layout search at what the settings say.

        A static helper rather than three lines inline, because the bounds are
        the one Explore setting that is not a straight copy: 0 means "the
        modality decides" and the running modality is dropped from the jump
        list.
        """
        from services.brains.layout_moves import bounds_from

        drv.layout_search = bool(ast.layout_search)
        drv.layout_move_chance = float(ast.layout_move_chance)
        drv.layout_bounds = bounds_from(
            ast.layout_max_depth, ast.layout_max_width, ast.layout_max_floats,
            ast.layout_modalities, layout)
```

and call it from `_handle_explore`, immediately after the `setattr` loop that
pushes the other driver settings:

```python
        self._push_layout_search(drv, ast, self._brain_layout())
```

- [ ] **Step 5: Run**

```bash
.venv/Scripts/python.exe -m pytest tests/test_layout_search_settings.py tests/test_archive_settings.py -q
```

- [ ] **Step 6: Commit**

```bash
git add state/archive_state.py command_handler.py tests/test_layout_search_settings.py
git commit -m "feat: layout search is a setting the archive remembers"
```

---

### Task 5: The panel

**Files:**
- Modify: `ui/archive_window.py`
- Test: `tests/test_archive_window_render.py`

**Interfaces:**
- Produces: `ArchiveWindowMixin._render_layout_search(ast)` above the fold and
  `_render_layout_bounds(ast)` inside a new `Brain Layout` header.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_archive_window_render.py`:

```python
# ---- brain layout search -------------------------------------------------

def test_the_layout_search_toggle_is_drawn_with_every_section_shut(gui):
    """The primary control, so it may not live in a folded header - a folded
    header's body does not run at all."""
    h = Harness(driver=_TracingDriver(), archive=_FakeArchive(), goals=GoalList())

    def draw():
        _close_all_sections()
        h.render_explore_tab()

    assert "Search Brain Layouts Too" in checkbox_labels(draw)


def test_the_bounds_are_drawn_once_the_search_is_on(gui):
    h = Harness(driver=_TracingDriver(), archive=_FakeArchive(), goals=GoalList())
    h.state.archive.layout_search = True

    def draw():
        _open_all_sections()
        h.render_explore_tab()

    labels = slider_int_labels(draw)
    assert any("Max Brain Floats" in s for s in labels)
    assert any("Max Layers" in s for s in labels)


def test_a_modality_checkbox_comes_from_the_registry(gui):
    """A second hand-written list of modalities is the declared-but-never-read
    defect this codebase has shipped twice."""
    from services.brains import REGISTRY

    h = Harness(driver=_TracingDriver(), archive=_FakeArchive(), goals=GoalList())
    h.state.archive.layout_search = True

    def draw():
        _open_all_sections()
        h.render_explore_tab()

    labels = checkbox_labels(draw)
    for key in REGISTRY:
        assert key in labels, key


def test_the_bounds_are_not_drawn_while_the_search_is_off(gui):
    h = Harness(driver=_TracingDriver(), archive=_FakeArchive(), goals=GoalList())

    def draw():
        _open_all_sections()
        h.render_explore_tab()

    assert not any("Max Brain Floats" in s for s in slider_int_labels(draw))


def test_the_ledger_is_drawn_when_the_archive_holds_moves(gui):
    class _WithMoves(_FakeArchive):
        def layout_moves(self):
            return [{"parent": "fourier-n10", "child": "fourier-n11",
                     "op": "grow", "gens": 50, "admitted": 3, "kept": True,
                     "gen": 7}]

    h = Harness(driver=_TracingDriver(), archive=_WithMoves(), goals=GoalList())
    h.state.archive.layout_search = True

    def draw():
        _open_all_sections()
        h.render_explore_tab()

    assert any("fourier-n11" in s for s in text_wrapped_lines(draw))
```

`text_wrapped_lines` may not exist — check for an existing spy over
`imgui.text_wrapped` / `layout.text_disabled_wrapped` in that file and reuse
it; add one on the same pattern as `checkbox_labels` if there is none.

Also add `"Brain Layout"` to the module's `SECTIONS` tuple, so
`_open_all_sections`/`_close_all_sections` reach it.

- [ ] **Step 2: Run and watch them fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_archive_window_render.py -q -k layout
```

- [ ] **Step 3: Implement**

In `ui/archive_window.py`, call the toggle right after `_render_physics_search`
in `render_explore_tab`:

```python
        self._render_physics_search(ast)
        self._render_layout_search(ast)
        imgui.separator()
```

and add the header inside the folded block, after `Expeditions`:

```python
        if ast.layout_search and imgui.collapsing_header("Brain Layout"):
            self._render_layout_bounds(ast)
```

Add the two renderers beside `_render_expedition_settings`:

```python
    def _render_layout_search(self, ast):
        """The primary control, so it sits above the fold beside the physics
        one: a folded header's body does not run at all."""
        _, ast.layout_search = imgui.checkbox(
            "Search Brain Layouts Too", ast.layout_search)
        hints.tip("Let an expedition also grow, shrink or restructure the "
                  "brain it is searching.")
        if not ast.layout_search:
            layout.text_disabled_wrapped(
                "one brain shape - the Brain window fixes it")
            return
        layout.text_disabled_wrapped(
            "a layout that admits nothing is handed back and not tried again")

    def _render_layout_bounds(self, ast):
        """How far a layout search may roam, and what the width bound costs."""
        from services.brains import MAX_BRAIN_FLOATS, REGISTRY
        from services.brains.mlp import MAX_DEPTH, MAX_WIDTH, scratch_width

        _, ast.layout_move_chance = imgui.slider_float(
            "Layout Move Chance", ast.layout_move_chance, 0.0, 1.0)
        hints.tip("How often an expedition also moves the brain layout.")
        _, ast.layout_max_floats = imgui.slider_int(
            "Max Brain Floats", ast.layout_max_floats, 64, MAX_BRAIN_FLOATS)
        hints.tip("The largest brain the search may build.")
        _, ast.layout_max_depth = imgui.slider_int(
            "Max Layers", ast.layout_max_depth, 0, MAX_DEPTH)
        hints.tip("0 leaves the limit to the brain type.")
        _, ast.layout_max_width = imgui.slider_int(
            "Max Layer Width", ast.layout_max_width, 0, MAX_WIDTH)
        hints.tip("0 leaves the limit to the brain type.")
        # The width bound names the SCRATCH BUCKET it implies: MAX_MLP_WIDTH is
        # a compile-time define whose cost every brain in the build pays, not
        # only the wide stack, so crossing a bucket is a cost the user should
        # see at the moment they set it.
        w = int(ast.layout_max_width) or MAX_WIDTH
        layout.text_disabled_wrapped(
            f"a stack this wide compiles at scratch width "
            f"{scratch_width((w, 0, w, 0))}")

        layout.text_disabled_wrapped("brain types the search may jump to:")
        chosen = {k.strip() for k in str(ast.layout_modalities).split(",")}
        picked = []
        for key in REGISTRY:
            # Derived from the registry, never a hand-written list.
            on = imgui.checkbox(key, key in chosen)[1]
            if on:
                picked.append(key)
            layout.wrap_row()
        imgui.new_line()
        ast.layout_modalities = ",".join(picked)
        layout.text_disabled_wrapped(
            "a jump carries no genome - it is a restart in another brain type")
        self._render_layout_ledger(ast)

    def _render_layout_ledger(self, ast):
        """The walk this archive has already taken."""
        arc = self.archive_obj
        rows = arc.layout_moves() if arc is not None else []
        if not rows:
            return
        imgui.separator()
        for r in rows[-12:]:
            layout.text_disabled_wrapped(
                f"gen {r.get('gen', 0)}  {r.get('op', '')} "
                f"{r.get('parent', '')} -> {r.get('child', '')}  "
                f"{r.get('admitted', 0)} separated, "
                f"{'kept' if r.get('kept') else 'reverted'}")
```

`_FakeArchive` in the render test has no `layout_moves`, so use
`getattr(arc, "layout_moves", None)` rather than a bare call — check the fake
before choosing.

Check `layout.wrap_row()` exists and takes no argument; if the row helper has a
different name in `ui/layout.py`, use that.

- [ ] **Step 4: Run the render suite and the label-width guard**

```bash
.venv/Scripts/python.exe -m pytest tests/test_archive_window_render.py tests/test_label_widths.py -q
```

Expected: all pass. If a new label is wider than `WIDEST_LABEL`, shorten the
label rather than widening the guard.

- [ ] **Step 5: Commit**

```bash
git add ui/archive_window.py tests/test_archive_window_render.py
git commit -m "feat: a panel for layout search, its bounds and its walk"
```

---

### Task 6: Record the rules

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/superpowers/specs/2026-08-17-brain-layout-search-design.md`
- Modify: `README.md`

- [ ] **Step 1: Add the caveats**

Under **Brains and modalities**, after the verdict caveat:

- The ledger is at the archive ROOT, is what BANS a move, and belongs to the
  archive rather than the run — so Reset does not forgive one and deleting
  `layouts.jsonl` is how a user does.
- `layout_modalities` is one string, and its checkboxes come from `REGISTRY`.
- Bounds of 0 mean "the modality decides"; a literal 0 would forbid every
  layout.
- The width bound names its scratch bucket, because `MAX_MLP_WIDTH` is paid by
  every brain in the build.

In the spec, mark §7's seeding rule as deliberately not implemented, with the
reason.

In `README.md`, add the user-facing lines: where the checkbox is, what it does,
and that a layout that admits nothing is handed back.

- [ ] **Step 2: Full suite**

```bash
.venv/Scripts/python.exe -m pytest -q
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md README.md docs/superpowers/specs/2026-08-17-brain-layout-search-design.md
git commit -m "docs: the ledger bans a move, and the bounds name what they cost"
```
