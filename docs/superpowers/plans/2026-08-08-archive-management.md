# Archive Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user keep several named exploration archives under
`Documents/Fluoddity/archives/`, and switch, create, empty and delete them from
the Explore tab without restarting the app.

**Architecture:** A new stdlib-only module `services/archive_library.py` owns
every directory-level operation (naming, listing, creating, emptying,
deleting) and returns `Result` objects instead of raising. `utilities/paths.py`
gains `get_archives_root()` and a one-time migration of the old single
`archive/` folder. The Explore tab renders a dropdown and three modals, setting
one-shot flags; `CommandHandler` consumes them and calls a new
`App._switch_archive`, which flushes the outgoing archive, releases its GL
thumbnail textures, rebuilds the archive/goals/projection/cache set and
repoints every holder at it.

**Tech Stack:** Python 3.12, imgui_bundle (ImGui), NumPy, pytest. No new
dependencies.

Spec: `docs/superpowers/specs/2026-08-07-archive-management-design.md`

## Global Constraints

- **Run pytest with the venv interpreter.** Bare `python` on this machine is
  3.10 with no pytest. Every test command in this plan uses
  `.venv/Scripts/python.exe -m pytest`.
- **Windows platform.** Use forward slashes or `os.path`/`pathlib`; use `rm`,
  not `del`, in bash commands.
- **CLIP/evolution dependencies stay lazy.** `onnxruntime-directml`,
  `tokenizers` and `cmaes` must never be imported at startup.
  `services/archive_library.py` imports nothing beyond the standard library, so
  it may be imported anywhere, including at module scope in `ui/`.
- **`sim.py` is user-owned.** This feature does not touch it.
- **UI is passive.** UI code renders widgets and sets one-shot flags. It runs no
  filesystem logic and calls nothing on the orchestrator directly.
- **One-shot flags** are set by the UI and cleared by `CommandHandler`, never
  inside `UI.get_state()`.
- **A disk problem must never block evolution.** Every operation added here
  degrades to a warning in `ArchiveState.warning`, not an exception.
- Existing tests stay green. Run the full suite before each commit:
  `.venv/Scripts/python.exe -m pytest tests/ -q`

---

## File Structure

| File | Responsibility |
| --- | --- |
| `services/archive_library.py` | **new.** Directory-level archive operations. Knows about folders, not about `Archive`, `ImgepDriver` or GL. Pure stdlib, so it is exhaustively testable against `tmp_path`. |
| `utilities/paths.py` | `get_archives_root()`; one-time migration of the legacy `archive/` folder. `get_archive_dir()` is removed. |
| `state/preferences_state.py` | `archive_name` — which archive is active, persisted between sessions. |
| `state/archive_state.py` | The cached archive listing, the two modal text buffers and four new one-shot flags. |
| `ui/archive_window.py` | The archive row and three modals. Two pure helper functions (`archive_row_model`, `new_archive_status`) hold the display logic so it can be tested without ImGui. |
| `main.py` | `_build_archive_set(path)` and `_switch_archive(name, ui_state)`; `_ensure_archive_service` resolves by name. |
| `command_handler.py` | `_handle_archive_management(ui_state)` — consumes the four new flags. |
| `tests/test_archive_library.py` | **new.** The bulk of the coverage: names, listing, create/clear/delete, migration. |
| `tests/test_archive_switch.py` | **new.** The ordered switch sequence and the archive round-trip. |
| `tests/test_archive_window_render.py` | Additions: the row, the modals, the pure helpers. |

---

### Task 1: `archive_library` — names, resolution and listing

The naming and path-resolution half of the module. Nothing here writes to disk,
which makes it the right place to be paranoid: `resolve` is the guard standing
in front of the one operation in this feature that deletes a directory tree.

**Files:**
- Create: `services/archive_library.py`
- Test: `tests/test_archive_library.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `CLEARED_MARK: str` = `".cleared-"`
  - `MAX_NAME: int` = `64`
  - `@dataclass Result(ok: bool, name: str = "", message: str = "")`
  - `safe_name(raw) -> str`
  - `resolve(root: Path | str, name: str) -> Path` (raises `ValueError`)
  - `list_archives(root: Path | str) -> list[dict]` with keys
    `name: str`, `entries: int`, `mtime: float`, `size_mb: float`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_archive_library.py`:

```python
"""Directory-level archive operations.

Which archive is loaded is an experimental variable - novelty is measured
against the archive, so its contents ARE the search's memory. This module is
the only place that knows archives are folders, and it is pure stdlib, so all
of it runs against tmp_path with no GL, no CLIP and no window.
"""
import json

import pytest

from services.archive_library import (
    CLEARED_MARK,
    MAX_NAME,
    list_archives,
    resolve,
    safe_name,
)


def make_archive(root, name, entries=0, thumb_bytes=0):
    """A directory shaped exactly like the one ArchiveStore writes."""
    d = root / name
    (d / "thumbs").mkdir(parents=True)
    with open(d / "index.jsonl", "w", encoding="utf-8") as fh:
        for i in range(entries):
            fh.write(json.dumps({"id": i, "thumb": f"{i:06d}.jpg"}) + "\n")
    if thumb_bytes:
        (d / "thumbs" / "000000.jpg").write_bytes(b"x" * thumb_bytes)
    return d


# ---- safe_name ----------------------------------------------------------

@pytest.mark.parametrize("raw", [
    "../secrets", "..\\secrets", "/etc/passwd", "C:\\Windows",
    "a/b", "a\\b", "with:colon", "star*", "quest?", 'quo"te', "pipe|d",
    "lt<gt>",
])
def test_names_lose_everything_that_could_redirect_a_path(raw):
    out = safe_name(raw)
    assert not any(c in out for c in '\\/:*?"<>|')
    assert not out.startswith(".")
    assert not out.endswith((".", " "))


@pytest.mark.parametrize("raw", ["Dense Trails", "\u0433\u0443\u0441\u0442\u043e\u0439",
                                 "run-07", "a.b.c", "CAPS", "emoji \U0001f9ea"])
def test_ordinary_names_survive_unchanged(raw):
    """Case and unicode are not the problem; separators are."""
    assert safe_name(raw) == raw


@pytest.mark.parametrize("raw", ["", "   ", "...", "..", ".", "/", "\\", "*?",
                                 None, 5, b"bytes"])
def test_a_name_that_sanitises_to_nothing_is_refused(raw):
    assert safe_name(raw) == ""


def test_control_characters_are_dropped():
    assert safe_name("run\x00\x1b07") == "run07"


def test_long_names_are_truncated():
    assert len(safe_name("x" * 200)) == MAX_NAME


def test_truncation_never_leaves_a_trailing_dot():
    """Windows silently trims trailing dots from directory names, so a name
    ending in one would not round-trip: the folder created would not be the
    folder looked up."""
    out = safe_name("x" * (MAX_NAME - 1) + "..")
    assert not out.endswith(".")
    assert out == "x" * (MAX_NAME - 1)


# ---- resolve ------------------------------------------------------------

def test_resolve_puts_the_archive_directly_under_the_root(tmp_path):
    assert resolve(tmp_path, "run-07").parent == tmp_path.resolve()


@pytest.mark.parametrize("raw", ["../escape", "..\\escape", "/abs", "C:\\Windows"])
def test_no_sanitised_name_can_escape_the_root(tmp_path, raw):
    assert tmp_path.resolve() in resolve(tmp_path, raw).parents


def test_an_unusable_name_raises(tmp_path):
    with pytest.raises(ValueError):
        resolve(tmp_path, "   ")


def test_resolve_raises_if_sanitising_ever_stops_working(tmp_path, monkeypatch):
    """Unreachable through the UI, because safe_name runs first. It exists so a
    future caller that skips safe_name fails loudly instead of quietly deleting
    something outside archives/."""
    import services.archive_library as lib

    monkeypatch.setattr(lib, "safe_name", lambda raw: "../escape")
    with pytest.raises(ValueError):
        lib.resolve(tmp_path, "anything")


# ---- list_archives ------------------------------------------------------

def test_an_empty_root_lists_nothing(tmp_path):
    assert list_archives(tmp_path) == []


def test_a_missing_root_lists_nothing_rather_than_raising(tmp_path):
    assert list_archives(tmp_path / "nope") == []


def test_every_archive_is_listed_sorted_by_name(tmp_path):
    make_archive(tmp_path, "zeta")
    make_archive(tmp_path, "alpha")
    assert [a["name"] for a in list_archives(tmp_path)] == ["alpha", "zeta"]


def test_the_entry_count_is_the_index_line_count(tmp_path):
    """index.jsonl is the authority for which entries exist, so its line count
    is honest even when vectors.npz has been quarantined."""
    make_archive(tmp_path, "runs", entries=7)
    assert list_archives(tmp_path)[0]["entries"] == 7


def test_an_archive_with_no_index_yet_counts_zero(tmp_path):
    (tmp_path / "fresh" / "thumbs").mkdir(parents=True)
    assert list_archives(tmp_path)[0]["entries"] == 0


def test_cleared_snapshots_are_not_archives(tmp_path):
    make_archive(tmp_path, "runs", entries=2)
    make_archive(tmp_path, f"runs{CLEARED_MARK}1700000000", entries=2)
    assert [a["name"] for a in list_archives(tmp_path)] == ["runs"]


def test_loose_files_in_the_root_are_ignored(tmp_path):
    make_archive(tmp_path, "runs")
    (tmp_path / "readme.txt").write_text("hi", encoding="utf-8")
    assert [a["name"] for a in list_archives(tmp_path)] == ["runs"]


def test_the_size_counts_the_thumbnails(tmp_path):
    """size_mb is dominated by thumbs/, which is the number the user needs when
    deciding what to delete."""
    make_archive(tmp_path, "big", entries=1, thumb_bytes=2 * 1024 * 1024)
    assert list_archives(tmp_path)[0]["size_mb"] == pytest.approx(2.0, abs=0.1)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_archive_library.py -q
```

Expected: collection error, `ModuleNotFoundError: No module named 'services.archive_library'`.

- [ ] **Step 3: Write the implementation**

Create `services/archive_library.py`:

```python
"""Named archive directories on disk.

Which archive is loaded IS an experimental variable: novelty is measured
against the archive, so its contents are the search's memory. This module owns
the directory-level half of that - naming, listing, creating, emptying and
deleting - and nothing else.

It knows about folders, not about Archive, ImgepDriver or GL, which is what
makes it testable against tmp_path with no GPU and no CLIP. It also imports
nothing beyond the standard library, so it is safe to import at module scope
anywhere, including in ui/ where onnxruntime must never be pulled in.

Nothing here raises for a user error. Every operation returns a Result the
orchestrator can put in front of the user; a disk problem must never stop the
search.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

MAX_NAME = 64
CLEARED_MARK = ".cleared-"

# Windows-reserved punctuation plus both separators. Removing these is what
# guarantees a name cannot change WHICH directory a path refers to.
_FORBIDDEN = set('\\/:*?"<>|')


@dataclass
class Result:
    ok: bool
    name: str = ""
    message: str = ""      # user-facing; empty when ok


def safe_name(raw) -> str:
    """A directory name that cannot redirect a path.

    Case and unicode survive - 'Dense Trails' and 'gustoy' in Cyrillic are both
    fine names. What is removed is separators, drive-letter colons, the
    Windows-reserved punctuation, control characters, and leading or trailing
    dots and spaces: Windows silently trims trailing dots and spaces from
    directory names, so a name ending in one would not round-trip.

    -> "" for anything that sanitises away to nothing. Callers treat "" as a
    refusal rather than inventing a name.
    """
    if not isinstance(raw, str):
        return ""
    kept = [
        ch for ch in raw
        if ch not in _FORBIDDEN and unicodedata.category(ch)[0] != "C"
    ]
    name = "".join(kept).strip(" .")
    # Truncation can expose a new trailing dot, so strip again after cutting.
    return name[:MAX_NAME].strip(" .")


def resolve(root, name: str) -> Path:
    """The directory for `name` under `root`, guaranteed to be inside it.

    Raises ValueError if it is not. Unreachable through the UI, because
    safe_name has already run - but delete() rmtree's what this returns, so a
    future caller that skips safe_name must fail loudly rather than quietly
    outside archives/.
    """
    base = Path(root).resolve()
    safe = safe_name(name)
    if not safe:
        raise ValueError(f"not a usable archive name: {name!r}")
    path = (base / safe).resolve()
    if base not in path.parents:
        raise ValueError(f"{path} escapes {base}")
    return path


def list_archives(root) -> list[dict]:
    """One dict per archive directory, sorted by name.

    NOT cheap enough to call per frame: an archive with 4,800 thumbnails is
    4,800 stat calls, and ImGui re-renders every frame. The orchestrator calls
    this on demand and caches the result on ArchiveState.archive_list.
    """
    base = Path(root)
    out: list[dict] = []
    try:
        children = sorted(base.iterdir(), key=lambda p: p.name)
    except OSError:
        return out                      # no root yet is an empty list, not an error
    for child in children:
        try:
            if not child.is_dir() or CLEARED_MARK in child.name:
                continue
        except OSError:
            continue                    # one bad folder must not blank the dropdown
        out.append({
            "name": child.name,
            "entries": _count_entries(child),
            "mtime": _mtime(child),
            "size_mb": _size_mb(child),
        })
    return out


def _count_entries(path: Path) -> int:
    n = 0
    try:
        with open(path / "index.jsonl", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    n += 1
    except OSError:
        return 0
    return n


def _size_mb(path: Path) -> float:
    total = 0
    try:
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                continue
    except OSError:
        return 0.0
    return total / (1024.0 * 1024.0)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_archive_library.py -q
```

Expected: all pass (about 40 tests once parametrisation expands).

- [ ] **Step 5: Run the whole suite**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

Expected: no new failures.

- [ ] **Step 6: Commit**

```bash
git add services/archive_library.py tests/test_archive_library.py
git commit -m "feat: archive names, path resolution and listing"
```

---

### Task 2: `archive_library` — create, clear and delete

The three operations that change the disk. `clear` is a rename, not a delete:
an archive is hours of exploration, and "empty it" should be one rename away
from being undone.

**Files:**
- Modify: `services/archive_library.py` (append)
- Test: `tests/test_archive_library.py` (append)

**Interfaces:**
- Consumes: `Result`, `safe_name`, `resolve`, `list_archives`, `CLEARED_MARK`
  from Task 1.
- Produces:
  - `create(root, name) -> Result`
  - `clear(root, name) -> Result`
  - `delete(root, name) -> Result`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_archive_library.py`:

```python
from services.archive_library import clear, create, delete


# ---- create -------------------------------------------------------------

def test_create_makes_an_archive_shaped_directory(tmp_path):
    res = create(tmp_path, "run-07")
    assert res.ok and res.name == "run-07"
    assert (tmp_path / "run-07" / "thumbs").is_dir()


def test_create_sanitises_the_name_it_reports(tmp_path):
    res = create(tmp_path, "a/b")
    assert res.ok and res.name == "ab"
    assert (tmp_path / "ab").is_dir()


def test_create_refuses_an_unusable_name(tmp_path):
    """Refuse rather than auto-generating 'archive-1'. A name the user did not
    choose is a folder they will not find again."""
    res = create(tmp_path, "   ")
    assert not res.ok
    assert "no usable characters" in res.message


def test_create_refuses_a_duplicate_rather_than_suffixing(tmp_path):
    """Silently creating 'run-2' when the user typed 'run' puts entries
    somewhere they did not ask for."""
    create(tmp_path, "run")
    res = create(tmp_path, "run")
    assert not res.ok
    assert "already exists" in res.message
    assert not (tmp_path / "run-2").exists()


# ---- clear --------------------------------------------------------------

def test_clear_moves_the_contents_aside_and_leaves_an_empty_archive(tmp_path):
    make_archive(tmp_path, "runs", entries=5)
    res = clear(tmp_path, "runs")
    assert res.ok
    assert (tmp_path / "runs" / "thumbs").is_dir()
    assert list_archives(tmp_path)[0]["entries"] == 0


def test_clear_deletes_nothing(tmp_path):
    """The undo: the old contents are one rename away from being back."""
    make_archive(tmp_path, "runs", entries=5)
    clear(tmp_path, "runs")
    aside = [p for p in tmp_path.iterdir() if CLEARED_MARK in p.name]
    assert len(aside) == 1
    assert sum(1 for _ in open(aside[0] / "index.jsonl", encoding="utf-8")) == 5


def test_clearing_twice_in_one_second_does_not_collide(tmp_path):
    """The snapshot name is timestamped to the second; two clicks in the same
    second must not have the second rename fail onto the first."""
    make_archive(tmp_path, "runs", entries=1)
    assert clear(tmp_path, "runs").ok
    make_archive_index = tmp_path / "runs" / "index.jsonl"
    make_archive_index.write_text('{"id": 9}\n', encoding="utf-8")
    assert clear(tmp_path, "runs").ok
    assert len([p for p in tmp_path.iterdir() if CLEARED_MARK in p.name]) == 2


def test_clearing_something_that_is_gone_is_a_warning_not_a_crash(tmp_path):
    """The folder can be deleted outside the app between rendering the dropdown
    and clicking the button."""
    res = clear(tmp_path, "ghost")
    assert not res.ok
    assert "no longer on disk" in res.message


# ---- delete -------------------------------------------------------------

def test_delete_removes_the_whole_tree(tmp_path):
    make_archive(tmp_path, "keep")
    make_archive(tmp_path, "drop", entries=3, thumb_bytes=16)
    assert delete(tmp_path, "drop").ok
    assert not (tmp_path / "drop").exists()
    assert (tmp_path / "keep").is_dir()


def test_delete_refuses_the_last_archive(tmp_path):
    """There must always be something to load."""
    make_archive(tmp_path, "only", entries=3)
    res = delete(tmp_path, "only")
    assert not res.ok
    assert "only archive" in res.message
    assert (tmp_path / "only").is_dir()


def test_deleting_something_that_is_gone_is_a_warning_not_a_crash(tmp_path):
    make_archive(tmp_path, "real")
    make_archive(tmp_path, "other")
    res = delete(tmp_path, "ghost")
    assert not res.ok
    assert "no longer on disk" in res.message


def test_a_cleared_snapshot_does_not_count_as_an_archive_for_delete(tmp_path):
    """Snapshots are not archives, so having one does not make it safe to
    delete the only real archive."""
    make_archive(tmp_path, "only", entries=1)
    make_archive(tmp_path, f"only{CLEARED_MARK}1700000000", entries=1)
    assert not delete(tmp_path, "only").ok
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_archive_library.py -q
```

Expected: `ImportError: cannot import name 'clear' from 'services.archive_library'`.

- [ ] **Step 3: Write the implementation**

Append to `services/archive_library.py` (and add `import shutil` and
`import time` to the imports at the top):

```python
def create(root, name: str) -> Result:
    base = Path(root)
    safe = safe_name(name)
    if not safe:
        return Result(False, "", "That name has no usable characters.")
    path = resolve(base, safe)
    if path.exists():
        return Result(False, safe, f"An archive named '{safe}' already exists.")
    try:
        (path / "thumbs").mkdir(parents=True)
    except OSError as exc:
        return Result(False, safe, f"Could not create '{safe}': {exc}")
    return Result(True, safe)


def clear(root, name: str) -> Result:
    """Empty an archive WITHOUT deleting anything.

    The contents move to <name>.cleared-<unix-ts> and the live directory comes
    back empty. An archive is hours of exploration; 'empty it' should be one
    rename away from being undone, not an rmtree. The snapshots are never
    garbage-collected - they are the user's data, and the app does not get to
    decide when their undo expires.
    """
    base = Path(root)
    safe = safe_name(name)
    if not safe:
        return Result(False, "", "That name has no usable characters.")
    path = resolve(base, safe)
    if not path.is_dir():
        return Result(False, safe, f"'{safe}' is no longer on disk.")

    stamp = int(time.time())
    aside = base / f"{safe}{CLEARED_MARK}{stamp}"
    n = 1
    while aside.exists():
        # Two clicks inside one second must not rename onto each other.
        n += 1
        aside = base / f"{safe}{CLEARED_MARK}{stamp}-{n}"
    try:
        path.rename(aside)
        (path / "thumbs").mkdir(parents=True)
    except OSError as exc:
        return Result(False, safe, f"Could not empty '{safe}': {exc}")
    return Result(True, safe)


def delete(root, name: str) -> Result:
    """The only irreversible operation here. The UI puts it behind a
    typed-name confirmation."""
    base = Path(root)
    safe = safe_name(name)
    if not safe:
        return Result(False, "", "That name has no usable characters.")
    path = resolve(base, safe)
    if not path.is_dir():
        return Result(False, safe, f"'{safe}' is no longer on disk.")
    if len(list_archives(base)) <= 1:
        return Result(False, safe, "This is the only archive.")
    try:
        shutil.rmtree(path)
    except OSError as exc:
        return Result(False, safe, f"Could not delete '{safe}': {exc}")
    return Result(True, safe)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_archive_library.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add services/archive_library.py tests/test_archive_library.py
git commit -m "feat: create, clear and delete archives"
```

---

### Task 3: `get_archives_root()` and the legacy migration

Existing users have one archive at `Documents/Fluoddity/archive/`, possibly
hundreds of megabytes of it. It becomes `archives/default/` by a move, not a
copy.

**Files:**
- Modify: `utilities/paths.py:76-78` (replace `get_archive_dir`), `utilities/paths.py:104`
- Test: `tests/test_archive_library.py` (append)

**Interfaces:**
- Consumes: `create` from Task 2.
- Produces:
  - `utilities.paths.DEFAULT_ARCHIVE: str` = `"default"`
  - `utilities.paths.get_archives_root(user_dir=None) -> Path`
  - `utilities.paths.migrate_legacy_archive(user_dir=None) -> Path`
- Removes: `utilities.paths.get_archive_dir()`. Its only caller is
  `main.py:241`, rewritten in Task 6; until then `main.py` still imports it, so
  this task also updates that import to keep the app runnable (see Step 3).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_archive_library.py`:

```python
# ---- migration ----------------------------------------------------------

from utilities.paths import DEFAULT_ARCHIVE, get_archives_root, migrate_legacy_archive


def test_the_legacy_archive_becomes_default_with_its_contents(tmp_path):
    """A move, not a copy: archives run to hundreds of megabytes of
    thumbnails, and duplicating them to be tidy is worse than moving them."""
    legacy = tmp_path / "archive"
    (legacy / "thumbs").mkdir(parents=True)
    (legacy / "index.jsonl").write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")

    root = migrate_legacy_archive(tmp_path)

    assert root == tmp_path / "archives"
    assert not legacy.exists(), "moved, not copied"
    listed = list_archives(root)
    assert [a["name"] for a in listed] == [DEFAULT_ARCHIVE]
    assert listed[0]["entries"] == 2
    assert (root / DEFAULT_ARCHIVE / "thumbs").is_dir()


def test_a_first_launch_with_no_archive_at_all_gets_an_empty_default(tmp_path):
    root = migrate_legacy_archive(tmp_path)
    assert (root / DEFAULT_ARCHIVE / "thumbs").is_dir()
    assert list_archives(root)[0]["entries"] == 0


def test_an_existing_archives_root_is_never_touched(tmp_path):
    """A user who has already migrated and then restores an old backup must not
    have that backup swallowed into a directory that already has contents."""
    make_archive(tmp_path / "archives", "default", entries=4)
    legacy = tmp_path / "archive"
    (legacy / "thumbs").mkdir(parents=True)
    (legacy / "index.jsonl").write_text('{"id": 99}\n', encoding="utf-8")

    root = migrate_legacy_archive(tmp_path)

    assert legacy.is_dir(), "the legacy folder must be left alone"
    assert list_archives(root)[0]["entries"] == 4


def test_migration_is_idempotent(tmp_path):
    (tmp_path / "archive" / "thumbs").mkdir(parents=True)
    (tmp_path / "archive" / "index.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
    migrate_legacy_archive(tmp_path)
    migrate_legacy_archive(tmp_path)
    assert [a["name"] for a in list_archives(tmp_path / "archives")] == [DEFAULT_ARCHIVE]


def test_a_failed_move_still_leaves_a_usable_default(tmp_path, monkeypatch):
    """Migration failing must not stop the app from launching."""
    import utilities.paths as paths

    (tmp_path / "archive" / "thumbs").mkdir(parents=True)

    def boom(src, dst):
        raise OSError("locked")

    monkeypatch.setattr(paths.os, "replace", boom)
    root = migrate_legacy_archive(tmp_path)

    assert (root / DEFAULT_ARCHIVE / "thumbs").is_dir()
    assert (tmp_path / "archive").is_dir(), "the user's data is still there"


def test_the_root_hangs_off_the_user_data_directory(tmp_path, monkeypatch):
    import utilities.paths as paths

    monkeypatch.setattr(paths, "get_user_data_dir", lambda: tmp_path)
    assert get_archives_root() == tmp_path / "archives"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_archive_library.py -q -k "migrat or root or default"
```

Expected: `ImportError: cannot import name 'DEFAULT_ARCHIVE' from 'utilities.paths'`.

- [ ] **Step 3: Write the implementation**

In `utilities/paths.py`, replace `get_archive_dir` (lines 76-78) with:

```python
DEFAULT_ARCHIVE = "default"


def get_archives_root(user_dir=None) -> Path:
    """Root of the named exploration archives (Documents/Fluoddity/archives).

    Each subdirectory is one archive. There is no registry file: the filesystem
    is the list, so an archive folder copied in from elsewhere just appears.
    """
    base = Path(user_dir) if user_dir is not None else get_user_data_dir()
    return base / "archives"


def migrate_legacy_archive(user_dir=None) -> Path:
    """Move the pre-2026-08-08 single archive under archives/default.

    A move, not a copy: archives run to hundreds of megabytes of thumbnails.

    If archives/ already exists this does NOTHING - including to a legacy
    archive/ folder sitting beside it. A user who has already migrated and then
    restores an old backup should not have that backup swallowed into a
    directory that already has contents.

    -> the archives root, which is guaranteed to contain a 'default'.
    """
    base = Path(user_dir) if user_dir is not None else get_user_data_dir()
    root = get_archives_root(base)
    if root.exists():
        return root

    legacy = base / "archive"
    if legacy.is_dir():
        try:
            root.mkdir(parents=True, exist_ok=True)
            os.replace(legacy, root / DEFAULT_ARCHIVE)
            print(f"[Fluoddity] archive moved to {root / DEFAULT_ARCHIVE}")
            return root
        except OSError as exc:
            # Launching matters more than migrating. The old folder is left
            # exactly where it is, so nothing is lost.
            print(f"[Fluoddity] could not move the old archive ({exc}); "
                  f"starting an empty '{DEFAULT_ARCHIVE}'")

    (root / DEFAULT_ARCHIVE / "thumbs").mkdir(parents=True, exist_ok=True)
    return root
```

In `initialize_user_data()`, replace line 104:

```python
    get_archive_dir().mkdir(exist_ok=True)
```

with:

```python
    migrate_legacy_archive()
```

- [ ] **Step 4: Keep `main.py` runnable**

`main.py:239-241` still imports the removed function. Change those lines to
resolve the default archive; Task 6 replaces this block entirely.

```python
        from utilities.paths import DEFAULT_ARCHIVE, get_archives_root

        store = ArchiveStore(get_archives_root() / DEFAULT_ARCHIVE)
```

- [ ] **Step 5: Run the tests**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 6: Verify nothing else referenced the old function**

```bash
git grep -n "get_archive_dir" -- "*.py"
```

Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add utilities/paths.py main.py tests/test_archive_library.py
git commit -m "feat: archives root and one-time migration of the old archive"
```

---

### Task 4: State fields

Where the active archive name lives across launches, and the fields the UI and
`CommandHandler` communicate through.

**Files:**
- Modify: `state/preferences_state.py` (append a field before `save_preferences`)
- Modify: `state/archive_state.py`
- Test: `tests/test_archive_state.py` (append)

**Interfaces:**
- Consumes: `DEFAULT_ARCHIVE` from Task 3 (as a literal default, not an import —
  `state/` must not depend on `utilities.paths` for a dataclass default).
- Produces, on `ArchiveState`:
  `archive_name: str = "default"`, `archive_list: list`,
  `new_archive_name: str = ""`, `confirm_delete_text: str = ""`,
  `switch_archive_name: str = ""`, `new_archive_requested: bool`,
  `clear_archive_requested: bool`, `delete_archive_requested: bool`,
  `refresh_archive_list_requested: bool`.
  On `PreferencesState`: `archive_name: str = "default"`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_archive_state.py`:

```python
def test_the_active_archive_defaults_to_default():
    assert ArchiveState().archive_name == "default"


def test_the_cached_listing_starts_empty_and_is_per_instance():
    a, b = ArchiveState(), ArchiveState()
    a.archive_list.append({"name": "x"})
    assert b.archive_list == []


def test_the_management_one_shots_start_clear():
    ast = ArchiveState()
    assert ast.switch_archive_name == ""
    assert ast.new_archive_requested is False
    assert ast.clear_archive_requested is False
    assert ast.delete_archive_requested is False
    assert ast.refresh_archive_list_requested is False


def test_the_modal_buffers_start_empty():
    ast = ArchiveState()
    assert ast.new_archive_name == ""
    assert ast.confirm_delete_text == ""


def test_the_active_archive_survives_a_preferences_round_trip(tmp_path):
    """Without this, every launch reverts to 'default' regardless of what the
    user was working in."""
    from state.preferences_state import (PreferencesState, load_preferences,
                                         save_preferences)

    p = tmp_path / "prefs.json"
    prefs = PreferencesState()
    prefs.archive_name = "dense-trails"
    save_preferences(prefs, p)
    assert load_preferences(p).archive_name == "dense-trails"
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_archive_state.py -q
```

Expected: `AttributeError: 'ArchiveState' object has no attribute 'archive_name'`.

- [ ] **Step 3: Write the implementation**

In `state/archive_state.py`, change the import line to:

```python
from dataclasses import dataclass, field
```

Add, immediately after the `show_browser: bool = False` line:

```python
    # which archive is loaded (mirrors preferences.archive_name, for display)
    archive_name: str = "default"
    # cached services.archive_library.list_archives(); NOT recomputed per frame -
    # it stats every thumbnail, and ImGui re-renders this tab every frame.
    archive_list: list = field(default_factory=list)
```

Add, after the `new_goal_text: str = ""` line:

```python
    # modal text buffers
    new_archive_name: str = ""
    confirm_delete_text: str = ""
```

Add, at the end of the one-shot block:

```python
    switch_archive_name: str = ""
    new_archive_requested: bool = False
    clear_archive_requested: bool = False
    delete_archive_requested: bool = False
    refresh_archive_list_requested: bool = False
```

In `state/preferences_state.py`, add after the `parameter_locks_enabled` line:

```python
    # Exploration archive
    archive_name: str = "default"  # which Documents/Fluoddity/archives/<name> Explore mode loads
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_archive_state.py tests/test_speedmult_persistence.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add state/archive_state.py state/preferences_state.py tests/test_archive_state.py
git commit -m "feat: state for the active archive and its management flags"
```

---

### Task 5: The archive row and its modals

The UI half. The display logic lives in two pure functions so it can be
asserted directly; the render methods only draw what those return and set
one-shot flags.

**Files:**
- Modify: `ui/archive_window.py`
- Test: `tests/test_archive_window_render.py` (append)

**Interfaces:**
- Consumes: `ArchiveState` fields from Task 4; `safe_name` from Task 1.
- Produces (module-level in `ui/archive_window.py`, importable without a GL
  context):
  - `archive_row_model(ast) -> dict` with keys `names: list[str]`,
    `labels: list[str]`, `index: int`, `delete_enabled: bool`, `summary: str`
  - `new_archive_status(ast) -> dict` with keys `safe: str`, `taken: bool`,
    `can_create: bool`, `hint: str`
  - `ArchiveWindowMixin._render_archive_row(ast)`,
    `ArchiveWindowMixin._render_archive_modals(ast)`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_archive_window_render.py`:

```python
from ui.archive_window import archive_row_model, new_archive_status

TWO = [{"name": "default", "entries": 1511, "mtime": 0.0, "size_mb": 214.0},
       {"name": "dense-trails", "entries": 12, "mtime": 0.0, "size_mb": 1.5}]


def _ast(names=(), active="default", **over):
    ast = ArchiveState()
    ast.archive_list = list(names)
    ast.archive_name = active
    for k, v in over.items():
        setattr(ast, k, v)
    return ast


# ---- the row's logic ----------------------------------------------------

def test_the_row_shows_the_active_archive_before_any_listing_exists():
    """The listing is built by the orchestrator, so the very first frame has
    none. The row must still name the archive being written to."""
    m = archive_row_model(_ast())
    assert m["names"] == ["default"]
    assert m["index"] == 0


def test_the_row_selects_the_active_archive():
    m = archive_row_model(_ast(TWO, active="dense-trails"))
    assert m["names"][m["index"]] == "dense-trails"


def test_the_row_falls_back_to_the_first_entry_if_the_active_one_is_gone():
    """The folder can vanish outside the app. A stale index would silently
    point the combo at somebody else's archive."""
    m = archive_row_model(_ast(TWO, active="deleted-elsewhere"))
    assert m["index"] == 0


def test_each_label_carries_its_entry_count():
    m = archive_row_model(_ast(TWO))
    assert "1511" in m["labels"][0]


def test_the_summary_describes_the_active_archive_not_the_first():
    m = archive_row_model(_ast(TWO, active="dense-trails"))
    assert "12" in m["summary"] and "1.5" in m["summary"]
    assert m["entries"] == 12


def test_the_summary_says_so_when_nothing_is_loaded_yet():
    m = archive_row_model(_ast())
    assert m["summary"] == "not loaded yet"
    assert m["entries"] == 0


def test_delete_is_disabled_when_there_is_only_one_archive():
    """There must always be something to load."""
    assert archive_row_model(_ast(TWO))["delete_enabled"] is True
    assert archive_row_model(_ast(TWO[:1]))["delete_enabled"] is False
    assert archive_row_model(_ast())["delete_enabled"] is False


# ---- the New modal's logic ---------------------------------------------

def test_a_new_name_is_previewed_sanitised():
    st = new_archive_status(_ast(TWO, new_archive_name="a/b"))
    assert st["safe"] == "ab"
    assert "ab" in st["hint"]


def test_an_unchanged_name_needs_no_preview():
    st = new_archive_status(_ast(TWO, new_archive_name="run-07"))
    assert st["hint"] == ""
    assert st["can_create"] is True


def test_a_duplicate_name_cannot_be_created():
    st = new_archive_status(_ast(TWO, new_archive_name="default"))
    assert st["taken"] is True
    assert st["can_create"] is False
    assert "already exists" in st["hint"]


def test_an_empty_name_cannot_be_created():
    assert new_archive_status(_ast(TWO, new_archive_name="  "))["can_create"] is False
    assert new_archive_status(_ast(TWO))["can_create"] is False


# ---- the widgets actually render ---------------------------------------

def test_the_explore_tab_renders_the_row_with_a_listing(gui):
    h = Harness(driver=_FakeDriver(), archive=_FakeArchive())
    h.state.archive.archive_list = TWO
    assert frame(h.render_explore_tab) > host_only()


def test_the_explore_tab_renders_the_row_with_no_listing(gui):
    h = Harness(driver=_FakeDriver(), archive=_FakeArchive())
    assert frame(h.render_explore_tab) > host_only()


def test_the_modals_render_when_open(gui):
    """A modal left half-built corrupts the whole ImGui frame, so every window
    in the app disappears at once. Completing the frame proves the stack
    balances."""
    h = Harness(driver=_FakeDriver(), archive=_FakeArchive())
    h.state.archive.archive_list = TWO

    def run():
        imgui.open_popup("New archive")
        h._render_archive_modals(h.state.archive)

    assert frame(run) > host_only()
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/test_archive_window_render.py -q
```

Expected: `ImportError: cannot import name 'archive_row_model' from 'ui.archive_window'`.

- [ ] **Step 3: Write the implementation**

In `ui/archive_window.py`, add to the imports at the top:

```python
from services.archive_library import safe_name
```

(`services.archive_library` is pure stdlib, so this does not break the lazy-import rule.)

Add these two module-level functions after the tooltip constants:

```python
def archive_row_model(ast) -> dict:
    """Everything the archive row draws, as data.

    Pure, so the rules that matter - which archive is selected, whether Delete
    is allowed - are assertable without an ImGui context.
    """
    names = [a["name"] for a in ast.archive_list] or [ast.archive_name]
    try:
        index = names.index(ast.archive_name)
    except ValueError:
        # The active folder was deleted outside the app. Pointing the combo at
        # a stale index would name somebody else's archive.
        index = 0
    labels = []
    for a in ast.archive_list:
        labels.append(f"{a['name']}  ({a['entries']} entries)")
    if not labels:
        labels = list(names)

    active = next((a for a in ast.archive_list if a["name"] == ast.archive_name),
                  None)
    summary = (f"{active['entries']} entries \u00b7 {active['size_mb']:.1f} MB"
               if active else "not loaded yet")
    return {
        "names": names,
        "labels": labels,
        "index": index,
        "entries": active["entries"] if active else 0,
        "delete_enabled": len(ast.archive_list) > 1,
        "summary": summary,
    }


def new_archive_status(ast) -> dict:
    """Whether the typed name can become an archive, and what to say if not."""
    typed = ast.new_archive_name
    safe = safe_name(typed)
    taken = any(a["name"] == safe for a in ast.archive_list)
    if taken:
        hint = f"An archive named '{safe}' already exists."
    elif safe and safe != typed.strip():
        hint = f"will be saved as: {safe}"
    else:
        hint = ""
    return {"safe": safe, "taken": taken,
            "can_create": bool(safe) and not taken, "hint": hint}
```

Add the two render methods to `ArchiveWindowMixin`:

```python
    def _render_archive_row(self, ast):
        """Which archive is active is an experimental variable, so it sits at
        the top of the tab rather than in a menu."""
        m = archive_row_model(ast)
        imgui.set_next_item_width(240)
        changed, idx = imgui.combo("Archive", m["index"], m["labels"])
        if changed and 0 <= idx < len(m["names"]) and m["names"][idx] != ast.archive_name:
            ast.switch_archive_name = m["names"][idx]
        imgui.same_line()
        if imgui.button("New##archive"):
            ast.new_archive_name = ""
            imgui.open_popup("New archive")
        imgui.same_line()
        if imgui.button("Empty##archive"):
            imgui.open_popup("Empty archive")
        imgui.same_line()
        imgui.begin_disabled(not m["delete_enabled"])
        if imgui.button("Delete##archive"):
            ast.confirm_delete_text = ""
            imgui.open_popup("Delete archive")
        imgui.end_disabled()
        imgui.same_line()
        if imgui.button("Refresh##archive"):
            ast.refresh_archive_list_requested = True

        imgui.text_colored(imgui.ImVec4(*_DIM), m["summary"])
        self._render_archive_modals(ast)

    def _render_archive_modals(self, ast):
        flags = imgui.WindowFlags_.always_auto_resize
        if imgui.begin_popup_modal("New archive", flags=flags)[0]:
            imgui.text("Name for the new archive:")
            imgui.set_next_item_width(280)
            _, ast.new_archive_name = imgui.input_text(
                "##new_archive", ast.new_archive_name)
            st = new_archive_status(ast)
            if st["hint"]:
                colour = _WARN if st["taken"] else _DIM
                imgui.text_colored(imgui.ImVec4(*colour), st["hint"])
            imgui.separator()
            imgui.begin_disabled(not st["can_create"])
            if imgui.button("Create", imgui.ImVec2(120, 0)):
                ast.new_archive_requested = True
                imgui.close_current_popup()
            imgui.end_disabled()
            imgui.same_line()
            if imgui.button("Cancel##new_archive", imgui.ImVec2(120, 0)):
                imgui.close_current_popup()
            imgui.end_popup()

        if imgui.begin_popup_modal("Empty archive", flags=flags)[0]:
            m = archive_row_model(ast)
            imgui.text_wrapped(
                f"Empty '{ast.archive_name}'? Its {m['entries']} entries move "
                f"to '{ast.archive_name}.cleared-<time>' and can be restored "
                f"by renaming that folder back. Nothing is deleted.")
            imgui.separator()
            if imgui.button("Empty it", imgui.ImVec2(120, 0)):
                ast.clear_archive_requested = True
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel##empty_archive", imgui.ImVec2(120, 0)):
                imgui.close_current_popup()
            imgui.end_popup()

        if imgui.begin_popup_modal("Delete archive", flags=flags)[0]:
            imgui.text_wrapped(
                f"Permanently delete '{ast.archive_name}' - every entry, "
                f"thumbnail and goal in it. This cannot be undone.")
            imgui.text("Type the archive name to confirm:")
            imgui.set_next_item_width(280)
            _, ast.confirm_delete_text = imgui.input_text(
                "##confirm_delete", ast.confirm_delete_text)
            imgui.separator()
            imgui.begin_disabled(ast.confirm_delete_text != ast.archive_name)
            if imgui.button("Delete forever", imgui.ImVec2(140, 0)):
                ast.delete_archive_requested = True
                imgui.close_current_popup()
            imgui.end_disabled()
            imgui.same_line()
            if imgui.button("Cancel##delete_archive", imgui.ImVec2(120, 0)):
                imgui.close_current_popup()
            imgui.end_popup()
```

Call the row at the top of `render_explore_tab`, immediately after the
`archive_unavailable` early return and before `self._render_explore_status(ast)`:

```python
        self._render_archive_row(ast)
        imgui.separator()
        self._render_explore_status(ast)
```

Finally, name the archive in the browser's title so a gallery screenshot says
which archive it came from. In `render_archive_window`, replace:

```python
        expanded, opened = imgui.begin("Archive", True)
```

with:

```python
        # The name is in the title, not the body: a screenshot of the gallery
        # should say which archive it came from.
        expanded, opened = imgui.begin(f"Archive - {ast.archive_name}###archive",
                                       True)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_archive_window_render.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add ui/archive_window.py tests/test_archive_window_render.py
git commit -m "feat: archive picker row and its three modals"
```

---

### Task 6: `_switch_archive` in the orchestrator

The actual switch. Order matters: the outgoing archive must be flushed before
its store is closed, and the thumbnail cache must be released before the new
archive reuses entry ids 0, 1, 2 under different pictures.

**Files:**
- Modify: `main.py:220-273` (`_ensure_archive_service`), `main.py:432`
- Test: `tests/test_archive_switch.py` (create)

**Interfaces:**
- Consumes: `list_archives`, `resolve`, `safe_name`, `create` (Tasks 1-2);
  `get_archives_root`, `DEFAULT_ARCHIVE` (Task 3); `ArchiveState` and
  `PreferencesState` fields (Task 4).
- Produces:
  - `App._build_archive_set(path: Path) -> None`
  - `App._archive_path_for(name: str, ast) -> Path`
  - `App._switch_archive(name: str, ui_state) -> bool`
  - `App._ensure_archive_service(ui_state) -> bool` (was zero-argument)

- [ ] **Step 1: Write the failing test**

Create `tests/test_archive_switch.py`:

```python
"""Switching archives without restarting.

Order is the whole content of this operation. Flush before closing the store or
the trailing entries are lost - index.jsonl is flushed per entry, but
vectors.npz is only rewritten every 200 admissions. Release the thumbnail cache
before rebuilding or the new archive shows the old one's pictures, because
entry ids restart at 0 in every archive.
"""
import json

import numpy as np
import pytest

from state.archive_state import ArchiveState
from state.preferences_state import PreferencesState


class _Log(list):
    def note(self, what):
        self.append(what)


class _FakeService:
    def __init__(self, log):
        self._log = log

    def pause(self):
        self._log.note("pause")


class _FakeDriver:
    def __init__(self, log):
        self._log = log
        self.archive = None
        self.goals = None

    def end_expedition(self):
        self._log.note("end_expedition")


class _FakeArchive:
    def __init__(self, log):
        self._log = log

    def maybe_flush(self, force=False):
        self._log.note(f"flush(force={force})")


class _FakeGoals:
    def __init__(self, log):
        self._log = log

    def save(self):
        self._log.note("save_goals")


class _FakeStore:
    def __init__(self, log):
        self._log = log

    def close(self):
        self._log.note("close_store")


class _FakeCache:
    def __init__(self, log):
        self._log = log

    def release(self):
        self._log.note("release_thumbs")


class _Bag:
    """Stands in for ui and command_handler: anything can be set on it."""


class _FakeApp:
    def __init__(self, log):
        self.ctx = None
        self.auto_service = _FakeService(log)
        self.imgep_driver = _FakeDriver(log)
        self.archive = _FakeArchive(log)
        self.goal_list = _FakeGoals(log)
        self.archive_store = _FakeStore(log)
        self.thumb_cache = _FakeCache(log)
        self.archive_projection = None
        self._last_projection_size = 0
        self.ui = _Bag()
        self.command_handler = _Bag()


class _UIState:
    def __init__(self):
        self.archive = ArchiveState()
        self.preferences = PreferencesState()


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """A Documents/Fluoddity with two archives in it."""
    import utilities.paths as paths

    monkeypatch.setattr(paths, "get_user_data_dir", lambda: tmp_path)
    root = tmp_path / "archives"
    for name, n in (("default", 3), ("dense-trails", 0)):
        (root / name / "thumbs").mkdir(parents=True)
        with open(root / name / "index.jsonl", "w", encoding="utf-8") as fh:
            for i in range(n):
                fh.write(json.dumps({"id": i}) + "\n")
    return root


def switch(app, name, ui_state):
    from main import App
    return App._switch_archive(app, name, ui_state)


def test_the_switch_happens_in_the_order_that_keeps_data(roots):
    log = _Log()
    app, ui = _FakeApp(log), _UIState()

    assert switch(app, "dense-trails", ui) is True

    assert log == ["pause", "end_expedition", "flush(force=True)", "save_goals",
                   "close_store", "release_thumbs"]


def test_the_switch_repoints_every_holder(roots):
    log = _Log()
    app, ui = _FakeApp(log), _UIState()
    driver = app.imgep_driver

    switch(app, "dense-trails", ui)

    assert app.archive is driver.archive
    assert app.goal_list is driver.goals
    assert app.ui.archive_obj is app.archive
    assert app.ui.archive_goals is app.goal_list
    assert app.ui.archive_projection is app.archive_projection
    assert app.ui.thumb_cache is app.thumb_cache
    assert app.command_handler.archive is app.archive
    assert app.command_handler.goal_list is app.goal_list
    assert app.command_handler.archive_projection is app.archive_projection


def test_the_driver_instance_survives_the_switch(roots):
    """The user's exploration settings live on the driver and are not part of
    the archive."""
    log = _Log()
    app, ui = _FakeApp(log), _UIState()
    driver = app.imgep_driver
    switch(app, "dense-trails", ui)
    assert app.imgep_driver is driver


def test_the_switch_persists_the_choice_and_refreshes_the_listing(roots):
    log = _Log()
    app, ui = _FakeApp(log), _UIState()

    switch(app, "dense-trails", ui)

    assert ui.preferences.archive_name == "dense-trails"
    assert ui.archive.archive_name == "dense-trails"
    assert [a["name"] for a in ui.archive.archive_list] == ["default", "dense-trails"]


def test_the_switch_does_not_resume_the_search(roots):
    """The user pressed a management button, not Start. Resuming for them would
    begin writing into an archive they may only have wanted to look at."""
    log = _Log()
    app, ui = _FakeApp(log), _UIState()
    ui.archive.running = True
    switch(app, "dense-trails", ui)
    assert ui.archive.running is False


def test_the_switch_clears_the_selected_entry(roots):
    """Entry ids restart per archive, so a carried-over selection would name a
    different creature."""
    log = _Log()
    app, ui = _FakeApp(log), _UIState()
    ui.archive.selected_entry_id = 42
    switch(app, "dense-trails", ui)
    assert ui.archive.selected_entry_id == -1


def test_switching_to_a_folder_that_vanished_warns_and_changes_nothing(roots):
    log = _Log()
    app, ui = _FakeApp(log), _UIState()
    before = app.archive

    assert switch(app, "ghost", ui) is False

    assert "no longer on disk" in ui.archive.warning
    assert app.archive is before
    assert log == []


def test_switching_loads_the_target_archives_entries(roots, tmp_path):
    """The point of the whole feature: A's contents come back when you go back
    to A."""
    from services.archive import Archive
    from services.archive_io import ArchiveStore

    # Give 'dense-trails' two real, loadable entries.
    store = ArchiveStore(roots / "dense-trails")
    arc = Archive(store=store, seed_n=0, liveness_min=0.0)
    _admit(arc, 2)
    arc.maybe_flush(force=True)
    store.close()

    log = _Log()
    app, ui = _FakeApp(log), _UIState()
    switch(app, "dense-trails", ui)

    assert len(app.archive) == 2


def _admit(arc, n, dim=512):
    from services.archive import Candidate

    for i in range(n):
        e = np.zeros(dim, dtype=np.float32)
        e[i] = 1.0
        arc.consider(
            Candidate(brain=np.zeros((10, 8), dtype=np.float32),
                      physics=np.zeros(8, dtype=np.float32),
                      embedding=e, liveness=1.0, spec="brain"),
            novelty=1.0)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/test_archive_switch.py -q
```

Expected: `AttributeError: type object 'App' has no attribute '_switch_archive'`.

- [ ] **Step 3: Rewrite `_ensure_archive_service` and add the new methods**

Replace `main.py:220-273` in full:

```python
    def _build_archive_set(self, path):
        """(Re)build everything that hangs off ONE archive directory, and point
        every holder at it.

        The ImgepDriver instance is deliberately kept: sigma, alpha, the
        expedition cadence and the rest are the user's settings, not the
        archive's.
        """
        from services.archive import Archive
        from services.archive_io import ArchiveStore
        from services.archive_projection import Projection
        from services.goal_source import GoalList
        from services.thumb_cache import ThumbCache, gl_loader

        store = ArchiveStore(path)
        archive = Archive(store=store)
        loaded, dropped = archive.load_from_store()
        print(f"[archive] {path.name}: loaded {loaded} entries ({dropped} dropped)")

        goals = GoalList(store=store)
        goals.load()

        self.archive_store = store
        self.archive = archive
        self.goal_list = goals
        self.archive_projection = Projection()
        self.archive_projection.fit(archive.embeddings)
        self._last_projection_size = len(archive)
        self.thumb_cache = ThumbCache(gl_loader(self.ctx, store), capacity=256)

        if self.imgep_driver is not None:
            self.imgep_driver.archive = archive
            self.imgep_driver.goals = goals
        self.ui.archive_obj = archive
        self.ui.archive_goals = goals
        self.ui.archive_projection = self.archive_projection
        self.ui.thumb_cache = self.thumb_cache
        self.command_handler.archive = archive
        self.command_handler.goal_list = goals
        self.command_handler.archive_projection = self.archive_projection

    def _archive_path_for(self, name, ast):
        """The directory for `name`, falling back to 'default' when it is gone.

        A missing folder means the user deleted it outside the app or moved
        their Documents. Substituting silently would have them exploring into a
        different archive than the one the UI says is loaded.
        """
        from services.archive_library import create, resolve, safe_name
        from utilities.paths import DEFAULT_ARCHIVE, get_archives_root

        root = get_archives_root()
        safe = safe_name(name) or DEFAULT_ARCHIVE
        path = resolve(root, safe)
        if path.is_dir():
            return path
        if safe != DEFAULT_ARCHIVE:
            ast.warning = (f"Archive '{safe}' is gone; "
                           f"loaded '{DEFAULT_ARCHIVE}' instead.")
        create(root, DEFAULT_ARCHIVE)
        return resolve(root, DEFAULT_ARCHIVE)

    def _switch_archive(self, name, ui_state):
        """Point the search at a different archive directory. -> success.

        The order below is the whole content of this method. Flush before
        closing the store, or the entries since the last 200-admission vector
        flush are lost. Release the thumbnail cache before rebuilding, or the
        new archive shows the old one's pictures - entry ids restart at 0 in
        every archive.
        """
        from services.archive_library import list_archives, resolve, safe_name
        from utilities.paths import get_archives_root

        ast = ui_state.archive
        root = get_archives_root()
        safe = safe_name(name)
        if not safe:
            ast.warning = "That name has no usable characters."
            return False
        path = resolve(root, safe)
        if not path.is_dir():
            ast.warning = f"'{safe}' is no longer on disk."
            ast.archive_list = list_archives(root)
            return False

        if self.auto_service is not None:
            self.auto_service.pause()
        ast.running = False
        if self.imgep_driver is not None:
            # The CMA-ES mean was seeded from a parent in the OUTGOING archive.
            self.imgep_driver.end_expedition()
        if self.archive is not None:
            self.archive.maybe_flush(force=True)
        if self.goal_list is not None:
            self.goal_list.save()
        if self.archive_store is not None:
            self.archive_store.close()
        if self.thumb_cache is not None:
            self.thumb_cache.release()

        self._build_archive_set(path)

        ui_state.preferences.archive_name = safe
        ast.archive_name = safe
        ast.archive_list = list_archives(root)
        ast.selected_entry_id = -1
        # Deliberately NOT resumed: the user pressed a management button, not
        # Start.
        return True

    def _ensure_archive_service(self, ui_state):
        """Build the archive, goal list and IMGEP driver on first use.

        Explore mode reuses the SAME AutoTournamentService instance - the
        rollout machine is identical - and only swaps its driver. Imports stay
        lazy: onnxruntime and cmaes must not be imported at startup.
        """
        if not self._ensure_auto_service():
            self.ui.archive_unavailable = self.ui.auto_unavailable
            return False
        if self.imgep_driver is not None:
            return True

        from services.archive_library import list_archives
        from services.imgep_driver import ImgepDriver
        from utilities.paths import get_archives_root

        ast = ui_state.archive
        path = self._archive_path_for(ui_state.preferences.archive_name, ast)
        self._build_archive_set(path)

        self.imgep_driver = ImgepDriver(
            self.tournament_service, self.clip_scorer,
            self.archive, self.goal_list)
        self.prompt_driver = self.auto_service.driver

        ast.archive_name = path.name
        ast.archive_list = list_archives(get_archives_root())
        ui_state.preferences.archive_name = path.name

        self.ui.archive_driver = self.imgep_driver
        self.ui.archive_service = self.auto_service
        self.ui.archive_unavailable = ""
        self.command_handler.imgep_driver = self.imgep_driver
        return True
```

- [ ] **Step 4: Pass `ui_state` at the one call site**

`main.py:432` becomes:

```python
            if self._ensure_archive_service(ui_state):
```

- [ ] **Step 5: Wire the callback `CommandHandler` will need in Task 7**

In `App.__init__`, immediately after `self.command_handler` is constructed, add:

```python
        # Archive switching is orchestration, so CommandHandler asks for it
        # rather than reaching into App.
        self.command_handler.switch_archive = self._switch_archive
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_archive_switch.py -q
```

Expected: all pass.

- [ ] **Step 7: Run the whole suite**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

Expected: no new failures.

- [ ] **Step 8: Commit**

```bash
git add main.py tests/test_archive_switch.py
git commit -m "feat: switch the active archive without restarting"
```

---

### Task 7: `CommandHandler` consumes the four flags

The one-shot flags from Task 5 become filesystem operations and a switch.

**Files:**
- Modify: `command_handler.py:32-36` (attribute), `command_handler.py:172-178`
  (dispatch), `command_handler.py:434-449` (`_clear_explore_flags`), and a new
  method after it
- Test: `tests/test_archive_commands.py` (create)

**Interfaces:**
- Consumes: `create`, `clear`, `delete`, `list_archives` (Tasks 1-2);
  `get_archives_root` (Task 3); the `ArchiveState` flags (Task 4);
  `self.switch_archive` set by `App` (Task 6).
- Produces: `CommandHandler._handle_archive_management(ui_state)`;
  `self.switch_archive` attribute, default `None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_archive_commands.py`:

```python
"""The archive-management one-shots.

These run BEFORE _handle_explore so a switch takes effect on the frame the
button was pressed, and so the settings push that follows lands on the archive
the user just chose.
"""
import pytest

from state.archive_state import ArchiveState


class _UIState:
    def __init__(self):
        self.archive = ArchiveState()


class _Handler:
    """Stands in for CommandHandler, with only the attributes the method under
    test touches."""

    def __init__(self):
        self.switched = []
        self.switch_archive = lambda name, ui: (self.switched.append(name), True)[1]


def run(handler, ui_state):
    from command_handler import CommandHandler
    return CommandHandler._handle_archive_management(handler, ui_state)


@pytest.fixture
def root(tmp_path, monkeypatch):
    import utilities.paths as paths

    monkeypatch.setattr(paths, "get_user_data_dir", lambda: tmp_path)
    r = tmp_path / "archives"
    (r / "default" / "thumbs").mkdir(parents=True)
    return r


def test_new_creates_the_archive_and_switches_to_it(root, tmp_path):
    h, ui = _Handler(), _UIState()
    ui.archive.new_archive_name = "run-07"
    ui.archive.new_archive_requested = True

    run(h, ui)

    assert (root / "run-07" / "thumbs").is_dir()
    assert h.switched == ["run-07"]


def test_a_refused_name_warns_and_does_not_switch(root, tmp_path):
    h, ui = _Handler(), _UIState()
    ui.archive.new_archive_name = "default"
    ui.archive.new_archive_requested = True

    run(h, ui)

    assert "already exists" in ui.archive.warning
    assert h.switched == []


def test_clear_empties_the_active_archive_and_reloads_it(root, tmp_path):
    (root / "default" / "index.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
    h, ui = _Handler(), _UIState()
    ui.archive.clear_archive_requested = True

    run(h, ui)

    assert (root / "default" / "index.jsonl").exists() is False
    assert h.switched == ["default"], "the in-memory archive must be reloaded"


def test_delete_removes_it_and_switches_to_what_is_left(root, tmp_path):
    (root / "keeper" / "thumbs").mkdir(parents=True)
    h, ui = _Handler(), _UIState()
    ui.archive.archive_name = "default"
    ui.archive.delete_archive_requested = True

    run(h, ui)

    assert not (root / "default").exists()
    assert h.switched == ["keeper"]


def test_delete_refuses_the_last_archive(root, tmp_path):
    h, ui = _Handler(), _UIState()
    ui.archive.delete_archive_requested = True

    run(h, ui)

    assert (root / "default").is_dir()
    assert "only archive" in ui.archive.warning
    assert h.switched == []


def test_the_dropdown_selection_switches(root, tmp_path):
    (root / "other" / "thumbs").mkdir(parents=True)
    h, ui = _Handler(), _UIState()
    ui.archive.switch_archive_name = "other"

    run(h, ui)

    assert h.switched == ["other"]


def test_refresh_rebuilds_the_cached_listing(root, tmp_path):
    (root / "added-outside" / "thumbs").mkdir(parents=True)
    h, ui = _Handler(), _UIState()
    ui.archive.refresh_archive_list_requested = True

    run(h, ui)

    assert [a["name"] for a in ui.archive.archive_list] == ["added-outside",
                                                            "default"]


def test_the_flags_are_cleared_after_one_frame(root, tmp_path):
    """Otherwise the modal's Create would fire again every frame forever."""
    h, ui = _Handler(), _UIState()
    ui.archive.new_archive_name = "run-07"
    ui.archive.new_archive_requested = True
    ui.archive.clear_archive_requested = True
    ui.archive.delete_archive_requested = True
    ui.archive.switch_archive_name = "default"
    ui.archive.refresh_archive_list_requested = True

    run(h, ui)

    ast = ui.archive
    assert ast.new_archive_requested is False
    assert ast.clear_archive_requested is False
    assert ast.delete_archive_requested is False
    assert ast.switch_archive_name == ""
    assert ast.refresh_archive_list_requested is False


def test_nothing_happens_without_a_switch_callback(root, tmp_path):
    """Explore mode has never been opened, so App has not wired it up."""
    h, ui = _Handler(), _UIState()
    h.switch_archive = None
    ui.archive.new_archive_name = "run-07"
    ui.archive.new_archive_requested = True

    run(h, ui)

    assert not (root / "run-07").exists()
    assert ui.archive.new_archive_requested is False
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/Scripts/python.exe -m pytest tests/test_archive_commands.py -q
```

Expected: `AttributeError: type object 'CommandHandler' has no attribute '_handle_archive_management'`.

- [ ] **Step 3: Write the implementation**

In `CommandHandler.__init__`, after `self.archive_projection = None`:

```python
        # App._switch_archive; None until Explore mode has been opened once.
        self.switch_archive = None
```

Add the method immediately before `_handle_explore`:

```python
    @staticmethod
    def _clear_archive_flags(ast):
        ast.switch_archive_name = ""
        ast.new_archive_requested = False
        ast.clear_archive_requested = False
        ast.delete_archive_requested = False
        ast.refresh_archive_list_requested = False

    def _handle_archive_management(self, ui_state):
        """Create, empty, delete and switch archives.

        Runs BEFORE _handle_explore so a switch lands on the same frame the
        button was pressed, and so the settings push that follows goes to the
        archive the user just chose.
        """
        from services.archive_library import (clear, create, delete,
                                              list_archives)
        from utilities.paths import get_archives_root

        ast = ui_state.archive
        if self.switch_archive is None:
            self._clear_archive_flags(ast)
            return

        root = get_archives_root()
        target = ""

        if ast.new_archive_requested:
            res = create(root, ast.new_archive_name)
            if res.ok:
                ast.new_archive_name = ""
                target = res.name
            else:
                ast.warning = res.message
        elif ast.clear_archive_requested:
            res = clear(root, ast.archive_name)
            if res.ok:
                # Same name, now an empty directory: the in-memory archive must
                # be rebuilt or it would keep serving entries that are gone.
                target = res.name
            else:
                ast.warning = res.message
        elif ast.delete_archive_requested:
            res = delete(root, ast.archive_name)
            if res.ok:
                remaining = list_archives(root)
                target = remaining[0]["name"] if remaining else ""
            else:
                ast.warning = res.message
        elif ast.switch_archive_name:
            target = ast.switch_archive_name

        if target:
            self.switch_archive(target, ui_state)
        elif ast.refresh_archive_list_requested:
            # A switch refreshes the listing itself, so this is only for the
            # Refresh button on its own.
            ast.archive_list = list_archives(root)

        self._clear_archive_flags(ast)
```

In `process_commands`, insert the call before `_handle_explore` (line 175):

```python
        # Archive management, before Explore so a switch lands this frame
        self._handle_archive_management(ui_state)

        # Explore (IMGEP) mode
        self._handle_explore(ui_state)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/test_archive_commands.py -q
```

Expected: all pass.

- [ ] **Step 5: Run the whole suite**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

Expected: no new failures.

- [ ] **Step 6: Commit**

```bash
git add command_handler.py tests/test_archive_commands.py
git commit -m "feat: handle the archive management commands"
```

---

### Task 8: Round-trip proof and documentation

One test for the property the whole feature exists to provide, then the docs a
future reader needs.

**Files:**
- Modify: `tests/test_archive_switch.py` (append)
- Modify: `docs/testing_checklist.md`
- Modify: `ARCHITECTURE.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: everything from Tasks 1-7.
- Produces: no new code interfaces.

- [ ] **Step 1: Write the failing round-trip test**

Append to `tests/test_archive_switch.py`:

```python
def test_an_archive_is_untouched_by_a_round_trip(roots):
    """Build A, switch to B, admit to B, come back to A. A must be exactly as
    it was - this is the entire point of the feature."""
    from services.archive import Archive
    from services.archive_io import ArchiveStore

    store = ArchiveStore(roots / "default")
    a = Archive(store=store, seed_n=0, liveness_min=0.0)
    _admit(a, 4)
    a.maybe_flush(force=True)
    store.close()

    log = _Log()
    app, ui = _FakeApp(log), _UIState()
    ui.preferences.archive_name = "default"

    switch(app, "default", ui)
    assert len(app.archive) == 4

    switch(app, "dense-trails", ui)
    assert len(app.archive) == 0
    _admit(app.archive, 2)
    app.archive.maybe_flush(force=True)

    switch(app, "default", ui)
    assert len(app.archive) == 4, "A must not have seen B's entries"

    switch(app, "dense-trails", ui)
    assert len(app.archive) == 2, "B kept what was admitted to it"


def test_goal_lists_do_not_leak_between_archives(roots):
    """A goal list is part of the experiment, not a global preference."""
    log = _Log()
    app, ui = _FakeApp(log), _UIState()

    switch(app, "default", ui)
    app.goal_list.add("coral reef")
    app.goal_list.save()

    switch(app, "dense-trails", ui)
    assert len(app.goal_list.items) == 0

    switch(app, "default", ui)
    assert [i["text"] for i in app.goal_list.items] == ["coral reef"]
```

- [ ] **Step 2: Run it**

```bash
.venv/Scripts/python.exe -m pytest tests/test_archive_switch.py -q
```

Expected: PASS with the implementation from Tasks 1-7. (`GoalList.items` is a
`list[dict]` with a `"text"` key — `services/goal_source.py:46`.)

- [ ] **Step 3: Add the manual checks**

Append to `docs/testing_checklist.md`, as a new section:

```markdown
## 14. Archive management

- [ ] First launch after updating: the old `Documents/Fluoddity/archive/`
      folder is gone and `Documents/Fluoddity/archives/default/` holds its
      contents. Explore mode's archive count matches what it was before.
- [ ] Explore tab: the Archive row names the active archive and its entry
      count matches the Archive Browser's.
- [ ] New -> type `a/b` -> the preview says it will be saved as `ab`; Create
      makes `archives/ab/` and the row switches to it.
- [ ] New -> type the name of an existing archive -> Create is disabled and the
      reason is shown.
- [ ] Switch archives while a search is running: it stops, the entry count
      changes, the gallery shows the new archive's thumbnails (not the old
      one's), and pressing Start explores into the new archive.
- [ ] Switch back: the first archive's entries and goal list are exactly as
      they were.
- [ ] Empty -> confirm: the archive shows 0 entries and a
      `<name>.cleared-<time>` folder appears beside it with the old contents.
- [ ] Delete: the button is disabled when only one archive exists; otherwise it
      requires typing the name, and afterwards the row switches to another
      archive.
- [ ] Delete an archive folder in Explorer while the app is running, then pick
      it in the dropdown: a warning appears and nothing else changes.
- [ ] Quit while a non-default archive is active; relaunch: the same archive is
      loaded.
```

- [ ] **Step 4: Document the layout**

In `ARCHITECTURE.md`, add a row to the "Search and exploration (`services/`)"
table (after the `archive_io.py` row at line 98) — fill in the real line count
with `git grep -c "" services/archive_library.py`:

```markdown
| `archive_library.py` | NN | Named archive directories: safe names, listing, create / clear / delete |
```

Then add this paragraph immediately below that table:

```markdown
Archives live under `Documents/Fluoddity/archives/<name>/`, one directory per
archive, each with the `index.jsonl` / `vectors.npz` / `goals.json` / `thumbs/`
shape `ArchiveStore` writes. There is no registry file: the filesystem is the
list, so a folder copied in from elsewhere just appears. `App._switch_archive`
rebuilds the archive, goal list, projection and thumbnail cache and repoints
every holder at the new directory; which archive is active is
`preferences.archive_name`.
```

(The line-count column is stale for the files this plan edits. Refresh
`archive_state.py`, `preferences_state.py` and `archive_window.py` too, with
`git grep -c "" <path>`.)

- [ ] **Step 5: Record the two traps in `CLAUDE.md`**

Add to the "Important Caveats" list in `CLAUDE.md`:

```markdown
- **Entry ids restart at 0 in every archive**, so `ThumbCache` must be released
  on a switch — it is keyed by thumbnail filename, which is derived from the
  entry id. Reusing it shows the previous archive's pictures under the new
  archive's entries.

- **`Archive.maybe_flush` only rewrites `vectors.npz` every 200 admissions.**
  `index.jsonl` is flushed per entry, so anything that closes an archive —
  quitting, or switching to another one — must call `maybe_flush(force=True)`
  first or lose the trailing entries.
```

- [ ] **Step 6: Run the whole suite one last time**

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add tests/test_archive_switch.py docs/testing_checklist.md ARCHITECTURE.md CLAUDE.md
git commit -m "test: archive round-trip; docs for archive management"
```

---

## Spec coverage

| Spec section | Task |
| --- | --- |
| §1 layout, `get_archives_root`, `archive_name` preference | 3, 4 |
| §1 migration (all four cases) | 3 |
| §2 `safe_name`, `resolve`, `list_archives` | 1 |
| §2 `create`, `clear`, `delete` | 2 |
| §2 listing not called per frame | 1 (docstring), 4 (`archive_list`), 7 (refresh) |
| §3 dropdown, New/Clear/Delete, modals, browser title | 5 |
| §3 `ArchiveState` fields | 4 |
| §3 switch sequence, steps 1-8 | 6 |
| §3 does not auto-resume | 6 |
| §3 New/Clear/Delete route through the switch | 7 |
| §4 `Result`, warnings not exceptions | 1, 2, 7 |
| §4 refuse empty / duplicate / last archive / vanished folder | 2, 7 |
| §4 `resolve` raises on escape | 1 |
| §4 testing (library, render, switch, round-trip) | 1, 2, 5, 6, 8 |
