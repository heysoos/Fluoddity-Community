# Archive management — design

Date: 2026-08-07
Status: approved design, ready for an implementation plan
Follows: `2026-08-07-imgep-novelty-archive-design.md`

## Problem

There is exactly one archive, at `Documents/Fluoddity/archive/`, hardcoded at
`main.py:241`. Every Explore run therefore pours into the same pot. Two
consequences:

- Runs cannot be compared, because they cannot be separated. An archive built
  under one preset, grid size and liveness floor is silently mixed with the
  next.
- The only way to start clean is to quit the app and delete a folder by hand.
  The user's stated plan before the next run was "we will erase that whole
  archive" — which today means File Explorer, not the app.

Novelty is measured against the archive, so its contents *are* the search's
memory. Which archive is loaded is an experimental variable, and it should be
visible and switchable in the UI.

## Goals

- Multiple named archives, one active at a time.
- Switch between them, create a new empty one, empty the active one, and delete
  one — all without restarting the app.
- The existing archive keeps working and keeps its contents.
- Nothing is destroyed without a confirmation, and "empty it" is recoverable.

## Non-goals

- Merging or copying entries between archives.
- Renaming an archive in place. (Create a new one; the old name stays.)
- Any change to entry format, novelty, admission or the search itself.
- Sharing goal lists across archives.

---

## 1. Layout and migration

### Layout

```
Documents/Fluoddity/archives/
    default/
        index.jsonl
        vectors.npz
        goals.json
        thumbs/000123.jpg …
    dense-trails/
        …
```

Each archive directory has exactly the shape `ArchiveStore` already writes —
nothing inside an archive changes. An archive is a directory; the directory
name is the archive's name. There is no registry file, no manifest, no JSON
index of archives. The filesystem is the list, which means an archive folder
copied in from elsewhere just appears, and one deleted outside the app just
disappears.

Goals stay inside the archive (`goals.json`), as they are today. A goal list is
part of the experiment, not a global preference.

`utilities.paths`:

- `get_archives_root() -> Path` — `get_user_data_dir() / "archives"`. Replaces
  `get_archive_dir()`, which is removed. `initialize_user_data()` creates the
  root and runs the migration below.
- The active archive's path is `archive_library.resolve(root, name)`, never a
  path built by string concatenation at the call site.

### Which archive is active

`PreferencesState.archive_name: str = "default"`, persisted with every other
preference. On launch the app resolves that name; if the folder is gone (user
deleted it outside the app, or moved their Documents), it falls back to
`"default"`, creating it if needed, and sets `ArchiveState.warning` so the
substitution is visible rather than silent.

### Migration

On first launch after this change, in `initialize_user_data()`:

- If `archives/` does not exist and legacy `archive/` does →
  `os.replace(archive, archives/default)`. A move, not a copy: archives run to
  hundreds of megabytes of thumbnails, and duplicating them to be tidy is worse
  than moving them.
- If `archives/` does not exist and legacy `archive/` does not either → create
  `archives/default/` empty.
- If `archives/` already exists → do nothing at all, and in particular leave any
  legacy `archive/` folder alone. A user who has already migrated and then
  restores an old backup should not have that backup swallowed into a directory
  that already has contents.
- If the move fails (file locked, permissions) → print one warning, create
  `archives/default/` empty, and leave the legacy folder untouched. Migration
  failing must not stop the app from launching.

---

## 2. The service — `services/archive_library.py`

A new module with no imports beyond the standard library and `pathlib`. It
knows about directories, not about `Archive`, `ImgepDriver` or GL. That is what
makes it testable against `tmp_path` with no GPU and no CLIP.

```python
def safe_name(raw: str) -> str
def list_archives(root: Path) -> list[dict]
def resolve(root: Path, name: str) -> Path
def create(root: Path, name: str) -> Result
def clear(root: Path, name: str) -> Result
def delete(root: Path, name: str) -> Result
```

`Result` is defined in §4.

### `safe_name(raw)`

Case and unicode are preserved — `Dense Trails` and `густой` are both fine
names. What is stripped is anything that can change *which directory* a path
refers to: path separators, drive-letter colons, `..`, control characters, and
leading or trailing dots and spaces (Windows silently trims trailing dots and
spaces from directory names, so a name ending in one would not round-trip).
Truncated to 64 characters.

Returns `""` for input that sanitises away to nothing. Callers treat `""` as a
refusal — see §4.

### `resolve(root, name)`

`(root / safe_name(name)).resolve()`, then asserts the result is inside
`root.resolve()`. If it is not, it raises — this is a programming error, not a
user error, because `safe_name` already ran. This is belt-and-braces around the
one operation in this feature that deletes a directory tree.

### `list_archives(root)`

One dict per subdirectory of `root`, sorted by name:

```python
{"name": str, "entries": int, "mtime": float, "size_mb": float}
```

Subdirectories whose name contains `.cleared-` are excluded — they are undo
snapshots, not archives (see `clear` below).

`entries` is the line count of `index.jsonl` — the same file that is the
authority for which entries exist. It is a cheap read, and it is honest: an
archive whose `vectors.npz` was quarantined still reports the rows it has.
`size_mb` is the summed file size of the directory tree, which is dominated by
`thumbs/`. A directory that cannot be read is skipped rather than propagated as
an error; a listing is a convenience, and one bad folder should not blank the
dropdown.

**This is not cheap enough to call per frame.** An archive with 4,800 thumbnails
is 4,800 `stat` calls, and ImGui is immediate-mode — the Explore tab renders
every frame. The orchestrator calls `list_archives` once when the archive
service is built and again after any create/clear/delete/switch, caching the
result on `ArchiveState.archive_list`. The UI only reads that cache. A Refresh
item at the bottom of the dropdown re-runs it on demand, for the case where the
user added a folder outside the app.

### `create(root, name)`

Makes `root/<safe>/` and its `thumbs/` subdirectory. Refuses if the name is
empty after sanitising or if the directory already exists. Deliberately does
**not** auto-suffix `-2`: silently creating `run-2` when the user typed `run`
puts entries somewhere they did not ask for. Refuse and say why.

### `clear(root, name)`

Renames `root/<name>` to `root/<name>.cleared-<unix-ts>`, then creates a fresh
empty `root/<name>`. Nothing is deleted. The `.cleared-` infix is the marker
`list_archives` filters on, and because `safe_name` only strips dots at the ends
of a name, a user *could* type a name containing `.cleared-` and have it hidden
from their own dropdown. That is accepted: refusing the substring would be a
confusing rule to explain, and the failure mode is an archive that must be
renamed on disk rather than one that is lost. A user who clears the wrong
archive recovers the same way — by renaming the folder back.

Old `.cleared-*` folders are never garbage-collected by the app. They are the
user's data; the app is not going to decide when their undo expires.

### `delete(root, name)`

`shutil.rmtree` on the resolved path. Refuses when it would remove the last
remaining archive — there must always be something to load. This is the only
irreversible operation in the feature, and it is behind a typed-name
confirmation (§3).

---

## 3. UI and switching

### Widgets

At the top of the Explore tab, above the status block, a single row:

```
Archive: [ default            v ]  [ New ]  [ Clear ]  [ Delete ]
             1,511 entries · 214 MB
```

The combo lists `list_archives()` by name with the entry count beside each. The
line beneath shows the active archive's counts, so the number the user is
looking at is the archive the search is writing to.

Buttons open modals rather than acting immediately:

- **New** — a text field, live-previewing the sanitised name if it differs from
  what was typed, and a Create button disabled while the name is empty or taken.
- **Clear** — "Empty *default*? Its 1,511 entries move to `default.cleared-…`
  and can be restored by renaming that folder." Confirm / Cancel.
- **Delete** — requires typing the archive's name to enable the button. Disabled
  entirely, with an explanation, when this is the only archive.

The Archive Browser window's title shows the active archive name, so a gallery
screenshot says which archive it came from.

### State — `state/archive_state.py`

New fields, following the existing one-shot convention (set by UI, cleared by
`CommandHandler`):

```python
archive_name: str = "default"     # mirrors the active archive, for display
archive_list: list = field(default_factory=list)   # cached list_archives()
new_archive_name: str = ""        # modal text buffer
switch_archive_name: str = ""     # one-shot
new_archive_requested: bool = False
clear_archive_requested: bool = False
delete_archive_requested: bool = False
refresh_archive_list_requested: bool = False
```

`warning` (already present, persistent, user-dismissed) carries every failure
message.

### Switching — `App._switch_archive(name)`

The orchestrator owns this; the UI only sets the flag. In order:

1. **Pause the search.** `ast.running = False` and the rollout service is
   stopped, the same path the Pause button takes. Switching mid-generation would
   score tiles against one archive and admit them to another.
2. **End any expedition.** `imgep_driver.end_expedition()`. The CMA-ES optimizer
   was seeded from a parent in the outgoing archive; its mean is meaningless
   against the new one.
3. **Flush the outgoing archive.** `archive.maybe_flush(force=True)`,
   `goal_list.save()`, `store.close()`. `index.jsonl` is flushed per entry, but
   `vectors.npz` is only rewritten every 200 admissions — without a forced flush
   the trailing entries are lost on switch.
4. **Release GL textures.** `thumb_cache.release()`. The cache holds up to 256
   GL textures keyed by entry id, and entry ids restart per archive — reusing it
   would show the previous archive's thumbnails under the new archive's ids.
5. **Rebuild.** New `ArchiveStore`, `Archive` (+ `load_from_store()`),
   `GoalList` (+ `load()`), `Projection` (+ `fit`), `ThumbCache`.
6. **Repoint every holder.** `imgep_driver.archive`, `imgep_driver.goals`,
   `command_handler.archive`, `command_handler.goal_list`,
   `command_handler.archive_projection`, `ui.archive_obj`, `ui.archive_goals`,
   `ui.archive_projection`, `ui.thumb_cache`, `self.archive_store`,
   `self.archive`, `self.goal_list`, `self.thumb_cache`,
   `self.archive_projection`, `self._last_projection_size`. The driver instance
   itself is kept, so exploration settings the user tuned survive the switch.
7. **Persist the choice.** `preferences.archive_name = name`.
8. **Clear the selection.** `ast.selected_entry_id = -1`, `ast.archive_name =
   name`.

Switching does **not** auto-resume. The user pressed a management button, not
Start; resuming for them would begin writing into an archive they may only have
wanted to look at.

New, Clear and Delete all route through the same path: create/clear/delete the
directory, then `_switch_archive` to the resulting active name (the new one for
New; the same name for Clear, which now resolves to an empty directory; the
first remaining archive for Delete).

`_ensure_archive_service` builds against `preferences.archive_name` on first
use, so nothing special happens at startup — the ordinary path already resolves
by name.

---

## 4. Failure handling and testing

### Failure handling

Every library operation returns a result rather than raising:

```python
@dataclass
class Result:
    ok: bool
    name: str = ""       # the sanitised name that was acted on
    message: str = ""    # user-facing, empty when ok
```

The orchestrator surfaces `message` through `ast.warning`, which the Explore tab
already renders in red with a Dismiss button. No dialog, no exception, no
console-only failure.

Decisions, stated explicitly so the implementation cannot pick the other
reading:

- **Empty sanitised name → refuse**, message "That name has no usable
  characters." Do not auto-generate `archive-1`.
- **Duplicate name → refuse**, message "An archive named X already exists." Do
  not auto-suffix.
- **Target folder vanished** (deleted outside the app between the dropdown
  render and the click) → a no-op with a warning, not a crash. `clear` and
  `delete` both treat "already gone" as nothing to do.
- **Delete the last archive → refuse**, message "This is the only archive." The
  button is also disabled in that state; the service check is the one that
  matters, because the button state can be stale.
- **Migration failure** → warning at startup, app launches with an empty
  `default`.
- **`resolve` escaping the root** → raises. Unreachable through the UI; it
  exists so that a future caller that skips `safe_name` fails loudly instead of
  deleting something outside `archives/`.

### Testing

The weight of the testing goes on `archive_library`, because it is pure
filesystem logic and can be exercised exhaustively against `tmp_path` with no
GL, no CLIP and no window:

- `safe_name`: path separators, `..`, absolute paths, drive letters, control
  characters, leading/trailing dots and spaces, length truncation, unicode
  preserved, empty result for pathological input.
- `resolve`: every `safe_name` output stays inside the root; a hand-crafted
  escaping name raises.
- `list_archives`: entry counts match `index.jsonl` line counts; `.cleared-*`
  folders are excluded; an unreadable directory is skipped, not fatal; empty
  root returns `[]`.
- `create`: makes `thumbs/`; refuses duplicates and empty names.
- `clear`: the original contents survive under `<name>.cleared-<ts>`; the live
  directory is empty afterwards; the cleared folder is not listed.
- `delete`: removes the tree; refuses when it is the last archive.
- Migration: legacy-only → moved with contents intact; neither → empty default;
  both present → legacy untouched; move failure → app still gets a default.

Then, above that:

- Render tests for the archive row and each modal, in the style of
  `test_archive_window_render.py`: the row renders with zero, one and many
  archives; Delete is disabled with one archive; Create is disabled on an empty
  or duplicate name; the button sets the expected one-shot flag and nothing
  else.
- A switch test with fakes asserting the whole ordered sequence of §3: paused,
  expedition ended, forced flush, goals saved, store closed, thumb cache
  released, every holder repointed, preference written, and *not* resumed.
- A round-trip test: build archive A with entries, switch to B, admit to B,
  switch back to A, and assert A's entry count and goal list are unchanged —
  the property the whole feature exists to provide.

---

## Files touched

| File | Change |
| --- | --- |
| `services/archive_library.py` | new — `safe_name`, `list_archives`, `resolve`, `create`, `clear`, `delete`, `Result` |
| `utilities/paths.py` | `get_archives_root()` replaces `get_archive_dir()`; migration in `initialize_user_data()` |
| `state/preferences_state.py` | `archive_name: str = "default"` |
| `state/archive_state.py` | eight new fields (§3) |
| `ui/archive_window.py` | archive row + three modals; browser title shows the name |
| `main.py` | `_switch_archive`; `_ensure_archive_service` resolves by name |
| `command_handler.py` | handle the four new one-shot flags |
| `tests/` | `test_archive_library.py`, additions to `test_archive_window_render.py`, `test_archive_switch.py` |
