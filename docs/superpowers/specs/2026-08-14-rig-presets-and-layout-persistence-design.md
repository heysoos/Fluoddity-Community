# Audio Rig Presets and Layout Persistence — Design

**Date:** 2026-08-14
**Status:** Design approved, ready to implement
**Branch context:** `worktree-audio-reactive`

## Goal

Two complaints, one theme — work that should survive a restart does not:

1. The audio rig comes back empty "many times", and there is no way to keep
   more than one rig around while prototyping.
2. Every launch needs the same windows opened and placed again.

## Decisions

| Decision | Choice |
|---|---|
| Rig wipe fix | Compare at exit; write only if this session changed it |
| Autosave polling | **Not implemented** — no evidence of the failure it guards |
| Rig write failures | Surfaced as a panel notice, not a swallowed `False` |
| Preset storage | `Documents/Fluoddity/audio_rigs/<name>.json` |
| Preset access | A row at the top of the Audio Reactive panel |
| Preset save path | The app's existing single save dialog, new `kind` |
| Preset deletion | Out of scope |
| `imgui.ini` | Moves to `Documents/Fluoddity/imgui.ini` |
| Window flags | Into `PreferencesState`, which already round-trips |
| Restored at launch | View windows + Archive Browser |
| Not restored | Tournament Mode, Brain Modality |

## Root causes

Both bugs were diagnosed before any design, and neither is what it looked like.

### The rig is not failing to save — it is being overwritten

`save_rig` runs in `cleanup()`, `load_rig` at startup, and both work: the file
on disk holds four mappings, and `to_dict`/`apply_dict` round-trip it exactly.
`get_state()` returns the live state object, not a copy, so cleanup sees the
user's edits.

The file is `Documents/Fluoddity/audio_rig.json` — **shared by every copy of
Fluoddity**, across five worktrees and any instance another session launches.
The write at exit is unconditional, so an instance that never touched the rig
still writes its own empty state over yours. Confirmed with the user: a second
copy is open "a lot".

`preferences.config` and `audio_rig.json` share a write timestamp to the second,
which is what rules out cleanup failing to run.

### The layout is not failing to save — it is per-folder

The app never calls `set_ini_filename`, so Dear ImGui writes `imgui.ini`
relative to the current working directory. `get_app_dir()` is the worktree
directory, so each of the five keeps a separate layout. `paths.py` seeds
`default_imgui.ini` into the app dir, which the running app may never read.

## Part 1 — The audio rig

### Stopping the wipe

The rule: **a session writes the rig only if it changed it.**

The orchestrator keeps the dict `load_rig` produced at startup. At exit it
builds `to_dict(audio)` and compares; equal means no write.

- An untouched instance produces an identical dict and writes nothing, so it
  cannot erase another instance's work.
- A rig loaded from a preset counts as a change, so it becomes the new
  last-used.
- Two instances that *both* edit still resolve last-writer-wins. Not fixable —
  two rigs cannot be merged — and out of scope.

**Polling was considered and rejected.** A once-a-second diff costs 4.5 us on a
real rig and 145 us on a 200-mapping one, so cost is not the objection: it
guards only a hard kill, which is the one thing that skips `cleanup()`'s
`finally`, and the crash log shows two entries in two days against a confirmed
second-instance cause. It is machinery for a failure mode with no evidence. If
hard-kill loss is ever observed, the poll is a few lines and gets added then.

### Making failure visible

`save_rig` catches every exception and returns `False` to a caller that ignores
it. It keeps returning a bool, and the orchestrator turns `False` into a notice
in the Audio panel. A save that only fails silently reads as a save.

### Named presets

Rigs live in `Documents/Fluoddity/audio_rigs/<name>.json` — deliberately NOT
the user configs folder, because everything there appears in the physics
File > Load menu and a rig is not a physics config.

`services/audio_rig_io.py` gains:

- `rigs_dir()` — the folder, created on demand.
- `list_rigs()` — names, sorted, no extension.
- `preset_path(name)` — a path under `rigs_dir()`.

`save_rig` and `load_rig` already accept an explicit path and are reused
unchanged.

The Audio Reactive panel gets one row above the matrix: a combo of the
available presets, **Save**, and **Load**. Load applies to the live rig
immediately. Save calls `UI.open_save_popup()` with a new `kind`, dispatched in
`CommandHandler._handle_file_save` — per CLAUDE.md that dialog is the only way
anything gets written, which brings the overwrite confirmation and `safe_stem()`
with it, and `safe_stem` is what stops a typed separator writing outside the
folder.

## Part 2 — The layout

### `imgui.ini`

`io.set_ini_filename(str(get_imgui_ini_path()))` at UI construction, with
`get_imgui_ini_path()` changed to return `Documents/Fluoddity/imgui.ini`.

- The data dir is always writable; the app dir is not, in a PyInstaller build
  under Program Files. This is a correctness fix as much as a convenience.
- `default_imgui.ini` is seeded there instead.
- **Migration:** if the new path is absent and an old app-dir `imgui.ini`
  exists, copy it once, so the current arrangement survives the move.

**The test suite must not touch it.** `tests/conftest.py` nulls the ini filename
by wrapping `create_context`; `UI.__init__` runs afterwards and would override
that, pointing the whole suite at the user's real layout. The existing caveat
records that the suite once consumed this file and made an unrelated test fail.
So the call lives in one function that `conftest` neutralises, guarded by a test
asserting the suite never resolves the real path. Never call
`get_ini_filename()` on a null — it segfaults.

### Which windows reopen

`PreferencesState` round-trips through `asdict`, so a field there persists for
free; that is why its five help-window flags already work.

**Moved into `PreferencesState`** (19 non-test references, mechanical):

- `show_sidebar`
- `show_physics_settings_window`
- `show_video_recording_window`
- `show_history_window`

`show_demo_window` is left alone — a developer toggle, not part of a layout.

**Mirrored, not moved:** `state.audio.show_window` and
`state.archive.show_browser` gain preference counterparts
(`show_audio_window`, `show_archive_browser`). Moving them outright would drag
the audio panel's visibility into rig presets — where loading a preset would
open and close the panel — and would churn the archive tests. Startup applies
preference to state; `get_state()` copies state back to preference, passively,
in keeping with the UI owning no logic.

### Archive Browser

Restored, at the user's request. At startup, if `show_archive_browser` is set,
`show_browser` and `open_browser_requested` are set **exactly once** — never per
frame, or the browser would reload every frame.

The cost is real and belongs in the docs rather than a surprise: reopening runs
`load_from_store`, whose `rescore_all()` is load-bearing — roughly 0.4 s at
~4,800 entries and ~4 s at capacity, added to launch. No CLIP session is built,
so the ONNX cost is not paid; `_open_archive` already draws that line.

Tournament Mode and Brain Modality are deliberately not restored: both change
how the app behaves rather than what is on screen.

## Testing

- An unchanged session writes nothing: load a rig, touch nothing, run the exit
  path, and the file is byte-identical. This is the regression that matters.
- A changed session writes; loading a preset counts as a change.
- A failed write sets a notice rather than passing silently.
- Preset round-trip: save under a name, load it back, get the same rig.
- A name with a path separator cannot write outside `audio_rigs/`.
- `list_rigs()` on a missing folder returns empty rather than raising.
- The ini path is absolute and under the data dir; migration copies once and
  never twice.
- The test suite never resolves the real ini path.
- Preferences round-trip including every moved and mirrored flag.
- The archive restore requests an open exactly once, and not at all when the
  flag is false.
- Existing render smoke tests still pass with the flags moved.

Manual items go to `docs/testing_checklist.md`: two instances open, edit in one,
close the untouched one first, and confirm the rig survives.

## Out of scope

- Preset deletion and renaming.
- Merging two concurrently-edited rigs.
- Restoring Tournament Mode or Brain Modality.
- Per-preset window layouts.
