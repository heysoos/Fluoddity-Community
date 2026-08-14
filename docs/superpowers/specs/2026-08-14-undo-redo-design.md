# Undo and redo for the live configuration

**Date:** 2026-08-14
**Status:** design approved, not yet implemented

## Problem

Every control in the app writes straight into live state and leaves no way
back. A stray `Z` blanks the brain, a `G` re-rolls the seed, hovering down the
Load menu applies presets one after another, and a slider dragged past the
value you liked cannot be dragged back to it — the number is gone. The only
memory the app has is the Config Clipboard, which remembers what you thought to
press `Ctrl+C` on, and `RuleManager`, which is a preview stack rather than a
history: `pop_rule()` answers `(None, None)` when the popped rule was the only
one, and the hover-preview paths are built on top of it.

So the app can destroy a configuration you were in the middle of finding, and
the loss is silent.

## Scope

Undo covers the **recipe** — everything that decides what the creature is and
how it is drawn — and nothing that would need the GPU's contents to come back.

In scope:

- `SimState`: the 13 physics parameters, slider ranges, all three sweep dicts,
  jitters, sim settings, appearance, notes.
- `PreferencesState`: world size, particle density, bloom, brightness,
  tonemap, exposure, motion blur, draw size/power, and the rest of the render
  settings.
- `AudioInState`: exactly its existing `PERSISTED_FIELDS` — mappings, brain
  mappings, strengths, global strength, auto gain, device, modulate, muted.
  That state lives on the `audio-reactive` worktree and needs one prerequisite
  there before it can join; see the `Mapping` uid section below.
- The rule, its brain layout signature, and the brain's decode settings.

**Out of scope, deliberately.** Each of these was considered and excluded, so
that a later reader knows the omission is a decision:

- **The canvas and entity buffers.** Undo restores the recipe, not the picture:
  the pattern regrows from the restored settings rather than snapping back to
  the pixels that were on screen. A snapshot is ~15 MB at world size 0.40 and
  ten times that at 4.0, which buys a handful of restore points, not a history.
- **Clear Canvas, Reset, Fill, and the drawing brush.** They are picture
  operations, so under the rule above there is nothing for them to restore.
- **Deleting a saved preset, deleting an archive entry, overwriting a save.**
  These are the app's genuinely irreversible acts, and they belong to a
  confirm-or-trash design rather than to an in-memory history.
- **The camera**, and the **force/strafe field textures** — megabytes, and
  `field_handler` already caps its own snapshots.

## Why a per-frame diff is safe here

The design detects a change by comparing the live state against the last
committed snapshot once per frame. That is only viable because three things
that could have made it fire constantly do not:

- **Sweeps and jitter are computed in the shader** from the slider value and
  the sweep mode. `calculate_setting()` reads `SimState`; nothing writes back.
- **Audio modulation returns a new instance.** `AudioRuntime.update()` builds
  `replace(ui_state.sim, **moved)` and hands that to `sim.apply_state`. The
  user's `ui_state.sim` is never written, which is what makes the authoritative
  state clean enough to diff.
- **`ui.get_state()` returns the same `UIState` object, mutated in place.** The
  service therefore holds its own deep copy of the last committed snapshot;
  that is the design rather than a caveat, and comparing against `ui_state`
  itself would compare an object to itself and never fire.

The consequence: the authoritative user state changes only when a human, a
load, or a command changes it. That is exactly the signal undo wants.

## Architecture

### `services/undo_history.py`

One service, owned by `App`, holding a linear list of snapshots and a cursor.

```
capture(ui_state, rule, layout) -> Snapshot     # build, do not commit
commit(snapshot, label)                         # push, truncate any redo tail
undo() -> Snapshot | None
redo() -> Snapshot | None
jump(index) -> Snapshot
```

A `Snapshot` is a plain dict of `{container_name: {field: value}}` plus
`rule` (a copied array), `brain_signature` and `brain_settings`. Values are
deep-copied on capture, because `SimState`'s sweep and jitter dicts are shared
mutable objects.

Cap: 200 steps, matching `RuleManager.MAX_HISTORY_SIZE`. In memory only —
history ends with the session. Anything worth keeping across sessions already
has two homes: `Ctrl+C` checkpoints and `File > Save`.

### What the containers declare

Each state container declares `UNDOABLE_FIELDS` beside its existing
`PERSISTED_FIELDS`. `AudioInState` sets `UNDOABLE_FIELDS = PERSISTED_FIELDS`,
because that allowlist was drawn for the same reason — most of the dataclass is
one-shot commands and view buffers, and restoring one would replay a command.

`PreferencesState` excludes every `show_*` window-visibility flag and the
physics/load-menu collapsed states: opening a panel must not become an undo
step.

### Where it runs

In `orchestrate_frame`, after `process_commands` and after the `apply_state`
calls, so a step reflects the state the frame actually ran under:

1. If `imgui.is_any_item_active()`, return. A drag or text edit is in flight and
   its intermediate values are not steps.
2. Build the current snapshot; compare with the committed one.
3. Different → derive a label, commit.

That deferral is the whole of gesture coalescing. A slider dragged across two
seconds and forty values commits once, on release, as one step.

**A baseline is committed at startup**, once the preferences and the initial
config are loaded. Without it the first edit of a session has nothing behind
it, and the first `Ctrl+Z` would find an empty list.

**Applying a step must re-baseline the committed snapshot to that step.**
Restoration writes the very state the diff watches, so a service that did not
re-baseline would see its own undo as a fresh change on the next frame and
commit it — every undo appending a step, and the history growing in the
direction it was asked to shrink.

**A hover preview must be refused a recording outright, and ordering does not
achieve that.** Capturing before the preview only defers the preview's write
to the *next* frame's capture, which commits it; the commit shifts the rows
under the pointer, so a different row previews and commits in turn, and the
history fills in seconds. The capture returns early while a preview is held,
which also covers the frame it is handed back on. Applying a step clears that
held state, or the pointer leaving afterwards restores the pre-hover state and
silently undoes the click.

### A prerequisite on the audio branch: `Mapping` needs a stable id

`services/audio_mapping.modulate` keys per-mapping shaper state by
`states.setdefault(id(m), ShaperState())`, and `AudioRuntime._states` is a bare
`dict[int, object]` that is never pruned. **`id()` is the address, and CPython
hands the same address straight back** — 1999 collisions over 2000
create/delete cycles of a `Mapping`, measured 2026-08-14.

That breaks undo: a snapshot holds deep copies, a deep copy has a new `id()`,
so restoring one mints fresh `ShaperState`s and every envelope re-attacks and
every LFO jumps phase. The same applies to a hover preview.

It is **also a defect independent of undo**. `_render_audio_row` calls
`mappings.remove(existing)` and nothing prunes `_states`, so the orphaned entry
survives and the next mapping added can be allocated at the freed address and
inherit the deleted row's envelope and LFO phase — starting mid-attack for no
visible reason. Other allocations intervene in the real app, so it is
probabilistic rather than certain. `shaped[id(m)]` aliases the same way, which
can draw one row's trace in another row's drawer for a frame.

One change closes both: give `Mapping` a `uid` assigned at construction, key
`_states` and `shaped` by `m.uid`, and prune `_states` to live uids each frame.

Two details, both checked rather than assumed. `uid` must take part in
`__eq__` — with `compare=False`, deleting a row and adding an identical one
compares equal and the diff never records it, a hole in the coverage this
design exists to promise. And that does not break `mappings.remove(existing)`,
which passes the list's own object, so identity implies equality. `uid` is
in-session only and stays out of `_mapping_to_dict`, so stored rigs are
untouched and `apply_dict` mints fresh ones on load.

**Sequencing.** This lands on `audio-reactive` before it merges. Until it does,
`AudioInState` declares no `UNDOABLE_FIELDS` and audio is simply outside the
history — undo of physics and preferences does not depend on it.

### Cost

Measured 2026-08-14 against the real dataclasses (35 undoable `SimState`
fields, 43 `PreferencesState` fields, a 580-float rule — the deepest reachable
MLP stack):

| | cost | of a 60 fps frame |
|---|---|---|
| compare, every frame | 6.9 us | 0.04% |
| capture, on a change | 47.8 us | 0.29% |
| memory | 4.8 KB/step | 200 steps = 0.9 MB |

**The per-frame path allocates nothing.** It compares the declared fields
against the committed snapshot and returns; the deep copy runs only on the
frame something differs, which the drag deferral holds to at most once per
gesture. The 6.9 us is the worst case rather than the average — the compare
short-circuits on the first difference, so the figure is the steady state where
everything matches and all 78 fields plus the rule array are walked. Nothing on
either path touches the GPU.

`AudioInState.mappings` is a list of dataclasses and is **not** in these
figures; measure the capture again when that worktree merges. It sits on the
capture path, so a large rig is paid per gesture and never per frame.

### Labels

Derived from what differs: one field → its UI label ("Sensor Gain"); several
within one container → the container ("Physics", "Render"); the rule → "Rule";
the signature → "Brain: mlp-n16-a0". A call site that knows better may call
`undo.tag("Load Karst")` before mutating, and the next commit takes that name.

**A tag is advisory and never a trigger.** Detection stays automatic, so a
forgotten tag costs a readable name and never coverage. This is the point of
the whole design: the alternative — an explicit push at every mutation site —
means roughly sixty one-shot flags plus every widget plus everything added
later, and a field that silently is not undoable fails no test. That defect
class has shipped in this repo twice already.

## Applying a step

Restoration is not a plain `setattr` loop. Three orderings are load-bearing,
and each is already established elsewhere in the app:

- **Layout before rule.** `sim.apply_rule` measures a rule against the live
  layout and *silently refuses* a mismatch, so a step whose rule was decoded
  under another brain must restore the layout first. The archive preview path
  encodes this ordering already.
- **Through the parameter locks.** Restoration goes through
  `_apply_config_with_locks`, which snapshots locked parameters and puts them
  back afterwards. A direct write would override a lock without saying so.
- **A missing field is skipped, not defaulted.** Same rule as `load_settings`
  returning `{}`: absent means "keep what is on screen", so a snapshot taken
  before a field existed stays applicable.

An all-zero rule keeps its existing meaning of "no brain", and restoring one
must substitute a generated brain rather than upload zeros — the rule that
already governs `apply_rule`, `_write_multi_load_ssbo` and
`write_tournament_rules`.

## Ownership: undo under a tournament

Under a tournament the population belongs to its owner. `sim.apply_rule` writes
slot 0, which is tile 0 of a running grid and is overwritten by the next
generation; under Auto or Explore the optimizer owns every tile.

So undo splits along the same line `_grid_owner()` already draws for `Z` and
`G`:

- **The settings half applies** — physics, appearance, preferences, audio.
- **The brain half is skipped** while an owner exists, with a notice naming
  why.

Steps are still *recorded* throughout, so leaving the mode leaves the history
intact.

## Redo

Linear and truncating: a new edit after an undo discards the redo tail. The
panel shows those entries greyed while they are still reachable, so what an
edit is about to discard is visible before it goes.

## UI

`ui/undo_window.py`, a new mixin on `UI`, modelled on the Config Clipboard
window:

- Steps newest-first with their labels, the current position marked, redo
  entries greyed.
- **Hover previews.** Hovering applies that snapshot live and restores on
  un-hover. `_clipboard_cached_config` is the working precedent — the clipboard
  preview already caches the full config before previewing and restores it
  after. The preview must be a **continuous** field written every frame and
  diffed by `CommandHandler`, never a one-shot, or a panel that stops being
  drawn mid-hover strands the preview. That is the archive browser's rule and
  it applies here unchanged.
- Click moves the cursor to that step — exactly what repeated `Ctrl+Z` or
  `Ctrl+Shift+Z` would reach. It does not append a new entry, so jumping back
  and forth in the panel never grows the history.
- `Ctrl+Z` / `Ctrl+Shift+Z`, registered in `default_keyboard_controls.json` so
  they rebind like every other key.

Two UI caveats the panel must respect: widget labels are pushed through
`layout.push_settings_width()`, and no label may collide with an existing
widget id — an ImGui id clash silently kills the loser.

## Testing

- `tests/test_undo_history.py` — commit, undo, redo, truncation on a new edit
  after an undo, the 200-step cap, and no commit while an item is active.
- **A schema-derived classification test.** Every field of `SimState` and
  `PreferencesState` must appear in either `UNDOABLE_FIELDS` or an explicit
  `NOT_UNDOABLE` set carrying a reason. Cases come from
  `__dataclass_fields__`, never a hand-written list, so a newly added field
  fails until someone classifies it. `tests/test_brain_scales.py` derives its
  cases the same way and for the same reason: the hand-written list is what let
  the previous "declared but never read" defects through.
- **A round-trip test** — mutate every undoable field, undo, assert the state
  matches. Catches a field that is captured but never applied, which is that
  same defect wearing a different hat.
- **A cross-brain test** — undo a step whose rule was decoded under another
  layout, and assert the rule actually reaches the GPU rather than hitting
  `apply_rule`'s width guard.
- `tests/test_undo_window_render.py` — a real ImGui frame with the host sized
  taller than the panel, asserting on the labels a frame draws. A window's
  contents are clipped to the window, so an undersized host draws no vertices
  and the assertion becomes a coin flip.
