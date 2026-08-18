# Searching brain layouts

**Date:** 2026-08-17
**Status:** design approved, not yet implemented

## Problem

The search cannot change brain. `GenomeSpec` fixes the layout at construction
and `z` is one vector of one width, so a run explores the creatures reachable
under whichever layout the Brain window happened to be showing when it started.
There are exactly three callers of `apply_brain_layout` — the Brain window, an
undo restore, and click-to-adopt — and all three are the user.

An archive that holds several layouts therefore holds them because someone
switched by hand. `debug15-physicssearch` holds 1230 `fourier-n10` entries and
489 `mlp-n16.8.8-a0.0.0`; every sampler filters through `native_rows()`, so
while Fourier runs the MLP entries browse and rank but cannot be bred from.
They are ballast.

Two things follow. The obvious one is that the shape of the brain — how many
centres, how deep the stack, which activation — is a parameter nobody is
searching, and it is plausibly the parameter that decides what is reachable at
all. The less obvious one is that switching is *expensive by construction*, so
even a manual exploration of layouts is discouraged by the tool. Worse, the
brain an archive was being built under is recorded nowhere, so closing the app
and coming back opens that archive under an unrelated brain — see section 1b.

Measured on `debug15-physicssearch` (1650 pooled entries):

| running layout | pooled | native | load | `Projection.fit` | `rescore_all` |
|---|---|---|---|---|---|
| `fourier-n10` | 1650 | 1230 | 67 ms | 72 ms | 25 ms |
| `mlp-n16.8.8-a0.0.0` | 1650 | 420 | 51 ms | 72 ms | 26 ms |

**The pooled set is identical.** A layout change inside one archive changes only
which rows are native, yet `_apply_brain_layout` flushes, closes and reloads
every signature directory from disk, refits a projection over embeddings that
did not move, and drops a thumbnail cache whose keys are already
signature-qualified. At 1650 entries that is ~140 ms of pure waste, and
`rescore_all` is O(n^2) — see the `rescore_all()` caveat in `CLAUDE.md` for what
it costs at capacity.

## Scope

In scope:

- A cheap layout switch inside one archive.
- An archive remembering the brain it was last searched under.
- Layout mutation operators, bounded, for every modality.
- Genome transfer across a layout change within one modality.
- A layout move as part of an ordinary expedition, with keep-or-revert.
- Bounds and a readout in the Explore tab, persisted per archive.
- A ledger of the layouts a run walked through.

Out of scope, deliberately:

- **A heterogeneous grid.** `BRAIN_MODALITY`, `BRAIN_LEN`, `BRAIN_SHAPE`,
  `BRAIN_DEPTH` and `BRAIN_LAYERS` are uniforms, so one dispatch is one layout.
  Per-slot layout would move them into a buffer *and* make `MAX_MLP_WIDTH` the
  maximum across the grid — a compile-time define whose bucket, per the
  variable-depth MLP caveat, is paid by every brain in the build and not only
  by the wide stack. One layout per generation needs no shader change at all.
  The design below does not assume either, so this stays open.
- **Migrating stored entries between layouts.** An entry records the layout it
  was authored under and replays under that one. Nothing is rewritten.
- **Searching decode scales.** `w_scale` and `b_scale` are already continuous
  and already reachable by an audio rig; they are not structural and do not
  belong in a structural search.

## 0. The browser prerequisite

`ast.enabled` is set by the Explore *tab being selected*
(`ui/archive_window.py`), not by a search running. Both the Live preview
checkbox and `_handle_archive_preview` gate on it, so with the Tournament
window open on Explore, clicking an entry of another brain does nothing at all
and says nothing. Measured against the real handler over a mixed archive:

```
explore_tab=False live_preview=True  | after hover=fourier  after click=gabor    switched=['gabor-n12']
explore_tab=False live_preview=False | after hover=fourier  after click=fourier  switched=[]
explore_tab=True  live_preview=True  | after hover=fourier  after click=fourier  switched=[]
explore_tab=True  live_preview=False | after hover=fourier  after click=fourier  switched=[]
```

Hovering never moves the Brain window and never should: a hover BORROWS a
layout and `_handle_brain_layout` is suppressed for its duration, because the
window's own layout would otherwise tear down and rebuild the archive once a
frame. Only a click switches.

The fix is to gate on `running` rather than on `enabled`, so the browser works
whenever the search is idle, and to say why in place of the greyed control when
it genuinely is running. A control that cannot do what it offers is worse than
no control — the same rule the Explore tab's permanently-disabled Encoder combo
already follows.

This is independent of everything below and worth doing first.

## 1. `Archive.retarget(layout)`

Every layout's entries are already in memory: `load_from_store` reads every
signature directory. A layout change inside one archive changes only which rows
are NATIVE. The embeddings, the novelty column, the rejects ring, the
projection and the thumbnail cache are all about PICTURES, and a picture does
not stop being one because a different brain is running.

`retarget` moves `self.layout`, takes `self.store` out of the `_stores` map the
archive already keeps — building one only for a signature never seen —
registers `_widths[sig]`, and calls the existing `_widen()` when the incoming
layout is wider. It bumps `revision`, because what a viewer draws changes.

It must **not** clear `_novelty_clean`. Nothing was admitted and nothing was
removed, so every entry's novelty was still scored against exactly the set now
held; clearing it would buy a full O(n^2) rescore per layout move for no
information. Only `_add` and `_remove` may clear that flag.

`_apply_brain_layout` gains a fast path taken when the archive DIRECTORY is
unchanged: retarget, `realloc_brain_buffers`, `_refresh_driver_specs`. No
flush, no reload, no `Projection.fit`, no thumbnail drop. `maybe_flush` already
writes per-layout, so nothing is at risk from not flushing at the switch.

The slow path stays exactly as it is for an archive NAME change, which really
does invalidate the thumbnails (entry ids restart at 0 in every archive) and
really does need the settings saved into the outgoing folder.

`settings.json`, `goals.json`, `settings_history.jsonl`, `runs/` and
`encoder.json` all live at the archive ROOT and are shared across layouts; only
`index.jsonl`, `vectors.npz` and `thumbs/` are per-signature. The comment in
`main.py` claiming settings live in the signature directory is wrong and should
go with this change. It is why layout bounds can be one setting for an archive
rather than one per layout.

## 1b. An archive remembers the brain it was being searched under

The app remembers which archive you were working in and not which brain you
were working on in it. `PreferencesState.archive_name` is persisted;
`BrainState` is never serialised at all, `PreferencesState` has no brain field,
and the archive's own `settings.json` has no layout key. `BrainState.modality`
defaults to `"fourier"`, and `_build_archive_set` takes `sim.brain_layout`
unconditionally — so an archive opens under whatever brain happens to be live.

Measured on a fresh launch's default layout:

```
debug14-deepmlp-small-2: opens as fourier-n10 | pooled=6708 native=0    -> bootstrap
debug15-physicssearch:   opens as fourier-n10 | pooled=6801 native=1230 -> expansion
```

`debug14` holds 6708 entries and every one of them is `mlp-n16.8.8-a0.0.0`.
Opened under the default brain it has ZERO native rows, so pressing Start
throws `seed_n` generations of random Fourier genomes at an archive that is
entirely MLP — silently, because the regime readout is telling the truth and
the mixed-archive note is the only thing on screen that hints at why.

This is not a resume bug. Nothing on the resume path is at fault; the layout is
simply never recorded, so every path that opens an archive inherits an
unrelated brain.

The checkpoint carries `brain_layout_signature`, but only as a GUARD that
raises on mismatch — it never restores — and Explore's Start has no checkpoint
at all.

**The layout an archive was last searched under belongs to the archive.**
`archive_name` lives in preferences because it names WHICH archive; which brain
is a fact ABOUT one, and the settings that suit a 20000-entry MLP archive are
not the ones that suit an empty Fourier archive, which is the same reasoning
that put the Explore settings there in the first place. So: a
`layout_signature` field in `ArchiveState`, named in `PERSISTED_FIELDS`, written
by the `save_settings` call that already runs on a switch and on quitting.

Restoring it needs three things this codebase already demands elsewhere:

- **It must move the Brain WINDOW as well as the sim.** `_handle_brain_layout`
  applies `ui_state.brain` every frame, so a restore that writes only
  `sim.brain_layout` is undone by the next frame — an apparent success that
  reverts. `_put_brain_window` is what `_restore_snapshot_brain` and
  `_adopt_foreign_entry` already use for this.
- **A signature this build cannot rebuild keeps the current layout and says
  so.** `layout_from_signature` already answers `None`, and
  `_adopt_foreign_entry` already has the warning to copy. Guessing a plausible
  layout of the wrong width is the one outcome worse than refusing.
- **A missing key keeps the current behaviour exactly**, so every archive
  written before this opens untouched and no migration runs — the rule
  `apply_settings` already follows for every other field.

`retarget` is what makes this cheap: the archive can be opened however it opens
and then re-pointed, with no ordering constraint between building the archive
set and applying the layout, no reload and no rescore. Without section 1 this
fix would cost a full teardown on every archive open.

The change is PRINTED like any other layout change, because a search that
silently adopts a different brain is exactly what the existing rule is about.

Under layout search this field also carries the walk forward: the layout a run
ends on is the one it resumes from, so a stack the search grew over a long run
is not thrown away by closing the app.

## 2. Layout mutation

A registry hook with a generic implementation, the pattern `signature_of`,
`settings_of`, `unit_count` and `shader_defines` already follow.

**Generic:** perturb one setting of `STRUCTURAL_KINDS` within bounds. That
covers `fourier-n10 -> fourier-n11`, `gabor-n7 -> gabor-n8` and Lenia without
any modality opting in, because for those three the structure IS one integer.

**MLP override:** a layer stack does not fit "nudge one int", so `mlp` declares
its own operators:

- grow one hidden layer by N units
- add a hidden layer
- shrink one hidden layer by N units, or drop it
- change one layer's activation

**N is 1 by default**, and small when it is not. The point of the move is that
the child is its parent plus a little room, which is what makes the novelty it
earns attributable; a large jump is a restart wearing a growth move's name.

Every move is **proposed, rebuilt, and compared** — the round-trip discipline
`layout_from_signature` already uses. This is not ceremony:
`_shape_from_layers` CLAMPS rather than raises, deliberately, because its input
may be a config written by a build with different limits. So a move that hits
`MAX_DEPTH`, `MAX_WIDTH`, `MAX_BRAIN_FLOATS` or the user's own bounds comes
back as a DIFFERENT layout than the one proposed, silently. Comparing the
rebuilt layout against the proposal turns that into a rejected move instead of
a move that did something else.

The operator also skips a `(parent, child)` pair the ledger records as tried
and reverted. The ban is per-parent, so the same child stays reachable from
elsewhere in layout space.

When every move a parent can reach is bounded out or already banned, the
operator returns nothing and **the expedition proceeds without a layout move**.
It does not fall back to an unbanned-but-worse move and it does not skip the
expedition: a cadence interval spent on an ordinary expedition is worth more
than one spent on a move already known to produce nothing.

## 3. Genome transfer

Also a registry hook, and this is what makes a layout move continuous rather
than a restart.

**Generic:** for the unit-structured modalities, copy `min(n_old, n_new)` whole
units and draw the remainder. `unit_floats()` already defines the unit and
`crossover` already relies on it, so `fourier-n10 -> fourier-n11` keeps ten
centres and gains one.

**MLP override:** repack the weight matrices. Growing hidden layer *l* adds a
row to `W_l`, an entry to `b_l`, and a column to every row of `W_{l+1}` — which
is `W_out` when *l* is last, and that one is stored OUTPUT-MAJOR, so the new
column is four strided writes and not a contiguous append. Offsets come from
`layer_spans`, which is the one definition `mlp.glsl` is checked against.

New **outgoing** weights start at zero, so the child's phenotype is
bit-identical to its parent at birth and any novelty it earns is earned rather
than an artefact of a random restart. New **incoming** weights are drawn, so
the unit has something to contribute the moment the search moves its output
weight; a unit zeroed on both sides is inert in a way `sigma` takes generations
to undo. Zero decodes from `z = 0` exactly, so the transferred genome
re-encodes without clipping.

**Only a cross-MODALITY jump has no transfer.** Within a modality every move
carries the parent forward. That is what makes searching all four modalities
affordable: a jump is a restart, but a rare and budgeted one, not the default.

## 4. A layout move rides on an ordinary expedition

No new regime and no new goal kind. When `start_expedition()` fires and layout
search is on, with probability `layout_move_chance`:

1. **Draw the goal normally.** `_draw_goal` is untouched — novelty, latent,
   text, in their existing shares.
2. **Pick the seed the goal would have picked anyway**, through `_seed_index`,
   which already filters to native rows.
3. **Mutate the layout** under bounds, rejecting a move the ledger has
   already tried and reverted from this parent.
4. **Transfer** that seed's decoded genome into the candidate layout.
5. **Request the switch.** The driver sets a one-shot `requested_layout`; the
   frame loop reads it and runs `_apply_brain_layout` on the fast path, because
   `App` owns the archive and the driver must not. `set_spec` then fires with
   the new spec, which already ends the outgoing expedition when the space
   moved.
6. **Start the expedition at an explicit `x0`.**

`_x0_index` is read in exactly one place, to compute `x0`. So
`start_expedition_with` splits in two: a thin wrapper that resolves a native
row to an `x0`, and `start_expedition_at(embedding, kind, text, x0)` that takes
one directly. A transferred genome has no archive row, so `_x0_index` is `None`
and the trace names the MOVE rather than an entry.

**A cross-modality jump takes the same path with no transfer**, seeding `x0`
from a `sigma0`-scale random draw. Staying inside an expedition is the whole
point: `regime` answers `"expedition"` on `_remaining > 0 and _goal is not
None`, so a layout with zero native entries gets a bounded budget. Without it
the jump lands in `bootstrap`, which on this archive's settings means 1022
random genomes before expansion is reachable — a layout search that spends its
entire life bootstrapping.

## 5. Keep or revert

`tell()` already counts `admitted`. Accumulate it while a layout expedition is
live, and at `end_expedition`:

- **admitted >= 1 -> keep.** The layout now has native entries, so ordinary
  expansion breeds from it next generation with no special case anywhere.
- **admitted 0 -> revert** to the parent layout, through a second retarget.

The archive is already the judge and this adds no second opinion: admission
means finite, viable, alive, and separated from everything stored. A layout
that cannot produce one such tile in a whole expedition has answered the
question.

The alternative — comparing admission RATE against the parent — was rejected.
A generation's tiles are not independent draws (they share one CMA-ES
population), so they clear or miss any bar together, which is the same reason
the adaptive admission threshold was removed.

## 6. The ledger

`<archive>/layouts.jsonl`, append-only at the archive root beside
`settings_history.jsonl` and for the same reason: a run that changes brain
silently redirects where its results are filed, and nothing else records that
it happened. One row per move — parent signature, child signature, which
operator, generations, entries admitted, kept or reverted, generation number.

Two readers. The mutation operator skips a reverted `(parent, child)` pair, so
the search does not spend its cadence re-proposing a move that produced
nothing. The Explore tab shows the walk.

## 7. Settings and UI

New `ArchiveState` fields, each named in `PERSISTED_FIELDS` — that list is an
explicit allowlist and a field not named there simply does not persist:

| field | what it bounds |
|---|---|
| `layout_search` | the checkbox, beside "Search Physics Too" |
| `layout_move_chance` | how often an expedition also moves the layout |
| `layout_max_depth`, `layout_max_width` | the stack, under `MAX_DEPTH` / `MAX_WIDTH` |
| `layout_max_floats` | the cost ceiling, under `MAX_BRAIN_FLOATS` |
| `layout_modalities` | comma-separated keys of the modalities in play |

Defaults: `layout_search` off, so opening the app never starts changing brain
under anyone; `layout_move_chance` low enough that most expeditions are
ordinary ones; the three bounds seeded from the LIVE layout rather than from
the hard maxima, so switching the feature on explores around the brain that is
already running instead of immediately reaching for the expensive end of the
space. `layout_modalities` starts as the running modality alone — a
cross-modality jump is a restart, and opting into one should be a decision.

`layout_modalities` is one string rather than four booleans so that a fifth
modality needs no new field, and its checkboxes are derived from `REGISTRY`
rather than a hand-written list — a second list of modalities is the
declared-but-never-read defect this codebase has already shipped twice.

The width bound must **name the scratch bucket it implies**. `MAX_MLP_WIDTH` is
a compile-time define and its cost is paid by every brain in the build, not
only by the wide stack, so a bound that quietly crosses from one bucket to the
next is a cost the user should see at the moment they set it.

## 8. What deliberately does not change

- **`native_rows()` stays strict everywhere.** Parents and seeds still come
  only from the running layout. Transfer is the single crossing, it is
  explicit, and it reaches the optimizer through `x0` — never through the
  parent sampler. A foreign row reaching a modality's reshape is the defect
  that killed a 2.5-hour overnight run inside Fourier's `encode`, and it must
  not reopen here.
- **A layout move does not mint a new `run_id`.** The run config records the
  PHYSICS a run was carried out under, a layout move does not touch physics,
  and `save_run_config` refuses to overwrite. Per-entry `layout` in
  `index.jsonl` is already the authority for which brain replays an entry; the
  run config's `brain_layout` names only what the run started under.
- **Admission, separation and liveness are untouched.** The archive judges a
  layout exactly as it judges everything else.

## 9. Verification

- **`retarget` is indistinguishable from the teardown it replaces.** Retarget,
  versus release-and-reload from disk, must give the same entries, embeddings,
  novelty column and native set. This is the load-bearing test: every
  performance claim above rests on the fast path being honest.
- **Transfer is phenotype-preserving**, checked as a PURE FUNCTION over a fixed
  input grid. A trajectory diff cannot resolve this at all — see the
  reproducibility caveat in `CLAUDE.md` for why, and for the `purefn` pattern
  that does work.
- **Every operator respects every bound**, with cases derived from `REGISTRY`
  and from the settings rather than from a hand-written list.
- **A modality jump enters as an expedition, not as bootstrap**, and produces a
  genome of the running layout's width.
- **A reverted move leaves the native count where it started** and is not
  immediately re-proposed.
- **The browser fix**: a click switches brain whenever the search is idle,
  including with the Explore tab open.
- **An archive reopens under the brain it was last searched under**, driven
  through the real open path rather than a model of it — including the three
  cases that must NOT switch: an archive with no recorded signature, one
  recording a signature this build cannot rebuild, and a restore that the next
  frame's `_handle_brain_layout` would otherwise undo.

## Sequencing

Sections 0, 1 and 1b are independently useful, independently testable, and
depend on nothing else here.

Section 0 is a UI gate with no consequences elsewhere. Section 1 is a latent
performance fix for switching brain by hand, worth having whether or not layout
search ever ships. Section 1b is a correctness fix that is live TODAY — an
archive of 6708 MLP entries currently bootstraps Fourier into itself on Start —
and it depends on section 1 to be cheap.

Build those three first; the rest rests on them.
