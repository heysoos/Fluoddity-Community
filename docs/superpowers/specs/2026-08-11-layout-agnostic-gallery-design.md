# One archive, many brains

## The premise

An archive is a library of **pictures** with a genome attached, and only the
genome is brain-specific. Everything the archive actually does with an entry
runs on its CLIP embedding:

| brain-agnostic | brain-specific |
|---|---|
| embeddings, descriptors, liveness | the genome floats |
| novelty, kNN, separation, admission, pruning | re-encoding a phenotype (`_parent_z`) |
| the map, the record book, text and latent goals | decode / `sim.apply_rule` |
| | the optimizer's search dimension |

`rescore_all` is `knn_novelty(self.embeddings, ...)` and `descriptor()` is a
CLIP centroid — neither reads a brain. So splitting the archive by brain splits
it along an axis nothing in it cares about, except four things.

## The defect this fixes

Because an archive is keyed `(name, brain)`, the gallery shows one signature
directory while the picker counts them all. Measured over the eleven archives
on this machine: under `mlp-n16-a0` the picker advertises **39015 entries and
the gallery shows 0**, and switching archives does not help, because
`migrate_to_signature_dir` only ever creates `fourier-n10` and every
pre-existing archive lives there. `BrainState.modality` is persisted nowhere,
so a relaunch runs Fourier and a Gabor archive opens empty.

An archive becomes one archive again: **one per NAME**, holding entries of
every brain.

## Storage

```
<archive>/goals.json          <- per archive, moves UP
<archive>/settings.json       <- per archive, moves UP
<archive>/runs/               <- per archive, moves UP
<archive>/<signature>/index.jsonl
<archive>/<signature>/vectors.npz
<archive>/<signature>/thumbs/
```

Entries stay filed under their own signature, because a bare float vector needs
the directory to say what it means. Everything that is a property of the
archive rather than of a genome moves up to the name level.

This corrects the merge, not just the split. `_LEGACY_MEMBERS` was widened at
`c6e76ef` to pull `settings.json` and `runs` down into the signature directory,
which is one level too deep: the Explore settings and a run's physics belong to
the archive, not to one of its brains. Same migration machinery, opposite
direction.

## Archive internals

Pooled, unchanged in shape:

- `_emb` `(n, 512)`, `_phys` `(n, PHYSICS_DIM)`, `entries`, novelty

Per layout:

- `_brains: dict[str, np.ndarray]` — signature -> `(m, layout.length)`
- `_brow: np.ndarray (n,) int32` — each entry's row within its own layout array
- `ArchiveEntry.layout: str` — the signature, set at load from the directory,
  not stored in `index.jsonl` (the directory already says it, and adding a
  column to an append-only file older builds also write is a migration for
  nothing)

`brains` as a single array stops existing; `brain_at(i)` and `layout_at(i)`
replace it. Four non-test call sites read it today — `imgep_driver._parent_z`
and three in `command_handler` — and every one already wants the entry's own
layout, which it currently has to guess.

`flush_vectors` fans out: one `vectors.npz` per signature, holding that
layout's rows. `load_from_store` reads every signature directory and
concatenates.

## Ids stay per layout; the UI addresses rows

Two layouts each hold an entry 0, and on this machine no archive yet has two
non-empty layouts — so the collision is latent, not yet realised, and will be
the moment this ships.

Ids are **not** renumbered. They stay per-layout exactly as they are on disk,
`_next_id` becomes per-layout, and the six `ArchiveState` fields that address
entries by bare int — `preview_entry_id`, `load_entry_id`, `seed_entry_id`,
`delete_entry_id`, `selected_entry_id`, `save_arg` — become **row indices**
into the pooled archive. No re-id migration, and the ambiguity cannot arise.
Safe to change: none of the six is in `PERSISTED_FIELDS`.

## What pooling changes about behaviour

**Separation now looks across brains.** A Gabor creature within
`min_separation` of a stored Fourier one is refused. That is the point of one
archive — they look the same, so the archive holds one of them — but it is a
real change: two brains that converge on similar pictures will stop being
stored twice.

**Pruning ranks across brains.** `prune_to_capacity` evicts the globally least
novel, so a brain that produces near-duplicates loses entries to one that does
not. Capacity is now shared rather than per-brain.

**Novelty is one pool**, which is what makes both of the above meaningful.
Stored per-layout novelty values are stale under pooling;
`load_from_store` already calls `rescore_all()` unconditionally, so this
corrects itself on first open.

## What stays per layout

Only what decodes:

- **Parent sampling.** The optimizer searches one width, so `_parent_z` and
  `sample_by_novelty`'s candidate set filter to the running layout. Novelty
  *ranking* stays pooled — a Fourier creature is genuinely novel against Gabor
  ones — but a parent has to be decodable.
- **Seeds.** `nearest()` and `ImgepDriver._seed_index` filter likewise.
- **Preview, export, seed-a-run** on an entry of another brain. Refused with a
  notice naming the brain it needs, except **click-to-load**, which switches
  the brain and then loads: the handoff at `7b1051f` already does exactly this,
  driven from the config loader instead of the browser.

Hover-preview is refused rather than switching, because it fires every frame
from pointer position and a teardown there would be violent.

## Thumbnails must key on the signature

`ThumbCache` is keyed by bare filename and resolves through one store. Both
layouts have a `000000.jpg`, so a pooled gallery through the current cache
shows **the wrong pictures**, silently. This is the existing "entry ids restart
at 0 in every archive, so `ThumbCache` must be released on a switch" caveat one
level down — and releasing no longer fixes it, because both layouts are now
live at once.

The key becomes `f"{signature}/{filename}"` and the loader resolves the store
from the signature.

## Migration

One pass, additive, on open:

1. A pre-modality archive still migrates into `fourier-n10` as it does now.
2. `goals.json`, `settings.json` and `runs/` move UP from any signature
   directory to the archive root. Where two layouts each have one, the
   newest wins and the others are left in place with a printed line — merging
   two settings files silently is worse than picking one loudly.

An archive with a single layout loads to exactly the entries, order and row
indices it has today.

## Testing

- Two signature directories, each with an entry id 0: rows resolve to distinct
  entries, and the thumb cache returns distinct textures. **This is the
  collision test and it fails today.**
- Pooled novelty: an entry's novelty changes when a look-alike from another
  brain is admitted.
- Separation refuses a cross-brain near-duplicate.
- `prune_to_capacity` can evict an entry of a brain that is not running.
- Parent sampling and `_seed_index` never return a row of another layout, at
  any archive composition — including one where the running layout is a
  minority.
- `brain_at(i)` round-trips through `encode(..., layout_at(i))` for every row.
- A foreign row refuses preview, seed and export, naming the brain; its
  click-to-load switches and lands the creature.
- Single-layout archives: entries, order and row indices identical to today.
- `flush_vectors` writes one npz per signature and reloads byte-identical.

## Open, and deliberately not decided here

**Should a search be able to span brains?** Nothing above enables it — one
CMA-ES searches one width. A multi-brain *search* would need either a per-brain
population under one archive or a portfolio over modalities, and that is a
separate design.

**`BrainState.modality` is still not persisted**, so a relaunch runs Fourier.
Under a pooled archive that no longer empties the gallery, so it stops being
urgent — but it still means the app forgets which brain you were evolving.
