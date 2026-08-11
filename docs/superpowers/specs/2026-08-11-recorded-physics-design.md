# Recording the physics an archive entry ran under

Status: designed, deferred until `worktree-brain-modalities` merges into
`tournament-mode`.

## The problem

`Archive._phys` is `(n, 8)`, hardcoded, and those 8 columns are
`services/physics_genome.PHYSICS_PARAMS` — the optimizer's **search space**.
Anything outside it is not recorded at all: `HAZARD_RATE` never was, and `V_MAX`
is not either.

So an entry stores the phenotype but not every number that shaped it. Preview
and export (`command_handler.py:678`, `:728`) set only those 8; the rest come
from whatever preset happens to be loaded. Today this is latent rather than
live, because V Max defaults to Off and nobody has run a search with it braking.
It stops being latent the moment someone does.

Widening the 8 is not the fix. A parameter should be *recordable* without
becoming *searched* — `V_MAX` is deliberately excluded from the search space,
and adding it there would change `PHYSICS_DIM`, `z`, and every optimizer.

## Why the brain's answer does not transfer

`worktree-brain-modalities` solved the analogous problem for brains by
**partitioning the archive by layout signature** — `<archive>/fourier-n10/`,
`<archive>/gabor-n12/`, with a migration for pre-modality archives. That is
correct there: a Gabor genome and a Fourier genome are incomparable, and a
shared directory would decode one through the other's squash and score the
result, which is worse than a crash.

Physics is the opposite case. Every entry in an archive shares one physics
vocabulary, and adding `V_MAX` must not fork an archive into two halves that
cannot see each other's novelty. Physics needs to be **additive and
self-describing in one directory**.

## Design

### 1. Split searched from recorded

`services/physics_genome.py`:

- `PHYSICS_PARAMS` (8) stays exactly as it is — the search space, the `z`
  dimension, `PHYSICS_DIM`. Nothing about the optimizer changes.
- New `RECORDED_PHYSICS`: every physics field an entry carries, searched or
  not. Starts as the 8 plus `V_MAX` and `HAZARD_RATE`. A future parameter is
  one line here and nothing else.

The two lists must not be merged later "for tidiness". They answer different
questions: what may the optimizer move, and what does an entry need to be
reproduced.

### 2. A self-describing column set

`vectors.npz` gains an optional `phys_names` array of field names.
`_OPTIONAL_ARRAY_KEYS` already carries `novelty`, so the mechanism exists.

On load, map the stored columns onto the current `RECORDED_PHYSICS` **by name**:

- a name present in both — copy the column;
- a name the file lacks — fill from `SimState()`'s default for that field;
- a name the file has and the code does not — drop it.

No `phys_names` key means a pre-migration archive: the legacy 8 in
`PHYSICS_PARAMS` order. Existing archives therefore load with `V_MAX` at its
default, which is Off, which is what they actually ran under.

### 3. No positional access outside the archive

`archive.physics[i][j]` zipped against `PHYSICS_PARAMS` is what makes step 2
invisible to callers today, and it breaks the moment the two lists differ. Add
`Archive.physics_of(i) -> dict[str, float]` and route the three call sites
through it: `_show_archive_preview`, `_export_archive_entry`, and
`ImgepDriver._phys_dict`. The raw `physics` array stays for anything that wants
the packed form.

## What this does not do

- It does not make `V_MAX` searchable. It stays out of `PHYSICS_PARAMS` until
  hand-testing says it earns a place there.
- It does not version the archive format. A missing optional key is the whole
  compatibility story, the same way `novelty` was added.
- It does not touch thumbnails, embeddings, or the index.

## Order of work

After the merge, because steps 2 and 3 land in `Archive.__init__`, `_grow`,
`load_from_store` and `ArchiveStore.__init__` — the four functions
`worktree-brain-modalities` rewrote. Building on the merged shape also puts the
two extension points side by side: layout signature for brains, named columns
for physics.

Until it lands, a search run with V Max braking produces archive entries whose
look depends on a number the archive does not store.
