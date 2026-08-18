# A UMAP atlas for the archive map, with thumbnails

Date: 2026-08-18

## The problem

The map is a 2-D PCA of the descriptors. It holds 46% of the variance at every
archive size, and two linear axes through a CLIP space is a blob by
construction — it does not separate or cluster in a way that is useful to look
at. Filtering, the colour ramps and the density heatmap all address CROWDING,
which is a different complaint and does not fix this one.

Second, the map draws dots. A dot says where something is but not what it is,
so reading the atlas means hovering entries one at a time.

## What this adds

1. A pluggable map layout, with UMAP alongside PCA, fitted off the frame loop
   and cached per archive.
2. A thumbnail layer over whatever layout is in force: one representative
   picture per screen cell, so the map can be read at a glance.

The two halves are independent. The thumbnail layer consumes 2-D positions and
does not care which engine produced them.

## 1. The layout interface

New `services/map_layout.py`. Two implementations behind one shape, which is
the shape `Projection` already has:

```python
fitted: bool
version: int                  # bumped per successful fit; views cache on it
fit(embeddings, ids) -> bool
transform(embeddings, ids) -> (n, 2) float32
```

- `PcaLayout` wraps the existing `Projection`. Behaviour is unchanged, and it
  is the default and the fallback.
- `UmapLayout` is new.

`services/archive_projection.py` is NOT touched. The search holds its own
`Projection(LATENT_DIMS)` for whitening, and this work must not reach it.

`ids` are `(layout signature, entry id)` pairs. An id is unique only inside one
brain's directory, and the map pools every layout — the same reason
`thumb_key` is built that way.

## 2. The UMAP engine

**Lazy import.** `import umap` happens inside `fit()`. Absent means
`UmapLayout.available` is False, the Layout combo offers PCA only with a note,
and nothing else changes. The rule `onnxruntime`, `tokenizers` and `cmaes`
already follow.

**PCA pre-reduction.** UMAP fits on the top 40 PCA components, not raw 512-d.
Our PCA is an eigendecomposition of the dim x dim covariance, so it costs the
same at 200 entries as at 20,000. The reduction is stored with the layout:
anything placed later has to go through the same one.

**No `umap.transform`, and no pickled reducer.** UMAP lays out a SNAPSHOT.
Entries admitted after a fit are placed by kNN interpolation — nearest
neighbours in the full embedding space through the existing blocked
`knn_distances`, positioned at the mean of those neighbours' 2-D locations.
This buys three things: the archive cache is only numbers, so there is no
reducer pickle to break across a library upgrade; the slow approximate
`transform()` is never called; and placement degrades smoothly instead of
failing. The cost is that placement drifts from what a real refit would give,
which is what the staleness readout and the refit trigger exist to bound.

**Stability.** A refit passes `init=` the previous layout — existing entries at
their current position, new entries at their kNN placement — and the result is
Procrustes-aligned to the previous layout before it is published. Without both,
a refit rearranges the map the user had learned, which is the reason UMAP was
rejected the first time. PCA solves the same problem one level down with its
eigenvector sign alignment.

**Off the frame loop.** A one-thread executor, the pattern
`AutoTournamentService` uses for `driver.precompute`. The map keeps drawing the
current layout and swaps when the future lands. The fit is handed a **copy** of
the embeddings: the archive mutates on the main thread, the same reason the
CLIP path copies its frame buffer.

## 3. The cache

`<archive>/map_layout.npz`, beside `settings.json` and `goals.json`:

| key | what |
|---|---|
| `pos` | (n, 2) float32 positions |
| `sigs`, `ids` | the (layout signature, entry id) each row belongs to |
| `encoder` | the archive's encoder key at fit time |
| `reduction`, `reduction_mean` | the PCA pre-reduction |
| `fitted_n` | archive size when the fit ran |
| `engine` | "umap" |

A cache whose `encoder` differs from the archive's is discarded rather than
trusted — at equal width a foreign vector is silently wrong, which is why
`Archive.load_from_store` already refuses one. A cache that fails to load for
any other reason is discarded too: it is a view, and the fallback is a refit.

Written when a fit completes and when the archive is let go. Reading it is what
makes reopening an archive instant.

## 4. When it refits

Exactly three triggers, and nothing else:

1. On open, when there is no usable cached layout.
2. When `(size - fitted_n) / max(1, fitted_n) >= 0.25`.
3. When **Relayout** is pressed.

Never while a fit is already in flight. The readout carries the engine and the
staleness — `UMAP · fitted at 4,800 of 6,120`, or `fitting…` — so the map never
moves without something on screen saying why.

## 5. The thumbnail atlas

Independent of the layout, and orthogonal to Colour and Draw — the existing
caveat requires that a new axis not silently override those.

`services/map_view.py` gains one function beside `cell_means` and
`cell_majority`:

```python
cell_argmax(flat, values, ncells) -> (winner index per cell, counts)
```

The drawing rule: bin the on-screen points with `bin_points` at
`cell_px = map_thumb_px`, take each occupied cell's highest-novelty entry as
its representative, and draw that entry's thumbnail in the cell. Everything
else stays a dot.

Two properties fall out of that and both matter:

- **The count is bounded by the VIEWPORT, not the archive.** At 490x320 with
  32px cells it is at most 150 thumbnails whatever the archive holds. The atlas
  calls `ThumbCache.reserve` with the occupied-cell count before drawing any of
  them, for the reason the gallery does.
- **Zoom subdivides for free.** `bin_points` bins in SCREEN space, so zooming
  in splits cells and reveals more entries with no new interaction.

## 6. State

New fields in `state/archive_state.py`, all in `PERSISTED_FIELDS`:

```python
map_layout: str = "pca"       # pca | umap
map_thumbs: bool = False      # draw the thumbnail atlas
map_thumb_px: int = 32        # 16..64
```

`map_layout` defaults to `pca`, so an archive opened by a build without
`umap-learn` behaves exactly as it does today.

## 7. UI

In the map toolbar: a **Layout** combo, a **Relayout** button, the staleness
readout, a **Thumbnails** checkbox and its size slider. When `umap-learn` is
missing the combo is PCA-only and says so, which is the same shape as the
missing-encoder branch in the Explore tab — a control that cannot do what it
offers is worse than no control.

## 8. Testing

`tests/test_map_layout.py`
- `PcaLayout` reproduces the existing `Projection` positions exactly.
- A missing `umap` module leaves `available` False and never raises.
- kNN placement puts a duplicate embedding on top of its twin.
- kNN placement of an entry already in the fit returns its fitted position.
- Procrustes alignment reduces total displacement against a rotated copy.
- Staleness arithmetic, including `fitted_n == 0`.
- A cache round-trips; a cache with a foreign encoder is refused; a corrupt
  cache is refused rather than raising.

`tests/test_map_view.py`
- `cell_argmax` picks the max per cell and skips empty cells.
- It agrees with `cell_means`/`cell_majority` on which cells are occupied.

`tests/test_archive_window_render.py`
- The map renders with thumbnails on, at 16/32/64.
- The atlas reserves before it asks for thumbnails, and the count is bounded by
  the viewport rather than the entry count.
- Layout combo, Relayout button and readout all draw; the combo offers UMAP
  only when it is available.

`tests/test_archive_settings.py`
- The three new fields round-trip.

`tools/measure_map_layout.py`
- kNN(10) preservation for both engines over a real archive, against PCA's
  recorded 27.0% at 500 entries and 3.1% at 13,049. If UMAP does not move that
  substantially, the dependency has not earned its place.

## 9. Documentation

One caveat in `CLAUDE.md` under the map: why UMAP lays out a snapshot and kNN
places the rest, why a refit is seeded and aligned, and that the thumbnail
count is bounded by the viewport rather than the archive.

`requirements.txt` gains `umap-learn`, marked optional in the same breath as
the other optional extras.

## Out of scope

- Replacing the search's `Projection(LATENT_DIMS)`. Different job, different
  instance, and `LATENT_DIMS` must stay small for reasons already recorded.
- A grid atlas with one creature per cell (SOM, or projection-then-assignment).
  Worth revisiting only if the per-cell representative still reads as mush.
- Migrating an existing archive's cached layout between encoders. A cache in
  the wrong space is discarded and refitted.
