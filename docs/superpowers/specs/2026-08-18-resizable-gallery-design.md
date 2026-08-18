# Resizable gallery thumbnails, and a list mode at the small end

Date: 2026-08-18

## The problem

`ui/archive_window.py` draws the gallery at a fixed `_THUMB = 96.0`. One size
serves every archive, and neither end of the range is reachable: a 13000-entry
archive shows about twenty entries at a time, and an entry's brain, source and
goal are only ever answered one selection at a time, in the panel below.

## What this adds

A continuous size slider over the gallery. Above a threshold it scales the
existing grid; at and below it the gallery becomes a sortable, column-toggled
detail table.

## 1. The control

A row in the gallery header, beside Sort and Pinned only:

```
[list] [=====|=======] [grid]   Sort [Novelty v] [v]   [x] Pinned only
```

- `ArchiveState.thumb_size`, an int, slider range **16..160**, default **96**.
  96 is today's `_THUMB`, so an archive opened for the first time is unchanged.
  160 is `THUMB_PX` in `services/archive_io.py`; past it the image is upscaled
  and there is nothing more to see.
- Both icons are drawn into the window draw list at frame height, the pattern
  `ui/slider_widgets.py` already uses. Left is three horizontal bars, right is
  a 2x2 of squares.
- Both icons are `invisible_button`s that snap the slider to their end. This is
  the only one-gesture route to list mode.
- The slider is `##`-labelled, so it contributes nothing to `WIDEST_LABEL` in
  `tests/test_label_widths.py`.

## 2. The two draw modes

The threshold is **48**. `thumb_size > 48` draws the grid; `<= 48` draws the
list.

### Grid (49..160)

Today's code, with `self._THUMB` replaced by `ast.thumb_size`. Same
`begin_child("gallery", (0, 360))`, same `ListClipper`, same wrap-to-window
`per_row`, same hover and click, same tooltip.

### List (16..48)

An ImGui table, 360 tall, in place of the grid's child window - the table
carries its own scrolling rather than nesting inside one:

- Flags `hideable | reorderable | resizable | sortable | row_bg | scroll_y`,
  with `table_setup_scroll_freeze(0, 1)` so the headers stay put.
- Columns: **thumb | # | Source | Brain | Novelty | Goal**. All visible by
  default.
- The thumb column is `width_fixed` at `thumb_size` and `no_sort`. Row height
  follows the slider, so dragging further left fits more rows on screen; at 16
  the thumbnail is still a colour swatch.
- Right-click any header for the show/hide menu. ImGui writes column
  visibility, width and order to `imgui.ini` by itself, so that choice persists
  with no new state and no new file.
- Each row is a `selectable` with `span_all_columns | allow_overlap` in the
  first cell: the whole row is the hit target, and the current selection is
  highlighted, which the grid has never shown.
- A `ListClipper` over rows, as the grid already uses.
- No tooltip. The columns carry the same facts the grid's tooltip does, and a
  tooltip over a row that is already legible is noise.

Hover and click semantics are the grid's exactly: hover writes
`ast.preview_entry_id`, click writes `ast.selected_entry_id` and
`ast.load_entry_id` when Live preview is on. The preview machinery underneath
is untouched.

The selection panel below the gallery (Save as config, Seed a run from here,
Delete) is shared by both modes and unchanged.

## 3. Sorting

`ast.sort_by` grows from three values to seven:
`novelty, recency, liveness, id, source, brain, goal`. A new
`ast.sort_desc: bool = True` carries the direction.

`_sorted_entries` today negates its keys to sort descending. It becomes a plain
key plus `reverse=ast.sort_desc`, which reproduces the current order exactly at
the defaults: novelty descending, recency (`ts`) descending, liveness
descending. `brain` reads `arc.layout_at(i)`. The cache key gains `sort_desc`,
so it still recomputes only when the archive or the order changes.

Both controls write the one field, so switching list<->grid never reorders
anything:

- **List**: `table_get_sort_specs()` is read each frame; when `specs_dirty`, the
  column index maps to a `sort_by` name and the sort direction to `sort_desc`,
  then dirty is cleared.
- **Grid**: the combo gains the four new entries, and an `arrow_button` beside
  it flips `sort_desc` - pointing down when `sort_desc` is True, which is the
  default and today's order.

## 4. The cache

`ThumbCache.reserve(n)` sets `capacity = max(floor, min(n, ceiling))`, where
the floor is the 256 it was constructed with and the ceiling is 1024 (about
105 MB of 160px RGBA). The gallery calls it once per frame with the exact tile
count it is about to draw, before drawing any of them. Capacity only takes
effect in the eviction loop inside `get`, so raising it mid-frame cannot drop a
texture the frame still needs.

The 48px grid floor is what makes that ceiling sufficient. Below the threshold
one thumbnail is drawn per row and the clipper caps that near 30 however the
window is sized; above it, a very wide window tops out near 280. Without the
floor, a 32px grid in a wide window asks for 400+ per frame, and a frame that
touches more than `capacity` evicts every texture and re-decodes the whole
visible set on the next frame, forever.

## 5. State

Two new fields in `state/archive_state.py`:

```python
thumb_size: int = 96      # gallery tile / list row height in px
sort_desc: bool = True
```

Both named in `PERSISTED_FIELDS`, in the browser-and-map group. `apply_settings`
already coerces int and bool. Per-archive, following `sort_by` and the map
settings: a 13000-entry archive wants the list and an empty one wants big tiles.

`_THUMB` is deleted; its 96 becomes the dataclass default.

## 6. Testing

`tests/test_archive_window_render.py`

- The list renders at 16, 32 and 48.
- The grid renders at 49, 96 and 160.
- Every one of the seven sort modes renders, in both modes.
- The flip is real: spy on `begin_table` and assert the gallery table id appears
  at 32 and does not at 96.
- The existing `id_clashes` test covers the new widgets already.

`tests/test_thumb_cache.py`

- `reserve` raises capacity.
- `reserve` never drops capacity below the construction floor.
- `reserve` clamps at the ceiling.
- After `reserve(n)`, getting `n <= ceiling` distinct keys evicts nothing.

`tests/test_archive_settings.py`

- `thumb_size` and `sort_desc` round-trip.

## 7. Documentation

One caveat in `CLAUDE.md`, under UI and platform: the grid's 48px floor and
`ThumbCache.reserve` exist together, and removing either brings the cache thrash
back.

## Out of scope

- Making the 360-tall gallery region resizable.
- Filtering the gallery by goal or source. Those exist on the Map, and this
  change adds sorting rather than a second filter UI.
- Regenerating thumbnails at a second resolution. 160px is the one stored size;
  the slider scales what is already on disk.
