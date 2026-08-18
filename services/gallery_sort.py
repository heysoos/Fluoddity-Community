"""The archive gallery's display order.

One registry behind the Sort combo, the list view's column headers and the sort
itself. Both controls write `ArchiveState.sort_by` and `sort_desc`, so
switching between the grid and the list never reorders anything.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class GallerySort:
    label: str
    # (archive, row index, entry) -> the value to order on.
    key: Callable


GALLERY_SORTS: dict[str, GallerySort] = {
    "novelty": GallerySort("Novelty", lambda a, i, e: e.novelty),
    "recency": GallerySort("Recency", lambda a, i, e: e.ts),
    "liveness": GallerySort("Liveness", lambda a, i, e: e.liveness),
    "id": GallerySort("Id", lambda a, i, e: e.id),
    "source": GallerySort("Source", lambda a, i, e: e.source),
    # layout_at, not the bare field: an entry written before that column
    # existed carries an empty layout and belongs to the archive's signature.
    "brain": GallerySort("Brain", lambda a, i, e: a.layout_at(i)),
    "goal": GallerySort("Goal", lambda a, i, e: e.goal),
}

DEFAULT_SORT = "novelty"


def sort_entries(archive, sort_by: str, sort_desc: bool,
                 pinned_only: bool) -> list[tuple]:
    """-> [(archive row index, entry)] in display order.

    The index is the archive's own row, which is how every command addresses an
    entry; the position in this list is only where it is drawn.
    """
    rows = list(enumerate(archive.entries))
    if pinned_only:
        rows = [(i, e) for i, e in rows if e.pinned]
    key = GALLERY_SORTS.get(sort_by, GALLERY_SORTS[DEFAULT_SORT]).key
    return sorted(rows, key=lambda p: key(archive, p[0], p[1]),
                  reverse=bool(sort_desc))
