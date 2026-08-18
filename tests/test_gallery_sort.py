"""The gallery's display order.

One registry behind the Sort combo, the list's column headers and the sort
itself, so a mode cannot exist in one of the three and not the others.
"""
import pytest

from services.gallery_sort import GALLERY_SORTS, sort_entries


class _Entry:
    def __init__(self, i, novelty=0.0, liveness=0.0, ts=0.0, source="",
                 goal="", layout="", pinned=False):
        self.id = i
        self.novelty = novelty
        self.liveness = liveness
        self.ts = ts
        self.source = source
        self.goal = goal
        self.layout = layout
        self.pinned = pinned


class _Archive:
    signature = "fourier-c10"

    def __init__(self, entries):
        self.entries = entries

    def layout_at(self, i):
        return self.entries[i].layout or self.signature


def _arc():
    return _Archive([
        _Entry(0, novelty=0.1, liveness=0.9, ts=30.0, source="record",
               goal="a cat", layout="mlp-n16-a0", pinned=True),
        _Entry(1, novelty=0.5, liveness=0.2, ts=10.0, source="bootstrap",
               goal="zebra", layout=""),
        _Entry(2, novelty=0.3, liveness=0.5, ts=20.0, source="expedition",
               goal="mist", layout="mlp-n16-a0", pinned=True),
    ])


def _ids(rows):
    return [e.id for _, e in rows]


# ---- the order the gallery has always drawn -----------------------------

def test_the_default_is_novelty_descending():
    assert _ids(sort_entries(_arc(), "novelty", True, False)) == [1, 2, 0]


def test_recency_descending_is_newest_first():
    assert _ids(sort_entries(_arc(), "recency", True, False)) == [0, 2, 1]


def test_liveness_descending_is_liveliest_first():
    assert _ids(sort_entries(_arc(), "liveness", True, False)) == [0, 2, 1]


def test_an_unknown_mode_falls_back_to_novelty():
    """A settings.json from a build with a mode this one lost must still open."""
    assert _ids(sort_entries(_arc(), "nonsense", True, False)) == [1, 2, 0]


# ---- direction ----------------------------------------------------------

def test_ascending_is_the_reverse_of_descending():
    up = _ids(sort_entries(_arc(), "novelty", False, False))
    down = _ids(sort_entries(_arc(), "novelty", True, False))
    assert up == list(reversed(down))


# ---- the new modes ------------------------------------------------------

def test_source_sorts_alphabetically():
    assert _ids(sort_entries(_arc(), "source", False, False)) == [1, 2, 0]


def test_goal_sorts_alphabetically():
    assert _ids(sort_entries(_arc(), "goal", False, False)) == [0, 2, 1]


def test_id_sorts_numerically():
    assert _ids(sort_entries(_arc(), "id", False, False)) == [0, 1, 2]


def test_brain_reads_the_archive_not_the_bare_field():
    """An entry written before the column existed carries an empty layout and
    belongs to the archive's own signature, which is what layout_at answers."""
    rows = sort_entries(_arc(), "brain", False, False)
    assert _ids(rows) == [1, 0, 2]


# ---- the filter ---------------------------------------------------------

def test_pinned_only_drops_the_rest():
    assert _ids(sort_entries(_arc(), "novelty", True, True)) == [2, 0]


def test_the_row_index_is_the_archive_row_not_the_display_position():
    rows = sort_entries(_arc(), "novelty", True, True)
    assert [i for i, _ in rows] == [2, 0]


# ---- the registry -------------------------------------------------------

@pytest.mark.parametrize("mode", sorted(GALLERY_SORTS))
def test_every_declared_mode_actually_sorts(mode):
    """Derived from the registry, so a mode added later cannot skip this."""
    rows = sort_entries(_arc(), mode, True, False)
    assert len(rows) == 3
    assert sorted(e.id for _, e in rows) == [0, 1, 2]


@pytest.mark.parametrize("mode", sorted(GALLERY_SORTS))
def test_every_declared_mode_has_a_label(mode):
    label = GALLERY_SORTS[mode].label
    assert label and label[0].isupper()
