"""The naming and collision rules behind the one save dialog.

Pure functions, so what a save is CALLED and which files it touches is
assertable without ImGui or a GPU.
"""
import pytest

from services import save_targets as st


# ---- suggested names ----------------------------------------------------

def test_a_single_tournament_tile_suggests_its_own_number():
    assert st.suggested_name(st.TOURNAMENT_TILE, tiles=(3,)) == "tile3"


def test_several_tiles_suggest_one_base_name():
    """The suffixes come from target_stems, so the box holds the stem the
    user actually edits rather than a name for one of the tiles."""
    assert st.suggested_name(st.TOURNAMENT_TILE, tiles=(1, 4, 9)) == "tournament"


def test_auto_names_carry_the_generation():
    assert st.suggested_name(st.AUTO_BEST, generation=42) == "best_gen0042"
    assert st.suggested_name(st.AUTO_TILE, arg=7, generation=42) == "tile7_gen0042"


def test_an_archive_entry_suggests_its_id():
    assert st.suggested_name(st.ARCHIVE_ENTRY, arg=123) == "archive_000123"


def test_the_live_config_suggests_the_open_project():
    assert st.suggested_name(st.CONFIG, current_project="reef") == "reef"
    assert st.suggested_name(st.CONFIG, current_project="") == ""


# ---- what the dialog says it is about -----------------------------------

@pytest.mark.parametrize("kind,arg,tiles", [
    (st.CONFIG, -1, ()),
    (st.TOURNAMENT_TILE, -1, (2,)),
    (st.TOURNAMENT_TILE, -1, (2, 5)),
    (st.AUTO_BEST, -1, ()),
    (st.AUTO_TILE, 4, ()),
    (st.ARCHIVE_ENTRY, 9, ()),
])
def test_every_subject_names_itself(kind, arg, tiles):
    """The same dialog serves the live sliders and a tournament tile, which
    produce different files - it has to say which."""
    label = st.subject_label(kind, arg, tiles)
    assert label and not label.startswith("None")


def test_the_label_lists_the_tiles_it_will_write():
    label = st.subject_label(st.TOURNAMENT_TILE, tiles=(2, 5))
    assert "2" in label and "5" in label


# ---- which files get written --------------------------------------------

def test_one_tile_writes_exactly_the_name_given():
    assert st.target_stems(st.TOURNAMENT_TILE, "reef", (3,)) == ["reef"]


def test_several_tiles_are_suffixed_so_they_cannot_clobber_each_other():
    """The reported bug was tournament_tile3.json overwriting itself. Sharing
    one name across N tiles would reproduce that inside a single click."""
    stems = st.target_stems(st.TOURNAMENT_TILE, "reef", (1, 4))
    assert stems == ["reef_tile1", "reef_tile4"]
    assert len(set(stems)) == 2


def test_non_tournament_kinds_write_one_file():
    for kind in (st.CONFIG, st.AUTO_BEST, st.AUTO_TILE, st.ARCHIVE_ENTRY):
        assert st.target_stems(kind, "reef") == ["reef"]


def test_an_empty_name_writes_nothing():
    """The dialog disables Save on this, and target_stems agrees, so a blank
    name cannot reach the filesystem by any route."""
    assert st.target_stems(st.CONFIG, "") == []
    assert st.target_stems(st.CONFIG, "   ") == []
    assert st.target_stems(st.TOURNAMENT_TILE, "", (1, 2)) == []


# ---- the name cannot escape the folder ----------------------------------

@pytest.mark.parametrize("raw", [
    "../evil", "../../etc/passwd", "a/b", "a\\b", "c:evil", "we?rd",
    "pipe|d", 'qu"ote', "..", "  ..  ",
])
def test_a_typed_name_cannot_escape_the_configs_folder(tmp_path, raw):
    """What matters is where the file lands, not what the stem looks like:
    ".._evil" is a legal filename and traverses nothing, while "../evil" would
    have written outside the folder and then not appeared in the load menu.
    """
    stems = st.target_stems(st.CONFIG, raw)
    for stem in stems:
        path = (tmp_path / f"{stem}.json").resolve()
        assert path.parent == tmp_path.resolve()
    assert not any(c in s for s in stems for c in '<>:"/\\|?*')


def test_trailing_dots_and_spaces_go():
    """Windows drops them silently, which would make the dialog's existence
    check disagree with what actually landed on disk."""
    assert st.safe_stem("reef. ") == "reef"
    assert st.safe_stem("  reef  ") == "reef"


def test_ordinary_names_are_left_alone():
    for name in ("reef", "my creature", "run-7_final", "café"):
        assert st.safe_stem(name) == name
