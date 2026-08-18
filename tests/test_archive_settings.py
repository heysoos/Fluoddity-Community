"""An archive remembers the settings it was last worked with.

Reopening an archive used to hand you the class defaults, so a 20000-entry
archive came back with the sliders of an empty one.
"""
import json

import pytest

from services.archive_io import ArchiveStore
from state.archive_state import PERSISTED_FIELDS, ArchiveState


# ---- what travels, and what deliberately does not -----------------------

def test_the_tuning_knobs_survive_a_round_trip():
    a = ArchiveState()
    a.alpha = 6.5
    a.min_separation = 0.04
    a.grid = 8
    a.goal_order = "least_matched"
    a.physics_enabled = True

    b = ArchiveState()
    b.apply_settings(a.to_settings())

    assert (b.alpha, b.min_separation, b.grid) == (6.5, 0.04, 8)
    assert b.goal_order == "least_matched"
    assert b.physics_enabled is True


def test_one_shot_commands_are_never_persisted():
    """Persisting a command would REPLAY it on load - opening an archive would
    delete an entry or start a search."""
    for name in ("start_requested", "delete_entry_id", "chase_tile", "pin_tile",
                 "switch_archive_name", "delete_archive_requested",
                 "new_archive_requested", "seed_entry_id"):
        assert name in vars(ArchiveState()), f"{name} moved; update this test"
        assert name not in PERSISTED_FIELDS


def test_opening_an_archive_does_not_resume_a_search():
    assert "enabled" not in PERSISTED_FIELDS
    assert "running" not in PERSISTED_FIELDS


def test_transient_text_is_not_persisted():
    for name in ("warning", "notice", "new_goal_text", "new_archive_name",
                 "confirm_delete_text", "archive_name", "archive_list"):
        assert name not in PERSISTED_FIELDS


def test_the_map_view_travels_with_the_archive():
    """Half of "open it in its last state" is where you were looking."""
    for name in ("map_zoom", "map_center_x", "map_center_y", "sort_by"):
        assert name in PERSISTED_FIELDS


def test_the_gallery_size_and_sort_direction_travel_with_the_archive():
    """A 13000-entry archive wants the list; an empty one wants big tiles."""
    for name in ("thumb_size", "sort_desc"):
        assert name in PERSISTED_FIELDS

    a = ArchiveState()
    a.thumb_size = 24
    a.sort_desc = False
    b = ArchiveState()
    b.apply_settings(a.to_settings())
    assert b.thumb_size == 24
    assert b.sort_desc is False


def test_a_brand_new_archive_state_looks_like_the_old_fixed_gallery():
    """96 was the hardcoded tile size, and descending was the only order."""
    a = ArchiveState()
    assert a.thumb_size == 96
    assert a.sort_desc is True


# ---- tolerance of files written by other versions -----------------------

def test_a_missing_field_keeps_its_current_value():
    """A settings.json written before a field existed must still load."""
    s = ArchiveState()
    s.alpha = 7.0
    applied = s.apply_settings({"min_separation": 0.03})
    assert applied == ["min_separation"]
    assert s.alpha == 7.0


def test_unknown_keys_are_ignored():
    s = ArchiveState()
    s.apply_settings({"not_a_field": 1, "alpha": 3.0})
    assert s.alpha == 3.0
    assert not hasattr(s, "not_a_field")


def test_a_value_of_the_wrong_type_is_skipped_rather_than_raising():
    """This runs during an archive switch; a hand-edited file must not take
    the app down."""
    s = ArchiveState()
    s.apply_settings({"alpha": "banana", "grid": None, "k": 12})
    assert s.alpha == ArchiveState().alpha
    assert s.grid == ArchiveState().grid
    assert s.k == 12


def test_a_bool_field_does_not_come_back_as_an_int():
    """bool is a subclass of int, so a naive int-first coercion would turn
    every checkbox into 0/1."""
    s = ArchiveState()
    s.apply_settings({"physics_enabled": True})
    assert s.physics_enabled is True
    assert isinstance(s.physics_enabled, bool)


# ---- the store -----------------------------------------------------------

def test_settings_land_beside_the_goal_list(tmp_path):
    store = ArchiveStore(tmp_path / "arc")
    store.save_settings({"alpha": 5.0})
    assert store.settings_path.name == "settings.json"
    assert store.settings_path.parent == store.goals_path.parent
    assert json.loads(store.settings_path.read_text())["alpha"] == 5.0


def test_an_archive_with_no_settings_yet_reads_as_empty(tmp_path):
    """{} means "keep what is on screen", so a brand new archive inherits the
    settings you were just using rather than snapping to defaults."""
    store = ArchiveStore(tmp_path / "arc")
    assert store.load_settings() == {}
    s = ArchiveState()
    s.alpha = 6.0
    s.apply_settings(store.load_settings())
    assert s.alpha == 6.0


def test_a_corrupt_settings_file_reads_as_empty(tmp_path):
    store = ArchiveStore(tmp_path / "arc")
    store.settings_path.write_text("{not json")
    assert store.load_settings() == {}


def test_a_settings_file_that_is_not_an_object_reads_as_empty(tmp_path):
    store = ArchiveStore(tmp_path / "arc")
    store.settings_path.write_text("[1, 2, 3]")
    assert store.load_settings() == {}


def test_the_store_round_trips_a_whole_state(tmp_path):
    store = ArchiveStore(tmp_path / "arc")
    a = ArchiveState()
    a.expedition_gens = 77
    a.novelty_share = 0.4
    a.show_browser = True
    store.save_settings(a.to_settings())

    b = ArchiveState()
    b.apply_settings(ArchiveStore(tmp_path / "arc").load_settings())
    assert (b.expedition_gens, b.novelty_share, b.show_browser) == (77, 0.4, True)


def test_two_archives_keep_separate_settings(tmp_path):
    one, two = ArchiveStore(tmp_path / "a"), ArchiveStore(tmp_path / "b")
    a, b = ArchiveState(), ArchiveState()
    a.alpha, b.alpha = 2.0, 8.0
    one.save_settings(a.to_settings())
    two.save_settings(b.to_settings())

    back = ArchiveState()
    back.apply_settings(one.load_settings())
    assert back.alpha == 2.0
    back.apply_settings(two.load_settings())
    assert back.alpha == 8.0


def test_settings_are_not_written_when_persistence_is_disabled(tmp_path):
    """ArchiveStore degrades to a no-op on disk trouble; settings follow."""
    store = ArchiveStore(tmp_path / "arc")
    store.enabled = False
    store.save_settings({"alpha": 5.0})
    assert not store.settings_path.exists()


# ---- the switch ----------------------------------------------------------

def test_restoring_a_different_grid_asks_for_the_grid_to_be_rebuilt(tmp_path):
    """The per-frame configure() push does not rebuild the tournament grid, so
    without this the sliders would read 8 while the sim still ran 4x4."""
    from main import App

    class _App:
        archive_store = ArchiveStore(tmp_path / "arc")

    ui = type("UI", (), {"archive": ArchiveState()})()
    ui.archive.grid = 4
    _App.archive_store.save_settings({"grid": 8})

    App._load_archive_settings(_App(), ui)

    assert ui.archive.grid == 8
    assert ui.archive.grid_changed is True


def test_an_unchanged_grid_does_not_request_a_rebuild(tmp_path):
    from main import App

    class _App:
        archive_store = ArchiveStore(tmp_path / "arc")

    ui = type("UI", (), {"archive": ArchiveState()})()
    _App.archive_store.save_settings({"grid": ui.archive.grid})

    App._load_archive_settings(_App(), ui)

    assert ui.archive.grid_changed is False


def test_loading_with_no_store_is_a_no_op():
    from main import App

    class _App:
        archive_store = None

    ui = type("UI", (), {"archive": ArchiveState()})()
    App._load_archive_settings(_App(), ui)
    App._save_archive_settings(_App(), ui)
    assert ui.archive.alpha == ArchiveState().alpha


def test_the_map_layout_and_atlas_travel_with_the_archive():
    for name in ("map_layout", "map_thumbs", "map_thumb_px"):
        assert name in PERSISTED_FIELDS

    a = ArchiveState()
    a.map_layout = "umap"
    a.map_thumbs = True
    a.map_thumb_px = 48
    b = ArchiveState()
    b.apply_settings(a.to_settings())
    assert (b.map_layout, b.map_thumbs, b.map_thumb_px) == ("umap", True, 48)


def test_a_build_without_umap_opens_every_archive_exactly_as_before():
    """pca is the default, so nothing changes until it is asked for."""
    a = ArchiveState()
    assert a.map_layout == "pca"
    assert a.map_thumbs is False
